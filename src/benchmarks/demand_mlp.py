"""Feature-matched multiproduct MLP.

One (store, week) row → J log-demands. Inputs are log prices, shared
store-week features once, vectorized product features, and a store embedding.
The observation mask is used only in the loss, metrics, and elasticity filter —
it is not concatenated into z.

Elasticities are Jacobian entries ∂ log q_i / ∂ log p_j on evaluate(),
rescaled from standardized prices back to log-log units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.benchmarks.constants import MLP_MAX_EPOCHS, MLP_PATIENCE
from src.benchmarks.prices import CausalPriceFill
from src.benchmarks.universe import (
    allow_missing_validation_products,
    assert_layout,
    require_training_products,
)


class _TrainOnlyEncoder:
    """Integer store ids from train. Unseen stores map to an extra unknown index."""
    def fit(self, values: pd.Series) -> "_TrainOnlyEncoder":
        cats = pd.Index(sorted(values.astype("object").unique()))
        self.categories_ = cats
        self._map = {v: i for i, v in enumerate(cats)}
        self.n_categories = len(cats) + 1
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        unknown = len(self.categories_)
        mapped = values.astype("object").map(self._map)
        return mapped.fillna(unknown).astype(np.int64).to_numpy()


def _pivot_slot(df, products, value):
    wide = df.pivot(index=["store_code", "week_id"], columns="product_code", values=value)
    return wide.reindex(columns=products)


class MarketDatasetBuilder:
    """One row per (store, week): prices, shared feats once, product feats, mask, log-demands."""

    def __init__(self, shared_cols: list[str], product_cols: list[str], products: list):
        self.shared_cols = list(shared_cols)
        self.product_cols = list(product_cols)
        self.products = list(products)

    def build(self, df: pd.DataFrame) -> dict:
        keys = ["store_code", "week_id", "product_code"]
        cols = keys + ["log_demand", "log_price"] + self.shared_cols + self.product_cols
        base = df[cols].drop_duplicates(keys).copy()
        base["product_code"] = base["product_code"].astype(str)

        y = _pivot_slot(base, self.products, "log_demand")
        u = _pivot_slot(base, self.products, "log_price")
        mask = y.notna()
        index = y.index

        if self.shared_cols:
            Xs = (
                base.groupby(["store_code", "week_id"], observed=True)[self.shared_cols]
                .first()
                .reindex(index)
                .to_numpy(dtype=float)
            )
        else:
            Xs = np.zeros((len(index), 0))

        xp_blocks = []
        for c in self.product_cols:
            block = _pivot_slot(base, self.products, c).reindex(index)
            xp_blocks.append(block.to_numpy(dtype=float))
        Xp = np.concatenate(xp_blocks, axis=1) if xp_blocks else np.zeros((len(index), 0))

        return {
            "store_code": index.get_level_values("store_code").to_numpy(),
            "week_id": index.get_level_values("week_id").to_numpy(),
            "u": u.reindex(index).to_numpy(dtype=float),
            "y": y.to_numpy(dtype=float),
            "mask": mask.reindex(index).to_numpy(dtype=bool),
            "Xs": Xs,
            "Xp": Xp,
        }


class MultiproductMLP(nn.Module):
    """Dense MLP: z = [u, X_shared, vec(X_product), store_emb] → J log-demands."""

    def __init__(self, n_products, n_shared, n_product, n_stores, hidden=(64, 32),
                 act="gelu", dropout=0.0, d_store=8):
        super().__init__()
        self.n_products = n_products
        self.emb_store = nn.Embedding(n_stores, d_store)
        act_fn = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        in_dim = n_products * (1 + n_product) + n_shared + d_store
        dims = [in_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act_fn(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_products))
        self.net = nn.Sequential(*layers)

    def forward(self, u, X, store_idx):
        z = torch.cat([u, X, self.emb_store(store_idx)], dim=-1)
        return self.net(z)


class DemandMLPPipeline:
    """Feature-matched multiproduct MLP. Elasticities = Jacobian on evaluate()."""

    def __init__(
        self,
        shared_cols: list[str],
        product_cols: list[str],
        products: list[str],
        hidden: tuple = (64, 32),
        act: str = "gelu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        n_epochs: int = MLP_MAX_EPOCHS,
        es_patience: int = MLP_PATIENCE,
        huber_delta: float = 1.0,
        d_store: int = 16,
        device: str | None = None,
        seed: int = 42,
    ):
        self.shared_cols = list(shared_cols)
        self.product_cols = list(product_cols)
        self.products = [str(p) for p in products]
        self.hidden = hidden
        self.act = act
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.es_patience = es_patience
        self.huber_delta = huber_delta
        self.d_store = d_store
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        self._fitted = False

    @staticmethod
    def _masked_moments(arr, mask):
        arr = np.asarray(arr, dtype=float)
        mask = np.asarray(mask, dtype=bool)
        mu = np.zeros(arr.shape[1], dtype=float)
        sd = np.ones(arr.shape[1], dtype=float)
        for k in range(arr.shape[1]):
            v = arr[:, k][mask[:, k]] if mask.ndim == 2 else arr[:, k][mask]
            if v.size == 0 or not np.isfinite(v).any():
                continue
            v = v[np.isfinite(v)]
            mu[k] = v.mean()
            s = v.std()
            sd[k] = s if s > 1e-12 else 1.0
        return mu, sd

    @staticmethod
    def _moments(arr):
        arr = np.asarray(arr, dtype=float)
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
        return mu, sd

    @staticmethod
    def _zfill(arr, mu, sd, mask=None):
        z = (arr - mu) / sd
        if mask is not None:
            z = np.where(mask, z, 0.0)
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def _u_frame(self, slot):
        return pd.DataFrame(
            slot["u"],
            index=pd.MultiIndex.from_arrays(
                [slot["store_code"], slot["week_id"]],
                names=["store_code", "week_id"],
            ),
            columns=self.products,
        )

    def _pack(self, slot, u, X, m):
        device = self.device
        return (
            torch.tensor(u, dtype=torch.float32, device=device),
            torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(
                self.store_enc.transform(pd.Series(slot["store_code"])),
                dtype=torch.long, device=device,
            ),
            torch.tensor(np.nan_to_num(slot["y"], nan=0.0), dtype=torch.float32, device=device),
            torch.tensor(m, dtype=torch.float32, device=device),
        )

    def _pack_scaled(self, slot, u_raw):
        u = np.nan_to_num((u_raw - self.u_mu) / self.u_sd, nan=0.0).astype(np.float32)
        Xs = self._zfill(slot["Xs"], self.xs_mu, self.xs_sd)
        n_prod = len(self.product_cols)
        mask_xp = np.tile(slot["mask"], (1, n_prod)) if n_prod else slot["mask"]
        Xp = self._zfill(slot["Xp"], self.xp_mu, self.xp_sd, mask_xp)
        X = np.concatenate([Xs, Xp], axis=1) if Xp.shape[1] else Xs
        m = slot["mask"].astype(np.float32)
        return self._pack(slot, u, X, m)

    def fit(self, train_df: pd.DataFrame, early_stop_df: pd.DataFrame) -> "DemandMLPPipeline":
        """Train on `train_df`; pick the checkpoint from `early_stop_df` only.

        Price fill for training and early stopping is fit on `train_df`.
        `evaluate()` uses a fill updated with `early_stop_df` as well.
        """
        torch.manual_seed(self.seed)
        train_df = require_training_products(train_df, self.products, "mlp train")
        early_stop_df = allow_missing_validation_products(early_stop_df, self.products)
        J = len(self.products)
        self.builder = MarketDatasetBuilder(self.shared_cols, self.product_cols, self.products)
        tr = self.builder.build(train_df)
        es = self.builder.build(early_stop_df)

        self.store_enc = _TrainOnlyEncoder().fit(pd.Series(tr["store_code"]))
        self.n_shared = len(self.shared_cols)
        self.n_product = len(self.product_cols)

        self.price_fill = CausalPriceFill().fit(train_df)
        self.price_fill_infer = CausalPriceFill().fit(
            pd.concat([train_df, early_stop_df], ignore_index=True)
        )
        self.u_mu, self.u_sd = self._masked_moments(tr["u"], np.isfinite(tr["u"]))
        self.xs_mu, self.xs_sd = self._moments(tr["Xs"])
        mask_xp = np.tile(tr["mask"], (1, self.n_product)) if self.n_product else tr["mask"]
        self.xp_mu, self.xp_sd = self._masked_moments(tr["Xp"], mask_xp)

        u_tr = self.price_fill.fill_wide(self._u_frame(tr), panel=train_df).to_numpy(dtype=float)
        u_es = self.price_fill.fill_wide(self._u_frame(es), panel=early_stop_df).to_numpy(dtype=float)
        Utr, Xtr, Str, Ytr, Wtr = self._pack_scaled(tr, u_tr)
        Ues, Xes, Ses, Yes, Wes = self._pack_scaled(es, u_es)

        device = self.device
        model = MultiproductMLP(
            n_products=J, n_shared=self.n_shared, n_product=self.n_product,
            n_stores=self.store_enc.n_categories,
            hidden=self.hidden, act=self.act, dropout=self.dropout,
            d_store=self.d_store,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        huber = nn.HuberLoss(delta=self.huber_delta, reduction="none")
        best_val, patience, best_state = float("inf"), 0, None
        n_train = Utr.shape[0]

        def masked_loss(y_hat, y, w):
            return (huber(y_hat, y) * w).sum() / w.sum().clamp_min(1.0)

        for epoch in range(1, self.n_epochs + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            train_sum, n_seen = 0.0, 0
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                y_hat = model(Utr[idx], Xtr[idx], Str[idx])
                loss = masked_loss(y_hat, Ytr[idx], Wtr[idx])
                loss.backward()
                opt.step()
                train_sum += float(loss.item()) * int(Wtr[idx].sum().item())
                n_seen += int(Wtr[idx].sum().item())
            train_loss = train_sum / max(n_seen, 1)

            model.eval()
            with torch.no_grad():
                es_loss = float(masked_loss(model(Ues, Xes, Ses), Yes, Wes).item())
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  es_loss={es_loss:.4f}")

            if es_loss < best_val:
                best_val, patience = es_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.es_patience:
                    print(f"early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model = model
        self.best_es_loss = best_val
        self._fitted = True
        assert_layout(self.builder.products, self.products, "mlp")
        assert model.n_products == J
        return self

    def evaluate(self, df: pd.DataFrame):
        """MAE/RMSE/R² on observed cells, Jacobian elasticities, and cell predictions."""
        if not self._fitted:
            raise RuntimeError("fit() first")
        df = allow_missing_validation_products(df, self.products)
        slot = self.builder.build(df)
        u_raw = self.price_fill_infer.fill_wide(self._u_frame(slot), panel=df).to_numpy(dtype=float)
        U, X, S, Y, W = self._pack_scaled(slot, u_raw)
        model = self.model
        model.eval()
        with torch.no_grad():
            y_hat_val = model(U, X, S)
        resid = (Y - y_hat_val).detach().cpu().numpy()
        w = W.detach().cpu().numpy() > 0
        r = resid[w]
        ynp = Y.detach().cpu().numpy()[w]
        ss_tot = float(np.sum((ynp - ynp.mean()) ** 2)) if ynp.size else 0.0
        metrics = {
            "mae_val": float(np.mean(np.abs(r))) if r.size else np.nan,
            "rmse_val": float(np.sqrt(np.mean(r ** 2))) if r.size else np.nan,
            "r2_val": np.nan if ss_tot == 0 else float(1 - np.sum(r ** 2) / ss_tot),
            "best_es_loss": self.best_es_loss,
            "n_cells": int(r.size),
        }
        u_grad = U.clone().requires_grad_(True)
        y_hat = model(u_grad, X, S)
        rows = []
        u_sd_t = torch.tensor(self.u_sd, dtype=torch.float32, device=self.device)
        y_np = y_hat.detach().cpu().numpy()
        y_true = slot["y"]
        mask_np = slot["mask"]
        products = self.products
        J = len(products)
        for i in range(J):
            grad_i, = torch.autograd.grad(
                y_hat[:, i].sum(), u_grad, retain_graph=True
            )
            e_i = (grad_i / u_sd_t).detach().cpu().numpy()
            for b in range(e_i.shape[0]):
                if not mask_np[b, i]:
                    continue
                for j in range(J):
                    if not mask_np[b, j]:
                        continue
                    rows.append({
                        "store_code": slot["store_code"][b],
                        "week_id": slot["week_id"][b],
                        "product_i": products[i],
                        "product_j": products[j],
                        "own_elasticity": e_i[b, i] if i == j else np.nan,
                        "cross_elasticity": e_i[b, j] if i != j else np.nan,
                        "elasticity": e_i[b, j],
                        "kind": "own" if i == j else "cross",
                        "observed_i": True,
                        "observed_j": True,
                        "y_true_i": y_true[b, i],
                        "y_hat_i": y_np[b, i],
                    })
        y_hat_np = y_hat_val.detach().cpu().numpy()
        b_idx, i_idx = np.where(mask_np)
        cells = pd.DataFrame({
            "store_code": slot["store_code"][b_idx],
            "product_code": np.asarray(products)[i_idx],
            "week_id": slot["week_id"][b_idx],
            "y_true": y_true[b_idx, i_idx],
            "y_pred": y_hat_np[b_idx, i_idx],
        })
        return metrics, pd.DataFrame(rows), cells