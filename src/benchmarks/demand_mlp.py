# src/benchmarks/demand_mlp.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _TrainOnlyEncoder:
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
    """One row per (store, week): prices, ICDN features, mask, log-demands."""

    def __init__(self, control_cols: list[str], products: list):
        self.control_cols = list(control_cols)
        self.products = list(products)

    def build(self, df: pd.DataFrame) -> dict:
        keys = ["store_code", "week_id", "product_code"]
        cols = keys + ["log_demand", "log_price"] + self.control_cols
        base = df[cols].drop_duplicates(keys).copy()

        y = _pivot_slot(base, self.products, "log_demand")
        u = _pivot_slot(base, self.products, "log_price")
        mask = y.notna()
        index = y.index

        x_blocks = []
        for c in self.control_cols:
            block = _pivot_slot(base, self.products, c).reindex(index)
            x_blocks.append(block.to_numpy(dtype=float))
        X = np.concatenate(x_blocks, axis=1) if x_blocks else np.zeros((len(index), 0))

        return {
            "store_code": index.get_level_values("store_code").to_numpy(),
            "week_id": index.get_level_values("week_id").to_numpy(),
            "u": u.reindex(index).to_numpy(dtype=float),
            "y": y.to_numpy(dtype=float),
            "mask": mask.reindex(index).to_numpy(dtype=bool),
            "X": X,
        }


class MultiproductMLP(nn.Module):
    """Dense MLP: z = [u, vec(X), m, store_emb] → J log-demands."""

    def __init__(self, n_products, n_controls, n_stores, hidden=(64, 32),
                 act="gelu", dropout=0.0, d_store=8):
        super().__init__()
        self.n_products = n_products
        self.emb_store = nn.Embedding(n_stores, d_store)
        act_fn = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        in_dim = n_products * (2 + n_controls) + d_store
        dims = [in_dim] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act_fn(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_products))
        self.net = nn.Sequential(*layers)

    def forward(self, u, X, m, store_idx):
        z = torch.cat([u, X, m, self.emb_store(store_idx)], dim=-1)
        return self.net(z)


class DemandMLPPipeline:
    """Feature-matched multiproduct MLP. Elasticities = Jacobian on val."""

    def __init__(
        self,
        control_cols: list[str],
        hidden: tuple = (64, 32),
        act: str = "gelu",
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        batch_size: int = 256,
        n_epochs: int = 200,
        es_patience: int = 25,
        huber_delta: float = 1.0,
        d_store: int = 16,
        device: str | None = None,
        seed: int = 42,
    ):
        self.control_cols = list(control_cols)
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
    def _zfill(arr, mu, sd, mask):
        z = (arr - mu) / sd
        z = np.where(mask, z, 0.0)
        return np.nan_to_num(z, nan=0.0).astype(np.float32)

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        torch.manual_seed(self.seed)
        products = sorted(train_df["product_code"].astype(str).unique())
        J = len(products)
        builder = MarketDatasetBuilder(self.control_cols, products)
        tr = builder.build(train_df)
        va = builder.build(val_df)

        store_enc = _TrainOnlyEncoder().fit(pd.Series(tr["store_code"]))
        n_ctrl = len(self.control_cols)
        mask_x_tr = np.tile(tr["mask"], (1, n_ctrl)) if n_ctrl else tr["mask"]
        mask_x_va = np.tile(va["mask"], (1, n_ctrl)) if n_ctrl else va["mask"]

        u_mu, u_sd = self._masked_moments(tr["u"], tr["mask"])
        x_mu, x_sd = self._masked_moments(tr["X"], mask_x_tr)

        u_tr = self._zfill(tr["u"], u_mu, u_sd, tr["mask"])
        u_va = self._zfill(va["u"], u_mu, u_sd, va["mask"])
        X_tr = self._zfill(tr["X"], x_mu, x_sd, mask_x_tr)
        X_va = self._zfill(va["X"], x_mu, x_sd, mask_x_va)
        m_tr = tr["mask"].astype(np.float32)
        m_va = va["mask"].astype(np.float32)
        y_tr, y_va = tr["y"], va["y"]

        device = self.device

        def pack(u, X, m, stores, y, mask):
            return (
                torch.tensor(u, dtype=torch.float32, device=device),
                torch.tensor(X, dtype=torch.float32, device=device),
                torch.tensor(m, dtype=torch.float32, device=device),
                torch.tensor(store_enc.transform(pd.Series(stores)), dtype=torch.long, device=device),
                torch.tensor(np.nan_to_num(y, nan=0.0), dtype=torch.float32, device=device),
                torch.tensor(mask, dtype=torch.float32, device=device),
            )

        Utr, Xtr, Mtr, Str, Ytr, Wtr = pack(
            u_tr, X_tr, m_tr, tr["store_code"], y_tr, m_tr
        )
        Uva, Xva, Mva, Sva, Yva, Wva = pack(
            u_va, X_va, m_va, va["store_code"], y_va, m_va
        )

        model = MultiproductMLP(
            n_products=J, n_controls=n_ctrl, n_stores=store_enc.n_categories,
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
                y_hat = model(Utr[idx], Xtr[idx], Mtr[idx], Str[idx])
                loss = masked_loss(y_hat, Ytr[idx], Wtr[idx])
                loss.backward()
                opt.step()
                train_sum += float(loss.item()) * int(Wtr[idx].sum().item())
                n_seen += int(Wtr[idx].sum().item())
            train_loss = train_sum / max(n_seen, 1)

            model.eval()
            with torch.no_grad():
                val_loss = float(masked_loss(model(Uva, Xva, Mva, Sva), Yva, Wva).item())
            if epoch == 1 or epoch % 10 == 0:
                print(f"epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val:
                best_val, patience = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.es_patience:
                    print(f"early stop at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_hat_val = model(Uva, Xva, Mva, Sva)

        resid = (Yva - y_hat_val).detach().cpu().numpy()
        w = Wva.detach().cpu().numpy() > 0
        r = resid[w]
        ynp = Yva.detach().cpu().numpy()[w]
        ss_tot = float(np.sum((ynp - ynp.mean()) ** 2)) if ynp.size else 0.0
        metrics = {
            "mae_val": float(np.mean(np.abs(r))) if r.size else np.nan,
            "rmse_val": float(np.sqrt(np.mean(r ** 2))) if r.size else np.nan,
            "r2_val": np.nan if ss_tot == 0 else float(1 - np.sum(r ** 2) / ss_tot),
            "best_val_loss": best_val,
        }

        u_grad = Uva.clone().requires_grad_(True)
        y_hat = model(u_grad, Xva, Mva, Sva)
        rows = []
        u_sd_t = torch.tensor(u_sd, dtype=torch.float32, device=device)
        y_np = y_hat.detach().cpu().numpy()
        y_true = va["y"]
        mask_np = va["mask"]
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
                        "store_code": va["store_code"][b],
                        "week_id": va["week_id"][b],
                        "product_i": products[i],
                        "product_j": products[j],
                        "own_elasticity": e_i[b, i] if i == j else np.nan,
                        "cross_elasticity": e_i[b, j] if i != j else np.nan,
                        "elasticity": e_i[b, j],
                        "kind": "own" if i == j else "cross",
                        "y_true_i": y_true[b, i],
                        "y_hat_i": y_np[b, i],
                    })
        elasticities = pd.DataFrame(rows)
        return metrics, elasticities