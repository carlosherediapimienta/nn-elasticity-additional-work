# src/benchmarks/demand_mlp.py
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class _TrainOnlyEncoder:
    def fit(self, values: pd.Series) -> "_TrainOnlyEncoder":
        self.categories_ = pd.Index(sorted(values.unique()))
        self._map = {v: i for i, v in enumerate(self.categories_)}
        self.n_categories = len(self.categories_) + 1  # unknown
        return self

    def transform(self, values: pd.Series) -> np.ndarray:
        unknown = len(self.categories_)
        return values.map(self._map).fillna(unknown).astype(np.int64).to_numpy()


class DemandMLP(nn.Module):
    def __init__(self, n_controls, n_stores, n_products, hidden=(64, 32),
                 act="gelu", dropout=0.0, d_store=8, d_product=8):
        super().__init__()
        act_fn = {"gelu": nn.GELU, "tanh": nn.Tanh, "relu": nn.ReLU}[act]
        self.emb_store = nn.Embedding(n_stores, d_store)
        self.emb_product = nn.Embedding(n_products, d_product)
        dims = [2 + n_controls + d_store + 2 * d_product] + list(hidden)
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), act_fn(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, controls, store_idx, product_i_idx, product_j_idx):
        inp = torch.cat(
            [x, controls, self.emb_store(store_idx),
             self.emb_product(product_i_idx), self.emb_product(product_j_idx)],
            dim=-1,
        )
        return self.net(inp).squeeze(-1)


class DemandMLPPipeline:
    """One global model on all dyadic rows. Elasticities via autodiff on val."""

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
        d_store: int = 8,
        d_product: int = 8,
        device: str | None = None,
        seed: int = 42,
    ):
        self.control_cols = control_cols
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
        self.d_product = d_product
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        torch.manual_seed(self.seed)
        needed = ["store_code", "pair_id", "product_i", "product_j",
                  "week_id", "log_v_i", "log_p_i", "log_p_j"] + self.control_cols
        train_df = train_df[needed].dropna().reset_index(drop=True)
        val_df = val_df[needed].dropna().reset_index(drop=True)

        store_enc = _TrainOnlyEncoder().fit(train_df["store_code"])
        product_enc = _TrainOnlyEncoder().fit(
            pd.concat([train_df["product_i"], train_df["product_j"]], ignore_index=True)
        )

        def cat_tensors(df):
            return (
                torch.tensor(store_enc.transform(df["store_code"]), dtype=torch.long),
                torch.tensor(product_enc.transform(df["product_i"]), dtype=torch.long),
                torch.tensor(product_enc.transform(df["product_j"]), dtype=torch.long),
            )

        cont_cols = ["log_p_i", "log_p_j"] + self.control_cols
        mean = train_df[cont_cols].mean()
        std = train_df[cont_cols].std().replace(0, 1.0)

        def cont_tensor(df):
            z = (df[cont_cols] - mean) / std
            return torch.tensor(z.to_numpy(), dtype=torch.float32)

        device = self.device
        Xtr = cont_tensor(train_df).to(device)
        Xval = cont_tensor(val_df).to(device)
        store_tr, i_tr, j_tr = (t.to(device) for t in cat_tensors(train_df))
        store_val, i_val, j_val = (t.to(device) for t in cat_tensors(val_df))
        ytr = torch.tensor(train_df["log_v_i"].to_numpy(), dtype=torch.float32, device=device)
        yval = torch.tensor(val_df["log_v_i"].to_numpy(), dtype=torch.float32, device=device)

        model = DemandMLP(
            n_controls=len(self.control_cols),
            n_stores=store_enc.n_categories,
            n_products=product_enc.n_categories,
            hidden=self.hidden, act=self.act, dropout=self.dropout,
            d_store=self.d_store, d_product=self.d_product,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        huber = nn.HuberLoss(delta=self.huber_delta)
        best_val, patience, best_state = float("inf"), 0, None
        n_train = len(train_df)

        for _ in range(self.n_epochs):
            model.train()
            perm = torch.randperm(n_train, device=device)
            for start in range(0, n_train, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                y_hat = model(Xtr[idx, :2], Xtr[idx, 2:], store_tr[idx], i_tr[idx], j_tr[idx])
                huber(y_hat, ytr[idx]).backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                val_loss = huber(
                    model(Xval[:, :2], Xval[:, 2:], store_val, i_val, j_val), yval
                ).item()
            if val_loss < best_val:
                best_val, patience = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= self.es_patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            y_hat_val = model(Xval[:, :2], Xval[:, 2:], store_val, i_val, j_val)
        resid = (yval - y_hat_val).cpu().numpy()
        ynp = yval.cpu().numpy()
        ss_tot = float(np.sum((ynp - ynp.mean()) ** 2))
        metrics = {
            "mae_val": float(np.mean(np.abs(resid))),
            "rmse_val": float(np.sqrt(np.mean(resid ** 2))),
            "r2_val": np.nan if ss_tot == 0 else float(1 - np.sum(resid ** 2) / ss_tot),
            "best_val_loss": best_val,
        }

        x_grad = Xval[:, :2].clone().requires_grad_(True)
        y_hat = model(x_grad, Xval[:, 2:], store_val, i_val, j_val)
        grad, = torch.autograd.grad(y_hat.sum(), x_grad)
        own = (grad[:, 0] / float(std["log_p_i"])).detach().cpu().numpy()
        cross = (grad[:, 1] / float(std["log_p_j"])).detach().cpu().numpy()

        elasticities = pd.DataFrame({
            "store_code": val_df["store_code"].to_numpy(),
            "pair_id": val_df["pair_id"].to_numpy(),
            "product_i": val_df["product_i"].to_numpy(),
            "product_j": val_df["product_j"].to_numpy(),
            "week_id": val_df["week_id"].to_numpy(),
            "own_elasticity": own,
            "cross_elasticity": cross,
            "y_true_i": val_df["log_v_i"].to_numpy(),
            "y_hat_i": y_hat.detach().cpu().numpy(),
        })
        return metrics, elasticities