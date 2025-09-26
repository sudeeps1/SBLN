import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Reuse the core model and dataset wrappers (support running from project root or from within sbln/)
try:
    from sbln.CausalSBLN import CausalSBLN, TabularMixedDataset, acyclicity_loss
except Exception:  # pragma: no cover - fallback when executed from sbln directory
    from CausalSBLN import CausalSBLN, TabularMixedDataset, acyclicity_loss

# sklearn iterative imputer (MICE-like)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_parse_to_float(value):
    """Parse strings like '10 (SRL->TAC)' or '3 something' to a float using the first token/segment.
    Returns np.nan on failure, so imputer can handle it."""
    try:
        s = str(value)
        # Keep left of '(' if present
        s = s.split('(')[0]
        # Take first whitespace-separated token
        s = s.strip().split()[0]
        # Empty strings -> NaN
        if s == '' or s.lower() in {'nan', 'none'}:
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def load_excel_subset(
    excel_path: str,
    filter_success_only: bool = True,
    feature_columns: list[str] | None = None,
    target_column: str = "MPA_dose",
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_excel(excel_path)
    if filter_success_only and 'Graftfailure' in df.columns:
        df = df[df['Graftfailure'] != 1]

    if feature_columns is None:
        feature_columns = [
            'INITIAL_MPA', 'INITIAL_MAINIS', 'INITIAL_CNI', 'DONOR_TYPE', 'PRA2', 'PRA1', 'Age(1)',
            'HEIGHT', 'WEIGHT', 'eGFR_DC', 'HLA_MN', 'KT_TYPE', 'sCR_DC', 'INDUCTION_TYPE',
            'INDUCTION_YN', 'GE0DER',
        ]

    # Select X and y
    X = df[feature_columns].copy()
    y = df[target_column].to_numpy()

    # Coerce any non-numeric entries using the robust parser
    for col in X.columns:
        if not np.issubdtype(X[col].dtype, np.number):
            X[col] = X[col].map(robust_parse_to_float)

    # Ensure numeric dtype
    X = X.apply(pd.to_numeric, errors='coerce')
    return X, y


def cross_validate_sbln(
    X_df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = 5,
    batch_size: int = 64,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    grad_clip: float | None = 0.5,
    device: str | None = None,
    num_entities: int = 7,
    hidden_dim: int = 64,
    num_steps: int = 3,
    gumbel_tau: float = 0.7,
    tau_anneal: float = 0.97,
    tau_min: float = 0.4,
    use_plasticity: bool = True,
    # Turn off auxiliary/regularization losses to mirror notebook's pure predictive objective
    acyclicity_lambda: float = 0.0,
    aux_recon_weight: float = 0.0,
    aux_mask_ratio: float = 0.0,
    graph_stability_lambda: float = 0.0,
    edge_entropy_lambda: float = 0.0,
    huber_beta: float = 100.0,
    regression_loss: str = "mse",
    early_stopping_patience: int | None = 10,
    scale_features: bool = False,
    restarts: int = 1,
    scheduler_type: str = "plateau",  # 'plateau' or 'onecycle'
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_df)):
        current_tau = gumbel_tau  # reset per fold for fairness
        X_train = X_df.iloc[train_idx].copy()
        X_val = X_df.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Iterative imputation fit only on the training fold to avoid leakage
        imputer = IterativeImputer(random_state=42, max_iter=10)
        X_train_num = imputer.fit_transform(X_train)
        X_val_num = imputer.transform(X_val)

        # Optional scaling after imputation (fit on train only)
        if scale_features:
            scaler = StandardScaler()
            X_train_num = scaler.fit_transform(X_train_num)
            X_val_num = scaler.transform(X_val_num)

        # Build datasets and loaders (no categorical features for this comparison)
        train_dataset = TabularMixedDataset(X_train_num, np.empty((len(X_train_num), 0), dtype=np.int64), y_train.astype(np.float32))
        val_dataset = TabularMixedDataset(X_val_num, np.empty((len(X_val_num), 0), dtype=np.int64), y_val.astype(np.float32))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        input_dim = X_train_num.shape[1]
        output_dim = 1

        # Ensemble over multiple restarts to reduce variance
        all_val_preds = []
        val_targets_cache = None

        for restart in range(max(1, int(restarts))):
            model = CausalSBLN(
                input_dim=input_dim,
                num_entities=num_entities,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                tau=gumbel_tau,
                num_steps=num_steps,
                use_plasticity=use_plasticity,
            ).to(device)
            model.build_encoders(num_in_features=input_dim, cat_cardinalities=[])

            if regression_loss.lower() == "huber":
                criterion = nn.SmoothL1Loss(beta=float(huber_beta))
            else:
                criterion = nn.MSELoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            if scheduler_type.lower() == "onecycle":
                scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=lr,
                    steps_per_epoch=max(1, len(train_loader)),
                    epochs=epochs,
                    pct_start=0.3,
                    anneal_strategy='cos',
                    div_factor=10.0,
                    final_div_factor=1e3,
                )
                use_plateau = False
            else:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)
                use_plateau = True

            best_val = float('inf')
            epochs_no_improve = 0
            current_tau = gumbel_tau

            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for xb, yb in train_loader:
                    (xb_num, xb_cat) = xb
                    xb_num = xb_num.to(device)
                    xb_cat = xb_cat.to(device) if xb_cat.numel() > 0 else xb_cat
                    yb = yb.to(device)

                    optimizer.zero_grad()
                    out, causal_graph_seq = model((xb_num, xb_cat), tau_override=current_tau)

                    out_s = out.squeeze()
                    yb_s = yb.float()
                    task_loss = criterion(out_s, yb_s)

                    # Optional losses disabled by default for parity
                    if acyclicity_lambda > 0.0:
                        A_last = causal_graph_seq[:, -1, :, :]
                        reg_loss = acyclicity_lambda * acyclicity_loss(A_last)
                    else:
                        reg_loss = torch.tensor(0.0, device=xb_num.device)

                    loss = task_loss + reg_loss
                    loss.backward()
                    if grad_clip is not None and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                    optimizer.step()

                    if not use_plateau:
                        scheduler.step()

                    total_loss += loss.item() * xb_num.size(0)

                # Anneal Gumbel temperature
                current_tau = max(tau_min, current_tau * tau_anneal)

                # Validation
                model.eval()
                val_preds = []
                val_targets = []
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb in val_loader:
                        (xb_num, xb_cat) = xb
                        xb_num = xb_num.to(device)
                        xb_cat = xb_cat.to(device) if xb_cat.numel() > 0 else xb_cat
                        yb = yb.to(device)
                        out, _ = model((xb_num, xb_cat), tau_override=current_tau)
                        out_s = out.squeeze()
                        yb_s = yb.float()
                        val_loss += criterion(out_s, yb_s).item() * xb_num.size(0)
                        val_preds.extend(out_s.detach().cpu().numpy())
                        val_targets.extend(yb_s.detach().cpu().numpy())

                avg_val = val_loss / max(1, len(val_loader.dataset))
                if use_plateau:
                    scheduler.step(avg_val)

                if early_stopping_patience is not None and early_stopping_patience > 0:
                    if avg_val + 1e-6 < best_val:
                        best_val = avg_val
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if epochs_no_improve >= early_stopping_patience:
                            break

            all_val_preds.append(np.asarray(val_preds, dtype=float).reshape(-1))
            if val_targets_cache is None:
                val_targets_cache = np.asarray(val_targets, dtype=float).reshape(-1)

        # Average predictions from all restarts
        y_true = val_targets_cache if val_targets_cache is not None else np.array([])
        y_pred = np.mean(np.stack(all_val_preds, axis=0), axis=0) if len(all_val_preds) > 0 else np.array([])
        mae = mean_absolute_error(y_true, y_pred) if y_true.size > 0 else float('nan')
        rmse = math.sqrt(mean_squared_error(y_true, y_pred)) if y_true.size > 0 else float('nan')
        # Guard for constant targets
        if np.var(y_true) < 1e-12:
            r2 = float('nan')
        else:
            r2 = r2_score(y_true, y_pred)

        fold_metrics.append({"fold": fold_idx + 1, "MAE": mae, "RMSE": rmse, "R2": r2})

    # Aggregate
    mae_vals = [m["MAE"] for m in fold_metrics]
    rmse_vals = [m["RMSE"] for m in fold_metrics]
    r2_vals = [m["R2"] for m in fold_metrics]

    summary = {
        "MAE_mean": float(np.nanmean(mae_vals)),
        "MAE_std": float(np.nanstd(mae_vals, ddof=1)),
        "RMSE_mean": float(np.nanmean(rmse_vals)),
        "RMSE_std": float(np.nanstd(rmse_vals, ddof=1)),
        "R2_mean": float(np.nanmean(r2_vals)),
        "R2_std": float(np.nanstd(r2_vals, ddof=1)),
        "per_fold": fold_metrics,
    }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Cross-validate CausalSBLN with notebook-matching data logic")
    parser.add_argument("--excel_path", type=str, default="input/S1File.xlsx")
    parser.add_argument("--target", type=str, default="MPA_dose")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_plasticity", action="store_true")
    parser.add_argument("--entities", type=int, default=7)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--loss", type=str, default="huber", choices=["mse", "huber"])
    parser.add_argument("--huber_beta", type=float, default=200.0)
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["plateau", "onecycle"])
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Load using notebook-like logic
    X_df, y = load_excel_subset(
        excel_path=args.excel_path,
        filter_success_only=True,
        target_column=args.target,
    )

    results = cross_validate_sbln(
        X_df=X_df,
        y=y,
        n_splits=args.folds,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_entities=args.entities,
        hidden_dim=args.hidden,
        num_steps=args.steps,
        use_plasticity=not args.disable_plasticity,
        regression_loss=args.loss,
        huber_beta=args.huber_beta,
        scale_features=args.scale,
        restarts=args.restarts,
        scheduler_type=args.scheduler,
    )

    os.makedirs("output", exist_ok=True)
    # Save per-fold and summary
    per_fold_df = pd.DataFrame(results["per_fold"])  # type: ignore[index]
    per_fold_df.to_csv("output/sbln_cv_per_fold.csv", index=False)
    pd.DataFrame([{
        "MAE_mean": results["MAE_mean"],
        "MAE_std": results["MAE_std"],
        "RMSE_mean": results["RMSE_mean"],
        "RMSE_std": results["RMSE_std"],
        "R2_mean": results["R2_mean"],
        "R2_std": results["R2_std"],
    }]).to_csv("output/sbln_cv_summary.csv", index=False)

    print("CausalSBLN 5-fold CV (notebook-like data logic)")
    print(per_fold_df)
    print("\nSummary:")
    print(pd.read_csv("output/sbln_cv_summary.csv"))


if __name__ == "__main__":
    main()


