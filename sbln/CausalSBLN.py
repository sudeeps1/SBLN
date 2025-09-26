import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.feature_extraction.text import HashingVectorizer
from collections import Counter

# --- Model and helper from before ---

class CausalSimulationModule(nn.Module):
    """
    Differentiable causal simulator with sparse attention via Gumbel-softmax and
    diagonal masking to discourage self-loops. Messages are computed from source/target
    pairs and aggregated per target node.
    """

    def __init__(self, num_entities, hidden_dim, tau=0.5):
        super().__init__()
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.register_buffer("tau", torch.tensor(float(tau)))

        self.entity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_logits = nn.Parameter(torch.randn(num_entities, num_entities) * 0.01)
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    @torch.no_grad()
    def set_tau(self, tau: float):
        self.tau.fill_(float(tau))

    def get_adjacency(self, deterministic: bool = False, tau: float | None = None):
        logits = self.edge_logits
        # Mask diagonal to discourage self-loops
        diag_mask = torch.eye(self.num_entities, device=logits.device, dtype=torch.bool)
        masked_logits = logits.masked_fill(diag_mask, -1e9)

        temp_tau = self.tau if tau is None else torch.as_tensor(float(tau), device=logits.device)
        if deterministic:
            scores = masked_logits / temp_tau
        else:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(masked_logits) + 1e-9) + 1e-9)
            scores = (masked_logits + gumbel_noise) / temp_tau
        A = F.softmax(scores, dim=-1)
        return A

    def sample_adjacency(self, tau: float | None = None):
        return self.get_adjacency(deterministic=False, tau=tau)

    def forward(self, entity_states, tau: float | None = None):
        B, N, D = entity_states.shape
        assert N == self.num_entities

        projected = self.entity_proj(entity_states)
        A = self.sample_adjacency(tau=tau).unsqueeze(0).expand(B, -1, -1)

        # Compute all pairwise messages in a vectorized way
        # source: [B, N, 1, D] -> broadcast to [B, N, N, D]
        # target: [B, 1, N, D] -> broadcast to [B, N, N, D]
        source = projected.unsqueeze(2).expand(-1, -1, N, -1)
        target = projected.unsqueeze(1).expand(-1, N, -1, -1)
        combined = torch.cat([source, target], dim=-1)  # [B, N, N, 2D]
        message = self.update_mlp(combined)             # [B, N, N, D]

        # Aggregate incoming messages per target j using attention weights A[:, j, i]
        # We want for each target j: sum over sources i of A[j,i] * message[i,j]
        # Rearrange message to [B, N(target=j), N(source=i), D]
        message_t = message.transpose(1, 2)
        weights = A.unsqueeze(-1)  # [B, N, N, 1] with dims [target=j, source=i]
        agg_update = (weights * message_t).sum(dim=2)  # [B, N, D]

        next_state = entity_states + agg_update
        return next_state, A


class PlasticLinear(nn.Module):
    """
    Neuromodulated Hebbian plastic linear layer used as a residual fast-adapting path.
    - Base weights are learned slowly.
    - Fast weights (Hebbian trace) are maintained per-batch during a rollout and updated online.
    y = x W^T + b + (fast ∘ alpha) x, where fast is updated by outer(y, x).
    """

    def __init__(self, in_features: int, out_features: int, init_eta: float = 0.1, decay: float = 0.9):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Learnable gain on the fast weights contribution
        self.alpha = nn.Parameter(torch.full((out_features,), 0.1))
        # Neuromodulator to scale eta per sample
        self.modulator = nn.Sequential(
            nn.Linear(in_features, max(16, in_features // 2)),
            nn.ReLU(),
            nn.Linear(max(16, in_features // 2), 1),
            nn.Sigmoid(),
        )
        self.register_buffer("eta_base", torch.tensor(float(init_eta)))
        self.register_buffer("decay", torch.tensor(float(decay)))

    def forward(self, x, fast_state=None):
        """
        x: [B, in_features]
        fast_state: [B, out_features, in_features] or None
        Returns y, new_fast_state
        """
        B = x.size(0)
        if fast_state is None:
            fast_state = x.new_zeros(B, self.out_features, self.in_features)
        # Use a detached copy for compute to avoid autograd seeing later mutations
        fast_state_compute = fast_state.detach()

        # Base path
        y_base = F.linear(x, self.weight, self.bias)

        # Fast path contribution: for each sample, (fast * x) aggregated over in_features
        x_expanded = x.unsqueeze(1)  # [B, 1, in]
        fast_contrib = (fast_state_compute * x_expanded).sum(dim=-1)  # [B, out]
        phi = self.modulator(x)  # [B, 1] in [0,1]
        y = y_base + (fast_contrib * self.alpha.unsqueeze(0)) * phi

        # Hebbian update with neuromodulation
        with torch.no_grad():
            eta = self.eta_base * phi  # [B, 1]
            outer = y.unsqueeze(-1) * x.unsqueeze(1)  # [B, out, in]
            new_fast_state = fast_state * self.decay + eta.view(B, 1, 1) * outer

        return y, new_fast_state

def acyclicity_loss(A):
    # A can be [B, N, N] or [N, N], handle both cases
    if A.dim() == 3:
        # If A is [B, N, N], compute loss for each batch and average
        batch_size = A.size(0)
        total_loss = 0
        for i in range(batch_size):
            A_i = A[i]  # [N, N]
            expm = torch.matrix_exp(A_i * A_i)
            total_loss += torch.trace(expm) - A_i.size(-1)
        return total_loss / batch_size
    else:
        # If A is [N, N], compute directly
        expm = torch.matrix_exp(A * A)
        return torch.trace(expm) - A.size(-1)

class CausalSBLN(nn.Module):
    """
    Simulation-Based Learning Network with:
    - Causal graph sampling (Gumbel-softmax with diagonal masking)
    - Recurrent temporal refinement
    - Neuromodulated Hebbian plastic residual for fast adaptation
    - Optional foundation adapter and auxiliary feature reconstruction head
    """

    def __init__(
        self,
        input_dim,
        num_entities,
        hidden_dim,
        output_dim,
        tau=0.5,
        foundation_dim=None,
        num_steps: int = 3,
        use_plasticity: bool = True,
        plastic_eta: float = 0.1,
        plastic_decay: float = 0.9,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.use_plasticity = use_plasticity

        # Mixed encoder: numeric linear + categorical embeddings
        self.num_encoder = nn.Linear(0, 0, bias=False)  # placeholder, reset at build time
        self.cat_embeddings = nn.ModuleList()
        self.concat_proj = nn.Linear(0, num_entities * hidden_dim, bias=True)
        self.causal_sim = CausalSimulationModule(num_entities, hidden_dim, tau)
        self.recurrent_sim = nn.GRUCell(hidden_dim, hidden_dim)

        # Plastic residual path operates on flattened entity states per step
        self.plastic_residual = PlasticLinear(hidden_dim, hidden_dim, init_eta=plastic_eta, decay=plastic_decay)

        self.decoder = nn.Sequential(
            nn.Linear(num_entities * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Auxiliary head for masked feature reconstruction (self-supervised)
        self.recon_head = nn.Linear(num_entities * hidden_dim, input_dim)

        if foundation_dim:
            self.foundation_adapter = nn.Linear(foundation_dim, input_dim)
        else:
            self.foundation_adapter = None

    def build_encoders(self, num_in_features: int, cat_cardinalities: list[int] | None):
        num_dim = int(num_in_features)
        cat_dims = 0
        self.num_encoder = nn.Linear(num_dim, max(0, num_dim), bias=True) if num_dim > 0 else nn.Linear(0, 0, bias=False)
        self.cat_embeddings = nn.ModuleList()
        if cat_cardinalities and len(cat_cardinalities) > 0:
            # Use a fixed small embedding size per categorical feature
            emb_size = max(4, min(32, self.hidden_dim // 2))
            for card in cat_cardinalities:
                self.cat_embeddings.append(nn.Embedding(card, emb_size, padding_idx=None))
            cat_dims = emb_size * len(cat_cardinalities)
        in_total = (num_dim if num_dim > 0 else 0) + cat_dims
        if in_total == 0:
            in_total = self.num_entities * self.hidden_dim
        self.concat_proj = nn.Linear(in_total, self.num_entities * self.hidden_dim)

    def simulate_temporal(self, entity_states, fast_state=None, tau: float | None = None):
        B, N, D = entity_states.shape
        states = entity_states
        causal_graphs = []

        # Fast weights state for plastic residual, maintained across steps
        # Allow persistent fast_state across calls for continual/streaming regimes
        if fast_state is None:
            fast_state_local = None
        else:
            fast_state_local = fast_state

        for _ in range(self.num_steps):
            next_states, A = self.causal_sim(states, tau=tau)
            causal_graphs.append(A)

            flat = next_states.view(B * N, D)
            updated = self.recurrent_sim(flat, flat)

            if self.use_plasticity:
                # Apply plastic residual per entity independently
                plast_out, fast_state_local = self.plastic_residual(updated, fast_state_local)
                updated = updated + plast_out

            states = updated.view(B, N, D)

        return states, torch.stack(causal_graphs, dim=1), fast_state_local

    def forward(self, x, tau_override: float = None, fast_state=None):
        # x is either a Tensor [B, input_dim] (legacy) or tuple (x_num, x_cat)
        if isinstance(x, tuple):
            x_num, x_cat = x
        else:
            x_num, x_cat = x, None
        B = x_num.size(0)

        # Numeric path
        if self.num_encoder.in_features > 0:
            num_repr = self.num_encoder(x_num)
        else:
            num_repr = x_num.new_zeros((B, 0))

        # Categorical path
        if x_cat is not None and x_cat.dim() == 2 and len(self.cat_embeddings) == x_cat.size(1):
            cat_embeds = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_embeds.append(emb(x_cat[:, i]))
            cat_repr = torch.cat(cat_embeds, dim=-1) if cat_embeds else x_num.new_zeros((B, 0))
        else:
            cat_repr = x_num.new_zeros((B, 0))

        concat = torch.cat([num_repr, cat_repr], dim=-1) if cat_repr.numel() > 0 or num_repr.numel() > 0 else x_num
        encoded = self.concat_proj(concat)
        entity_states = encoded.view(B, self.num_entities, self.hidden_dim)

        final_states, causal_graph_seq, _ = self.simulate_temporal(entity_states, fast_state=fast_state, tau=tau_override)
        flat_state = final_states.view(B, -1)
        out = self.decoder(flat_state)
        return out, causal_graph_seq

    def reconstruct(self, x, tau_override: float = None, fast_state=None):
        """Auxiliary masked feature reconstruction head."""
        if isinstance(x, tuple):
            x_num, x_cat = x
        else:
            x_num, x_cat = x, None
        B = x_num.size(0)
        if self.num_encoder.in_features > 0:
            num_repr = self.num_encoder(x_num)
        else:
            num_repr = x_num.new_zeros((B, 0))
        if x_cat is not None and len(self.cat_embeddings) == x_cat.size(1):
            cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
            cat_repr = torch.cat(cat_embeds, dim=-1) if cat_embeds else x_num.new_zeros((B, 0))
        else:
            cat_repr = x_num.new_zeros((B, 0))
        concat = torch.cat([num_repr, cat_repr], dim=-1) if cat_repr.numel() > 0 or num_repr.numel() > 0 else x_num
        encoded = self.concat_proj(concat)
        entity_states = encoded.view(B, self.num_entities, self.hidden_dim)
        final_states, _, _ = self.simulate_temporal(entity_states, fast_state=fast_state, tau=tau_override)
        flat_state = final_states.view(B, -1)
        recon = self.recon_head(flat_state)
        return recon

    def stream_step(self, x, tau_override: float = None, fast_state=None):
        """
        Streaming API: returns (out, causal_graph_seq, fast_state_out) enabling persistence
        of Hebbian fast weights across non-i.i.d. sequences or continual learning settings.
        """
        B = x.size(0)
        if self.foundation_adapter:
            x = self.foundation_adapter(x)
        encoded = self.encoder(x)
        entity_states = encoded.view(B, self.num_entities, self.hidden_dim)
        final_states, causal_graph_seq, fast_state_out = self.simulate_temporal(
            entity_states, fast_state=fast_state, tau=tau_override
        )
        flat_state = final_states.view(B, -1)
        out = self.decoder(flat_state)
        return out, causal_graph_seq, fast_state_out

    @torch.no_grad()
    def get_current_adjacency(self, deterministic: bool = True):
        """Return current adjacency matrix [N, N] for interpretability."""
        return self.causal_sim.get_adjacency(deterministic=deterministic).detach().cpu()

    @torch.no_grad()
    def top_causal_edges(self, k: int = 10, deterministic: bool = True):
        """
        Return list of (source, target, weight) sorted by weight descending.
        """
        A = self.get_current_adjacency(deterministic=deterministic)
        N = A.size(0)
        edges = []
        for target in range(N):
            for source in range(N):
                if source == target:
                    continue
                edges.append((int(source), int(target), float(A[target, source].item())))
        edges.sort(key=lambda x: x[2], reverse=True)
        return edges[:k]

    @torch.no_grad()
    def counterfactual_effect(self, x, feature_mask: torch.Tensor, replace_with: float = 0.0, tau_override: float = None):
        """
        Estimate counterfactual effect on output by intervening on a subset of input features.
        - feature_mask: [B, input_dim] binary mask (1 = intervene/replace)
        - replace_with: scalar to fill intervened features
        Returns (y_cf, y_obs, delta)
        """
        if tau_override is not None:
            self.causal_sim.set_tau(tau_override)
        x_obs = x
        x_cf = x * (1 - feature_mask) + replace_with * feature_mask
        y_obs, _ = self.forward(x_obs, tau_override=tau_override)
        y_cf, _ = self.forward(x_cf, tau_override=tau_override)
        delta = y_cf - y_obs
        return y_cf, y_obs, delta

# --- Dataset wrapper with preprocessing ---

class TabularMixedDataset(Dataset):
    """
    Dataset that keeps numeric features as float32 (scaled) and categorical features
    as integer indices for embeddings.
    """
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        # Ensure int64 for embedding
        self.X_cat = torch.tensor(X_cat, dtype=torch.long) if X_cat is not None and X_cat.shape[1] > 0 else torch.zeros((len(self.X_num), 0), dtype=torch.long)
        self.y = y
        if isinstance(y, np.ndarray) and y.dtype.kind in 'iu':
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return (self.X_num[idx], self.X_cat[idx]), self.y[idx]

def load_and_preprocess(csv_path, target_col, exclude_cols=None, hash_seq_cols=True, ngram_range=(3,3), n_hash_features=1024):
    # Robust loader for CSV or whitespace-delimited .data
    if str(csv_path).lower().endswith('.csv'):
        df = pd.read_csv(csv_path)
    else:
        try:
            df = pd.read_csv(csv_path, sep='\s+', header=None)
            # Try to detect header row
            first_row = df.iloc[0].astype(str)
            if any(tok in ' '.join(first_row.tolist()).lower() for tok in ['class', 'id', 'sequence']):
                df = pd.read_csv(csv_path, sep='\s+', header=0)
        except Exception:
            df = pd.read_csv(csv_path, sep='\s+', header=0)
    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Set default exclude columns if none provided
    if exclude_cols is None:
        exclude_cols = []

    # Drop rows with all NaNs in features or target
    df = df.dropna(subset=[target_col])
    df = df.dropna(axis=1, how='all')

    # Separate features and target
    y = df[target_col].values
    X = df.drop(columns=[target_col] + exclude_cols)

    # Optionally hash sequence-like high-cardinality object columns into numeric features
    if hash_seq_cols:
        obj_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        seq_like_cols = []
        n_rows = len(X)
        for col in obj_cols:
            avg_len = X[col].astype(str).str.len().mean()
            nu = X[col].nunique(dropna=True)
            if avg_len > 50 or nu > 0.5 * n_rows or nu > 1000:
                seq_like_cols.append(col)
        for col in seq_like_cols:
            hv = HashingVectorizer(analyzer='char', ngram_range=ngram_range, n_features=n_hash_features, norm=None, alternate_sign=False)
            hashed = hv.transform(X[col].astype(str).values)
            hashed_df = pd.DataFrame(hashed.toarray(), index=X.index, columns=[f"{col}_h{i}" for i in range(hashed.shape[1])])
            # Drop original sequence col and add hashed numeric features
            X = X.drop(columns=[col])
            X = pd.concat([X, hashed_df], axis=1)

    # Detect categorical columns (object or category)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Detect task type automatically
    unique_vals = np.unique(y)
    if y.dtype.kind in 'iufc' and len(unique_vals) > 10:
        task_type = "regression"
    elif len(unique_vals) <= 20:
        # Check if all values are integers (for numeric data)
        if y.dtype.kind in 'iufc':
            if np.all(np.mod(unique_vals, 1) == 0):
                task_type = "classification"
            else:
                task_type = "regression"
        else:
            # For string/categorical data, always treat as classification
            task_type = "classification"
    else:
        # Default fallback
        task_type = "classification"

    return X, y, task_type, cat_cols, num_cols

def preprocess_data(
    X,
    y,
    cat_cols,
    num_cols,
    fit_preprocessors=True,
    scaler=None,
    encoders=None,
    target_encoder=None,
    num_imputer=None,
    cat_imputer=None,
    task_type: str | None = None,
):
    """Preprocess data into numeric (scaled) and categorical (indices) parts.
    Returns: X_num, X_cat, y, scaler, encoders, target_encoder, cat_cardinalities, num_imputer, cat_imputer
    """

    # Imputers
    if fit_preprocessors:
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
    else:
        if num_imputer is None or cat_imputer is None:
            raise ValueError("num_imputer and cat_imputer must be provided when fit_preprocessors=False")

    # Numerical
    if len(num_cols) > 0:
        X_num = num_imputer.fit_transform(X[num_cols]) if fit_preprocessors else num_imputer.transform(X[num_cols])
    else:
        X_num = np.empty((len(X), 0))

    # Scale numeric only
    if fit_preprocessors:
        scaler = StandardScaler()
        if X_num.shape[1] > 0:
            X_num = scaler.fit_transform(X_num)
    else:
        if X_num.shape[1] > 0:
            X_num = scaler.transform(X_num)

    # Categorical -> indices
    if len(cat_cols) > 0:
        X_cat_raw = cat_imputer.fit_transform(X[cat_cols]) if fit_preprocessors else cat_imputer.transform(X[cat_cols])
        if fit_preprocessors:
            encoders = {}
            X_cat = np.zeros_like(X_cat_raw, dtype=np.int64)
            cat_cardinalities = []
            for i, col in enumerate(cat_cols):
                le = LabelEncoder()
                X_cat[:, i] = le.fit_transform(X_cat_raw[:, i])
                encoders[col] = le
                cat_cardinalities.append(len(le.classes_) + 1)  # +1 for unknown bucket
        else:
            X_cat = np.zeros_like(X_cat_raw, dtype=np.int64)
            cat_cardinalities = []
            for i, col in enumerate(cat_cols):
                le = encoders[col]
                try:
                    X_cat[:, i] = le.transform(X_cat_raw[:, i])
                except ValueError:
                    # Map unseen to unknown bucket (len(classes_))
                    # We must map per element
                    mapped = []
                    known = set(le.classes_)
                    for v in X_cat_raw[:, i]:
                        if v in known:
                            mapped.append(int(le.transform([v])[0]))
                        else:
                            mapped.append(len(le.classes_))
                    X_cat[:, i] = np.array(mapped, dtype=np.int64)
                cat_cardinalities.append(len(le.classes_) + 1)
    else:
        X_cat = np.empty((len(X), 0), dtype=np.int64)
        cat_cardinalities = []

    # Encode target only for classification; ensure float for regression
    if task_type == "classification":
        if target_encoder is None:
            if fit_preprocessors:
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
            else:
                target_encoder = LabelEncoder()
                y = target_encoder.transform(y)
        else:
            # Transform using fitted encoder; map any unseen labels to an existing index (0) to avoid errors
            try:
                y = target_encoder.transform(y)
            except ValueError:
                classes = set(target_encoder.classes_)
                mapped = []
                for v in y:
                    if v in classes:
                        mapped.append(int(target_encoder.transform([v])[0]))
                    else:
                        mapped.append(0)
                y = np.array(mapped, dtype=int)
    else:
        # Regression: force numeric float target (even if integers in data)
        y = np.asarray(y, dtype=np.float32)
        target_encoder = None

    return X_num, X_cat, y, scaler, encoders, target_encoder, cat_cardinalities, num_imputer, cat_imputer

# --- Training loop ---

def train_model(
    csv_path,
    target_col,
    exclude_cols=None,
    batch_size=64,
    epochs=50,
    lr=1e-3,
    acyclicity_lambda=0.01,
    device=None,
    gumbel_tau=0.7,
    tau_anneal=0.97,
    aux_recon_weight=0.1,
    aux_mask_ratio=0.15,
    use_plasticity=True,
    graph_stability_lambda=0.05,
    graph_aug_noise=0.005,
    edge_entropy_lambda=0.005,
    num_entities=7,
    hidden_dim=64,
    num_steps=3,
    tau_min=0.4,
    weight_decay=1e-5,
    grad_clip=0.5,
    early_stopping_patience=None,
    regression_loss: str = "mse",  # 'mse' or 'huber'
    huber_beta: float = 100.0,      # mg scale for SmoothL1
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, y, task_type, cat_cols, num_cols = load_and_preprocess(csv_path, target_col, exclude_cols)

    # Train/val split (group-aware if SubjectID present to avoid leakage)
    try:
        df_full = pd.read_csv(csv_path)
        df_full = df_full.dropna(subset=[target_col])
        if 'SubjectID' in df_full.columns:
            groups = df_full['SubjectID'].astype(str).values
            # Try multiple seeds to ensure both classes appear in val for classification
            seeds = [42, 1, 7, 11, 19, 27, 33, 101, 202, 303]
            for rs in seeds:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=rs)
                train_idx, val_idx = next(gss.split(X, y, groups))
                y_val_try = y[val_idx]
                # For classification, require at least 2 classes in val
                if 'classification' in locals() or True:
                    if len(np.unique(y_val_try)) >= 2:
                        break
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess training and validation data
    X_train_num, X_train_cat, y_train_processed, scaler, encoders, target_encoder, cat_cards, num_imputer, cat_imputer = preprocess_data(
        X_train, y_train, cat_cols, num_cols, task_type=task_type
    )
    X_val_num, X_val_cat, y_val_processed, _, _, _, _, _, _ = preprocess_data(
        X_val, y_val, cat_cols, num_cols, fit_preprocessors=False, scaler=scaler, encoders=encoders, target_encoder=target_encoder, num_imputer=num_imputer, cat_imputer=cat_imputer, task_type=task_type
    )

    # Create datasets and loaders
    train_dataset = TabularMixedDataset(X_train_num, X_train_cat, y_train_processed)
    val_dataset = TabularMixedDataset(X_val_num, X_val_cat, y_val_processed)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = (X_train_num.shape[1] if X_train_num is not None else 0) + (0)  # legacy
    output_dim = len(np.unique(y_train_processed)) if task_type == "classification" else 1

    model = CausalSBLN(
        input_dim=input_dim,
        num_entities=num_entities,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        tau=gumbel_tau,
        num_steps=num_steps,
        use_plasticity=use_plasticity,
    ).to(device)
    # Build mixed encoders for numeric and categorical
    model.build_encoders(num_in_features=X_train_num.shape[1], cat_cardinalities=cat_cards)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        if str(regression_loss).lower() == "huber":
            # SmoothL1Loss is Huber; beta controls transition point from L2 to L1
            criterion = nn.SmoothL1Loss(beta=float(huber_beta))
        else:
            criterion = nn.MSELoss()
    recon_criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=False)

    print(f"Training on device: {device}")
    print(f"Task type: {task_type}")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Entities: {num_entities}, Hidden: {hidden_dim}, Steps: {num_steps}")

    current_tau = gumbel_tau

    # Infer common dose step if labels look discretized (for reporting rounded MAE)
    dose_step = None
    try:
        y_train_unique = np.sort(np.unique(y_train))
        if y_train_unique.size > 1 and y_train_unique.size <= 100:
            diffs = np.round(np.diff(y_train_unique), 6)
            diffs = diffs[diffs > 0]
            if diffs.size > 0:
                counts = Counter(diffs.tolist())
                most_common_step, cnt = max(counts.items(), key=lambda kv: kv[1])
                if cnt >= 3 and most_common_step > 0:
                    dose_step = float(most_common_step)
    except Exception:
        dose_step = None

    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for xb, yb in train_loader:
            (xb_num, xb_cat) = xb
            xb_num = xb_num.to(device)
            xb_cat = xb_cat.to(device) if xb_cat.numel() > 0 else xb_cat
            yb = yb.to(device)

            optimizer.zero_grad()

            # Task forward
            out, causal_graph_seq = model((xb_num, xb_cat), tau_override=current_tau)

            # Task loss
            if task_type == "classification":
                task_loss = criterion(out, yb)
            else:
                out_s = out.squeeze()
                yb_s = yb.float()
                task_loss = criterion(out_s, yb_s)

            # Acyclicity regularization on last causal graph of sequence
            A_last = causal_graph_seq[:, -1, :, :]
            reg_loss = acyclicity_lambda * acyclicity_loss(A_last)

            # Auxiliary masked feature reconstruction
            if aux_recon_weight > 0.0 and aux_mask_ratio > 0.0:
                # Mask only numeric features for reconstruction
                if xb_num.numel() > 0:
                    mask = (torch.rand_like(xb_num) < aux_mask_ratio).float()
                    noise = torch.randn_like(xb_num) * 0.01
                    xb_num_masked = xb_num * (1 - mask) + noise * mask
                else:
                    mask = torch.zeros_like(xb_num)
                    xb_num_masked = xb_num
                recon = model.reconstruct((xb_num_masked, xb_cat), tau_override=current_tau)
                recon_loss = recon_criterion(recon * mask, xb_num * mask)
            else:
                recon_loss = torch.tensor(0.0, device=xb.device)

            # Causal graph stability under small input perturbations (consistency regularization)
            if graph_stability_lambda > 0.0:
                xb_num_aug = xb_num + torch.randn_like(xb_num) * graph_aug_noise if xb_num.numel() > 0 else xb_num
                _, causal_graph_seq_aug = model((xb_num_aug, xb_cat), tau_override=current_tau)
                A_last_aug = causal_graph_seq_aug[:, -1, :, :]
                stability_loss = F.l1_loss(A_last_aug, A_last)
            else:
                stability_loss = torch.tensor(0.0, device=xb.device)

            # Attention entropy regularization to encourage sparse, decisive graphs
            if edge_entropy_lambda > 0.0:
                A_eps = (A_last + 1e-9).clamp(min=1e-9)
                entropy = -(A_eps * A_eps.log()).sum(dim=-1).mean()  # average over targets and batch
                entropy_loss = edge_entropy_lambda * entropy
            else:
                entropy_loss = torch.tensor(0.0, device=xb.device)

            loss = task_loss + reg_loss + aux_recon_weight * recon_loss + graph_stability_lambda * stability_loss + entropy_loss
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            total_loss += loss.item() * xb_num.size(0)

        # Anneal Gumbel temperature
        current_tau = max(tau_min, current_tau * tau_anneal)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f} - tau: {current_tau:.3f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                (xb_num, xb_cat) = xb
                xb_num = xb_num.to(device)
                xb_cat = xb_cat.to(device) if xb_cat.numel() > 0 else xb_cat
                yb = yb.to(device)
                out, _ = model((xb_num, xb_cat), tau_override=current_tau)

                if task_type == "classification":
                    val_loss += criterion(out, yb).item() * xb_num.size(0)
                    preds = out.argmax(dim=1)
                    correct += (preds == yb).sum().item()
                    total += xb_num.size(0)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(yb.cpu().numpy())
                else:
                    out_s = out.squeeze()
                    yb_s = yb.float()
                    val_loss += criterion(out_s, yb_s).item() * xb_num.size(0)
                    all_preds.extend(out_s.cpu().numpy())
                    all_labels.extend(yb_s.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader.dataset)
        if task_type == "classification":
            accuracy = correct / max(1, total)
            f1 = f1_score(all_labels, all_preds, average='weighted') if total > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs} - Val loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        else:
            # Ensure proper 1D numpy arrays
            y_true = np.asarray(all_labels, dtype=float).reshape(-1)
            y_pred = np.asarray(all_preds, dtype=float).reshape(-1)

            # Guard against near-constant targets for R^2
            y_var = np.var(y_true)
            if y_var < 1e-12:
                r2 = float("nan")
            else:
                r2 = r2_score(y_true, y_pred)

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # Scale context: baseline and normalized errors
            y_mean = float(np.mean(y_true))
            y_std = float(np.std(y_true))
            y_min = float(np.min(y_true))
            y_max = float(np.max(y_true))
            baseline_pred = np.full_like(y_true, y_mean)
            baseline_mae = mean_absolute_error(y_true, baseline_pred)
            baseline_rmse = np.sqrt(mean_squared_error(y_true, baseline_pred))
            range_span = max(1e-12, (y_max - y_min))
            nmae = mae / range_span  # normalized MAE by range

            msg = (
                f"Epoch {epoch+1}/{epochs} - Val loss: {avg_val_loss:.4f}, "
                f"R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, "
                f"y_std: {y_std:.4f}, NMAE: {nmae:.6f}, baseline_MAE: {baseline_mae:.4f}, baseline_RMSE: {baseline_rmse:.4f}"
            )
            if dose_step is not None and dose_step > 0:
                y_pred_rounded = np.round(y_pred / dose_step) * dose_step
                mae_round = mean_absolute_error(y_true, y_pred_rounded)
                msg += f", MAE@round({dose_step:g}): {mae_round:.4f}"
            print(msg)

        # Scheduler and early stopping
        scheduler.step(avg_val_loss)
        if early_stopping_patience is not None and early_stopping_patience > 0:
            if avg_val_loss + 1e-6 < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement in {early_stopping_patience} epochs)")
                    break

if __name__ == "__main__":
    # Point to the windowed balanced dataset for fair supervised training
    csv_path = "C:/Users/sudee/sbln_test/data/kidney.csv"
    # Use binary AF detection target by default
    target_column = "Dosage"

    # Exclude identifiers, text, window/time metadata, and other label columns
    exclude_columns = []

    train_model(csv_path, target_column, exclude_cols=exclude_columns)
