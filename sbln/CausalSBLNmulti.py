import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from torch.utils.data import Dataset, DataLoader

# --- Model and helper from before ---

class CausalSimulationModule(nn.Module):
    def __init__(self, num_entities, hidden_dim, tau=0.5):
        super().__init__()
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.tau = tau

        self.entity_proj = nn.Linear(hidden_dim, hidden_dim)
        self.edge_logits = nn.Parameter(torch.randn(num_entities, num_entities))
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def sample_adjacency(self):
        logits = self.edge_logits
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
        noisy_logits = (logits + gumbel_noise) / self.tau
        return F.softmax(noisy_logits, dim=-1)

    def forward(self, entity_states):
        B, N, D = entity_states.shape
        assert N == self.num_entities

        projected = self.entity_proj(entity_states)
        A = self.sample_adjacency().unsqueeze(0).expand(B, -1, -1)

        updates = []
        for i in range(N):
            target = projected[:, i, :].unsqueeze(1).expand(-1, N, -1)
            source = projected
            combined = torch.cat([source, target], dim=-1)
            message = self.update_mlp(combined)
            weighted = A[:, i, :].unsqueeze(-1) * message
            agg_update = weighted.sum(dim=1)
            updates.append(agg_update)

        updates = torch.stack(updates, dim=1)
        next_state = entity_states + updates
        return next_state, A

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

class MultiOutputCausalSBLN(nn.Module):
    def __init__(self, input_dim, num_entities, hidden_dim, output_dims, tau=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.num_entities = num_entities
        self.hidden_dim = hidden_dim
        self.output_dims = output_dims  # Dictionary of {column_name: output_dim}
        self.num_steps = 3

        self.encoder = nn.Linear(input_dim, num_entities * hidden_dim)
        self.causal_sim = CausalSimulationModule(num_entities, hidden_dim, tau)
        self.recurrent_sim = nn.GRUCell(hidden_dim, hidden_dim)

        # Separate decoders for each output
        self.decoders = nn.ModuleDict()
        for col_name, output_dim in output_dims.items():
            self.decoders[col_name] = nn.Sequential(
                nn.Linear(num_entities * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )

    def simulate_temporal(self, entity_states):
        B, N, D = entity_states.shape
        states = entity_states
        causal_graphs = []

        for _ in range(self.num_steps):
            next_states, A = self.causal_sim(states)
            causal_graphs.append(A)

            flat = next_states.view(B * N, D)
            updated = self.recurrent_sim(flat, flat)
            states = updated.view(B, N, D)

        return states, torch.stack(causal_graphs, dim=1)

    def forward(self, x):
        B = x.size(0)

        encoded = self.encoder(x)
        entity_states = encoded.view(B, self.num_entities, self.hidden_dim)

        final_states, causal_graph_seq = self.simulate_temporal(entity_states)
        flat_state = final_states.view(B, -1)
        
        # Predict all outputs
        outputs = {}
        for col_name, decoder in self.decoders.items():
            outputs[col_name] = decoder(flat_state)
            
        return outputs, causal_graph_seq

# --- Multi-output dataset wrapper ---

class MultiOutputTabularDataset(Dataset):
    def __init__(self, X, y_dict):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_dict = {}
        
        for col_name, y in y_dict.items():
            if isinstance(y, np.ndarray) and y.dtype.kind in 'iu':  # int for classification
                self.y_dict[col_name] = torch.tensor(y, dtype=torch.long)
            else:  # regression float
                self.y_dict[col_name] = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        y_batch = {col: y[idx] for col, y in self.y_dict.items()}
        return self.X[idx], y_batch

def load_and_preprocess_multi_output(csv_path, target_cols, exclude_cols=['id', 'SMILES']):
    df = pd.read_csv(csv_path)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Drop excluded columns
    df = df.drop(columns=exclude_cols, errors='ignore')
    
    # Drop columns that are all NaN
    df = df.dropna(axis=1, how='all')
    
    print(f"After dropping excluded columns: {df.shape}")
    print(f"Remaining columns: {df.columns.tolist()}")
    
    # Separate features and targets
    y_dict = {}
    for col in target_cols:
        if col in df.columns:
            # Impute missing values in target columns instead of dropping rows
            if df[col].dtype.kind in 'iufc':  # numeric columns
                imputer = SimpleImputer(strategy='mean')
                y_dict[col] = imputer.fit_transform(df[[col]]).flatten()
            else:  # categorical columns
                imputer = SimpleImputer(strategy='most_frequent')
                y_dict[col] = imputer.fit_transform(df[[col]]).flatten()
        else:
            print(f"Warning: Column {col} not found in dataset")
    
    # Remove target columns from features
    X = df.drop(columns=[col for col in target_cols if col in df.columns], errors='ignore')
    
    print(f"Features shape after removing targets: {X.shape}")
    print(f"Feature columns: {X.columns.tolist()}")
    
    # If we have no features, create some from SMILES
    if X.shape[1] == 0:
        print("No features left, creating features from SMILES...")
        # Read original data to get SMILES
        original_df = pd.read_csv(csv_path)
        
        # Create simple features from SMILES
        smiles_features = []
        for smiles in original_df['SMILES']:
            # Simple features: length, character counts, etc.
            features = [
                len(str(smiles)),  # length
                str(smiles).count('C'),  # carbon count
                str(smiles).count('O'),  # oxygen count
                str(smiles).count('N'),  # nitrogen count
                str(smiles).count('('),  # parentheses count
                str(smiles).count('='),  # double bond count
                str(smiles).count('#'),  # triple bond count
                str(smiles).count('*'),  # wildcard count
                str(smiles).count('['),  # bracket count
                str(smiles).count('@'),  # chirality count
            ]
            smiles_features.append(features)
        
        X = pd.DataFrame(smiles_features, columns=[
            'length', 'C_count', 'O_count', 'N_count', 'paren_count',
            'double_bond', 'triple_bond', 'wildcard', 'bracket', 'chirality'
        ])
        print(f"Created SMILES features: {X.shape}")

    # Detect categorical columns (object or category)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    print(f"Numerical columns: {num_cols}")
    print(f"Categorical columns: {cat_cols}")

    # Simple imputers
    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='most_frequent')

    # Impute numerical
    if len(num_cols) > 0:
        X_num = num_imputer.fit_transform(X[num_cols])
    else:
        X_num = np.empty((len(X), 0))

    # Impute categorical and encode
    X_cat = None
    if len(cat_cols) > 0:
        X_cat_raw = cat_imputer.fit_transform(X[cat_cols])
        # Encode categorical as integers
        X_cat = np.zeros_like(X_cat_raw, dtype=np.float32)
        for i, col in enumerate(cat_cols):
            le = LabelEncoder()
            X_cat[:, i] = le.fit_transform(X_cat_raw[:, i])

    # Combine numerical and categorical features
    X_all = np.hstack([X_num, X_cat]) if X_cat is not None else X_num

    print(f"Final feature matrix shape: {X_all.shape}")

    # Scale all features
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # Determine task types and output dimensions for each target
    output_dims = {}
    task_types = {}
    
    for col_name, y in y_dict.items():
        unique_vals = np.unique(y)
        if y.dtype.kind in 'iufc' and len(unique_vals) > 10:
            task_types[col_name] = "regression"
            output_dims[col_name] = 1
        elif len(unique_vals) <= 20:
            if y.dtype.kind in 'iufc':
                if np.all(np.mod(unique_vals, 1) == 0):
                    task_types[col_name] = "classification"
                    y_dict[col_name] = LabelEncoder().fit_transform(y)
                    output_dims[col_name] = len(unique_vals)
                else:
                    task_types[col_name] = "regression"
                    output_dims[col_name] = 1
            else:
                task_types[col_name] = "classification"
                y_dict[col_name] = LabelEncoder().fit_transform(y)
                output_dims[col_name] = len(unique_vals)
        else:
            task_types[col_name] = "classification"
            if y.dtype.kind not in 'iufc':
                y_dict[col_name] = LabelEncoder().fit_transform(y)
            output_dims[col_name] = len(unique_vals)

    return X_all, y_dict, task_types, output_dims

# --- Multi-output training loop ---

def train_multi_output_model(csv_path, target_cols, exclude_cols=['id', 'SMILES'], 
                           batch_size=32, epochs=20, lr=1e-3, acyclicity_lambda=0.01, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    X, y_dict, task_types, output_dims = load_and_preprocess_multi_output(csv_path, target_cols, exclude_cols)

    # Train/val split
    indices = np.arange(len(X))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)
    
    X_train, X_val = X[train_indices], X[val_indices]
    y_train_dict = {col: y[train_indices] for col, y in y_dict.items()}
    y_val_dict = {col: y[val_indices] for col, y in y_dict.items()}

    # Create datasets and loaders
    train_dataset = MultiOutputTabularDataset(X_train, y_train_dict)
    val_dataset = MultiOutputTabularDataset(X_val, y_val_dict)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    input_dim = X.shape[1]

    model = MultiOutputCausalSBLN(input_dim=input_dim, num_entities=5, hidden_dim=32, 
                                 output_dims=output_dims).to(device)

    # Create loss functions for each output
    criteria = {}
    for col_name, task_type in task_types.items():
        if task_type == "classification":
            criteria[col_name] = nn.CrossEntropyLoss()
        else:
            criteria[col_name] = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on device: {device}")
    print(f"Target columns: {list(target_cols)}")
    print(f"Task types: {task_types}")
    print(f"Input dim: {input_dim}, Output dims: {output_dims}")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for xb, yb_dict in train_loader:
            xb = xb.to(device)
            yb_dict = {col: yb.to(device) for col, yb in yb_dict.items()}

            optimizer.zero_grad()
            outputs, causal_graph_seq = model(xb)

            # Compute loss for each output
            loss = 0
            for col_name, output in outputs.items():
                if task_types[col_name] == "classification":
                    loss += criteria[col_name](output, yb_dict[col_name])
                else:
                    output = output.squeeze()
                    yb = yb_dict[col_name].float()
                    loss += criteria[col_name](output, yb)

            # Acyclicity regularization
            A_last = causal_graph_seq[:, -1, :, :]
            loss += acyclicity_lambda * acyclicity_loss(A_last)

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_metrics = {}
        
        with torch.no_grad():
            for xb, yb_dict in val_loader:
                xb = xb.to(device)
                yb_dict = {col: yb.to(device) for col, yb in yb_dict.items()}
                outputs, _ = model(xb)

                for col_name, output in outputs.items():
                    if col_name not in val_metrics:
                        val_metrics[col_name] = {'preds': [], 'labels': []}
                    
                    if task_types[col_name] == "classification":
                        preds = output.argmax(dim=1)
                        val_metrics[col_name]['preds'].extend(preds.cpu().numpy())
                        val_metrics[col_name]['labels'].extend(yb_dict[col_name].cpu().numpy())
                    else:
                        output = output.squeeze()
                        val_metrics[col_name]['preds'].extend(output.cpu().numpy())
                        val_metrics[col_name]['labels'].extend(yb_dict[col_name].cpu().numpy())

        # Print validation metrics
        print(f"Epoch {epoch+1}/{epochs} - Validation metrics:")
        for col_name in target_cols:
            preds = np.array(val_metrics[col_name]['preds'])
            labels = np.array(val_metrics[col_name]['labels'])
            
            if task_types[col_name] == "classification":
                accuracy = (preds == labels).mean()
                f1 = f1_score(labels, preds, average='weighted')
                print(f"  {col_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}")
            else:
                mse = mean_squared_error(labels, preds)
                r2 = r2_score(labels, preds)
                print(f"  {col_name}: MSE={mse:.4f}, R²={r2:.4f}")

if __name__ == "__main__":
    # Multi-output prediction for polymer dataset
    csv_path = "../polymer.csv"
    target_columns = ["Tg", "FFV", "Tc", "Density", "Rg"]  # All numeric columns except id and SMILES
    
    train_multi_output_model(csv_path, target_columns)