from CausalSBLN import train_model
train_model("data/titanic.csv","Survived",
  epochs=80, batch_size=64, lr=5e-4,
  gumbel_tau=1.2, tau_anneal=0.95, tau_min=0.7,
  aux_recon_weight=0.35, aux_mask_ratio=0.35,
  edge_entropy_lambda=0.02, graph_stability_lambda=0.05, graph_aug_noise=0.02,
  use_plasticity=True, num_entities=8, hidden_dim=64, num_steps=4,
  weight_decay=5e-4, grad_clip=1.0,
  early_stopping_patience=None)
