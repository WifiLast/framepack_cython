# Timestep Residual/Modulation Integration

- [ ] Wire the new `relationship_trainer_mode` radio in `demo_gradio.py` to instantiate `DiTTimestepResidualTrainer` and `DiTTimestepModulationTrainer` per transformer block and toggle them based on the UI selection.
- [ ] Extend the block I/O callback data to include encoder states, attention masks, timestep embeddings, and RoPE information so the new trainers can build `(h_in, h_full, h_base)` and `(t, γ(t), β(t))` datasets.
- [ ] Implement persistence for the residual/modulation predictors (similar to the hidden-state trainer) and save/load their states inside `_persist_runtime_caches_on_exit`.
- [ ] Add block override hooks in `HunyuanVideoTransformer3DModelPacked` so inference can call `h_base + gφ(h_in, T_t)` or substitute γ̂/β̂; ensure overrides are cleared when modes are disabled.
- [ ] After each generation, run training steps for the active trainer using the collected tuples; expose configurable learning rates/batch sizes and log convergence metrics.
- [ ] Update README/OPTIMIZATIONS/UI docs to describe the new modes, usage instructions, and limitations so users know when to enable residual vs modulation predictors.
