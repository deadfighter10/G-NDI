# General Neuro-Dynamic Importance (G-NDI)
A causal approach to layer importance and model pruning.

**Summary**  
G-NDI estimates each layer’s causal impact by simulating a *virtual intervention*—replacing its output with a neutral baseline—and predicting the effect via a first-order Jacobian approximation.  
It aligns strongly with actual layer importance in CIFAR-10 / ResNet-18 experiments.

**Results**
- Pearson r = 0.55  
- Spearman ρ = 0.66  
- AUC = 0.71 (critical-layer detection)

**Highlights**
- Causal measure (no retraining needed)  
- Zero & mean baseline variants  
- Supports ResNet and MLP architectures  
- Built in PyTorch, lightweight & reproducible

**Status:** Independent research, 2025  
**License:** MIT
