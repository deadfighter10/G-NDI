# General Neuro-Dynamic Importance (G-NDI)

**G-NDI** (General Neuro-Dynamic Importance) is a *causal framework* for measuring the importance of neural network layers and units.  
Instead of relying on weight magnitude or gradient sensitivity, G-NDI estimates how much the model’s output **would change if a layer were neutralized** — a virtual intervention approximated via Jacobian–vector products.
*G-NDI reframes pruning as a causal reasoning problem rather than a correlational heuristic.*

---

## Method

For each layer $L$ in a model $f(x)$, the **causal importance** is defined as:

$$
\text{GNDI}_L = \mathbb{E}_x \, \left\| J^{>L}(x) \, (h_L(x) - b_L) \right\|_p
$$

where:
- $J^{>L}(x)$ – downstream Jacobian of layers following $L$
- $h_L(x)$ – activation of layer $L$
- $b_L$ – neutral baseline (zero or mean)
- $\|\cdot\|_p$ – $p$-norm (typically $p=2$)

This approximates the **counterfactual intervention** $do(h_L = b_L)$ in Pearl’s causal calculus, and measures how much the model’s prediction would shift under that intervention.

G-NDI requires only:
- **1 forward pass**
- **1 Jacobian–Vector Product (JVP)** per layer  
→ No retraining, no second-order derivatives.

---

## Experimental Results

Evaluated on **CIFAR-10 / ResNet-18** and **CIFAR-100 / ResNet-18**:

| Dataset | Model | Pearson ($r$) | Spearman ($\rho$) | Kendall ($\tau$) |
|----------|--------|----------------|-------------------|------------------|
| CIFAR-10 | ResNet-18 | **0.95** | **0.98** | **0.90** |
| CIFAR-100 | ResNet-18 | 0.80 | 0.87 | 0.69 |

Compared against SNIP, GraSP, SynFlow, Magnitude, and HRank.  
G-NDI achieves the **highest causal validity** (correlation with true ablation damage) while maintaining competitive accuracy after pruning.

---

## Causal Comparison

| Method | Pearson | Spearman | Kendall | Spearman CI (95%) |
|--------|----------|-----------|----------|--------------------|
| **G-NDI** | **0.958** | **0.981** | **0.900** | [0.970, 0.988] |
| SNIP | 0.797 | 0.905 | 0.728 | [0.877, 0.923] |
| HRank | 0.624 | 0.663 | 0.478 | [0.591, 0.729] |
| Magnitude | -0.543 | -0.506 | -0.352 | [-0.584, -0.418] |
| GraSP | -0.069 | 0.055 | -0.040 | [-0.193, 0.080] |

---

## Key Features

- **Causal interpretability:** Quantifies *actual dependency* between layers and outputs.  
- **Efficiency:** Requires only first-order JVPs.  
- **Cross-domain:** Applicable to CNNs, MLPs, Transformers.  
- **No retraining:** Instant causal importance estimation.

---

## Citation

If you use G-NDI in your work, please cite the mini-paper:

@article{nagy2025gndi,\
title={General Neuro-Dynamic Importance (G-NDI): Causal Layer Attribution via Virtual Interventions},\
author={Nagy, David Leonard},\
year={2025},\
institution={Independent Research, Northwest Missouri State University}\
}

---

## Contact

For collaboration or visiting-research opportunities:  
**Email:** s583993@nwmissouri.edu  
**Portfolio:** [davidleonardnagy.com](https://davidleonardnagy.com)
