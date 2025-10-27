# General Neuro-Dynamic Importance (G-NDI)

**Causal Layer Attribution via Virtual Interventions**  
*David Leonard Nagy, Independent Research, 2025*

---

## Overview
**G-NDI** estimates each layer’s **causal impact** on a model’s output by simulating a *virtual intervention*:

> What would happen to the model’s prediction if this layer’s activation were replaced with a neutral baseline?

By approximating the effect of the intervention do(h_L := b_L) via a **Jacobian–Vector Product (JVP)**, G-NDI captures *true causal dependence* rather than mere correlation (as in SNIP, GraSP, or Magnitude pruning).

---

## Method

For each layer \( L \) in model \( f(x) = F_L(h_L(x)) \):

$$
\text{G-NDI}_L = \mathbb{E}_x \, \| J_{>L}(x) \, (h_L(x) - b_L) \|_p
$$

where  
- \( J_{>L}(x) \): downstream Jacobian  
- \( h_L(x) \): layer activation  
- \( b_L \): neutral baseline (zero or mean)  
- \( p \): norm order (typically 2)

**Requires:** one forward + one JVP per layer.  
**No retraining required.**

---

## Results

**Dataset:** CIFAR-10 / CIFAR-100 **Model:** ResNet-18  
**Baselines:** SNIP, GraSP, SynFlow, Magnitude, HRank

| Metric | **G-NDI** | SNIP | HRank | Magnitude | GraSP |
|:--|:--:|:--:|:--:|:--:|:--:|
| **Pearson (r)** | **0.96** | 0.80 | 0.62 | −0.54 | −0.07 |
| **Spearman (ρ)** | **0.98** | 0.91 | 0.66 | −0.51 | 0.06 |
| **Kendall (τ)** | **0.90** | 0.73 | 0.48 | −0.35 | −0.04 |

---

## Highlights
- **Causal interpretability:** measures true functional dependence  
- **High causal validity:** Spearman ρ ≈ 0.98 on CIFAR-10  
- **Retraining-free & efficient:** only forward + JVP  
- **Cross-domain:** CV → NLP generalization  
- **PyTorch implementation:** lightweight and reproducible

---

## Usage
```bash
git clone https://github.com/yourusername/g-ndi
cd g-ndi
pip install -r requirements.txt
```
```python
from gndi.scoring import compute_gndi
score = compute_gndi(model, dataloader, baseline='zero', norm='l2')
```

David Leonard Nagy (2025)
"General Neuro-Dynamic Importance (G-NDI): Causal Layer Attribution via Virtual Interventions"
https://davidleonardnagy.com/gndi.pdf

License: MIT

