---
title: "Reproduce Grokking with Weight Decay Ablation"
tags: [grokking, generalization, weight-decay]
categories: [experiments]
layout: post
date: 2026-06-11
---

<!-- Rough notes. Run /blogify to publish. -->

## Question
- **Grokking**: train accuracy hits 100% early (memorization), but validation generalizes
  *much later*.
- Reproduce it on `(a+b) mod 97`, and ablate weight decay — **does weight decay cause it?**

## Setup
- **Task:** modular addition `(a+b) mod 97`, all 97² pairs, 50% train / 50% val (fixed split).
- **Models:** 1-layer Transformer (d=128) and 2-layer MLP (d=128, hidden=256).
- **Optimizer:** AdamW, lr=1e-3, betas=(0.9,0.98), full-batch, 30k steps, seed 0.
- **Ablation:** `weight_decay ∈ {1.0, 0.0}` (model × wd grid). On ciplab-r6 / `jhbyun_toy_gpuall`.

## Result — grokking happens, and weight decay causes it
- All four runs **memorize by ~step 200** (train acc 100%). Only **weight decay** decides whether
  validation follows.

| model | wd | final val acc | val 90% at |
|-------|----|---------------|------------|
| transformer | 1.0 | 1.00 | ~step 1500 |
| transformer | 0.0 | 0.36 | never |
| MLP | 1.0 | 1.00 | ~step 2100 |
| MLP | 0.0 | 0.03 | never |

{% include figure.liquid loading="lazy" path="assets/img/posts/reproduce-grokking-with-weight-decay-ablation/grokking_curve.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- With wd: train saturates ~200, val only reaches 100% much later (~1500 / ~2100). **That delay
  on the log-step axis *is* grokking.**

{% include figure.liquid loading="lazy" path="assets/img/posts/reproduce-grokking-with-weight-decay-ablation/wd_ablation.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- **wd → val 100%.** Without wd, neither fully groks — but they differ: the **MLP stays near
  chance (~0.03)**, while the **transformer *partially* generalizes (~0.36)**. *Why:* weight-sharing
  inductive bias (it can't memorize each pair in isolation) + CE's implicit **max-margin** drift —
  a weaker pressure toward generalization than weight decay. (TODO: run wd=0 longer + log norm —
  is the 0.36 a slow implicit-bias grok?)
- **High norm = memorizing solution** (spiky, per-example; needs large weights).
- **Low norm = generalizing solution** (smooth, the shared `mod` rule).
- Both fit train perfectly → loss can't choose; the model lands on the memorizing one first.
  **Weight decay shrinks the norm every step → migrates** memorizing → generalizing = the delayed val jump. No wd, no migration.

{% include figure.liquid loading="lazy" path="assets/img/posts/reproduce-grokking-with-weight-decay-ablation/loss_curves.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- Val loss **overfits (rises), then collapses** at grokking and converges to train loss.
- **MLP**: clean, textbook. **Transformer**: same shape but **spiky** through the transition.

## Observations
- **MLP is much more stable than the transformer** (same qualitative picture). *Why:* a
  transformer layer combines weights *multiplicatively* (bilinear `QKᵀ`, softmax, LayerNorm) →
  sharper loss landscape (larger curvature) → at fixed lr / full-batch it sits near the **edge of
  stability** and oscillates.
- *Caveat (unverified):* the instability may be partly that `wd=1.0`, `lr=1e-3` is too aggressive
  *for the transformer*, not purely architectural. TODO: lr=5e-4 / warmup / `wd∈{0.3,0.5}`.

## Takeaway
- Grokking = **separation of timescales**: memorization is fast, finding the *generalizing*
  solution is slow; **weight decay bridges them.**
- Both solutions fit train, so the loss can't pick — but they differ in **weight norm**
  (memorizing = high, generalizing = low). Weight decay shrinks the norm, migrating the model
  from one to the other.
- *Inferred hypothesis:* **small weight norm biases toward simpler, generalizing solutions.**
- *Follow-up:* does this transfer to natural data? Harder than it looks — see the companion post,
  **"Inducing grokking on natural (image) data."**

## Code
[`run.ipynb`](/assets/code/posts/reproduce-grokking-with-weight-decay-ablation/run.ipynb)

## References
- Power et al. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. arXiv:2201.02177.
- Nanda et al. *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. arXiv:2301.05217.
- Cohen et al. *Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability.* ICLR 2021. arXiv:2103.00065.
