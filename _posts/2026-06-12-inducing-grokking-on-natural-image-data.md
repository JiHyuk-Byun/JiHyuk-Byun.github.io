---
title: "Inducing grokking on natural (image) data"
tags: [grokking, generalization, mnist, weight-decay]
categories: [experiments]
layout: post
date: 2026-06-12
---

<!-- Rough notes. Run /blogify to publish. Companion to the modular-addition post. -->

## Question
- Algorithmic tasks (modular addition) make a "memorize ≠ generalize" gap **intrinsically** —
  memorizing the train pairs says nothing about unseen ones.
- **MNIST has no such intrinsic gap**: its quickly-found solution already generalizes, so a
  *non-generalizing* quick fit — a **memorization trap** — is hard to fall into. So it doesn't
  grok on its own.
- **Can we manufacture a trap?** Candidate knobs: **loss**, **architecture**, **init scale α** —
  plus **weight decay** to traverse it. We ablate to see which actually matter.

## Setup
- **Dataset:** MNIST, 1000-image train subset (full 10k test).
- **Models:** MLP (784-256-256-10) and a small **BN-free** CNN (BN cancels the init-scale trick).
- **Loss (ablated):** MSE (regress to one-hot) vs cross-entropy.
- **Init scale α (ablated):** scales the initial weights (large α = start far from generalizing).
- **Weight decay:** the traversal force — 0.1, with a 0.0 control.
- **Optimizer:** AdamW, lr=1e-3, full-batch, 40k steps, seed 0; log acc + weight norm /200 steps.
- **Runs:** knockout grid (arch × loss × α∈{1,8}, wd=0.1, +wd=0 control) + α-sweep (MLP, MSE vs
  CE, α∈{1,2,4,8,16}). On ciplab-r6 / `jhbyun_toy_gpuall` (A6000).

## Result 1 — the loss decides; the architecture doesn't
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_curves.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- **Grokking only with MSE** — for *both* MLP and CNN (train memorizes early, test jumps ~10×
  later). **Never with CE** (test rises with train, no delay).
- **wd = 0 → no grokking** (gray dashed): the model memorizes and stays stuck (test ~0.24).
- *Overturned our guess:* we expected the CNN's image bias to **prevent** grokking. It doesn't —
  CNN groks too (even sharper, test 0.96). **Architecture does not gate grokking.**
- It only changes the **delay**: CNN resists memorization longer (memorizes ~step 4800 vs MLP
  ~1200), so its grokking gap is **~2× on the step axis vs the MLP's ~12×** (same accuracy jump).

## Result 2 — init scale only digs the trap *under MSE*
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_alpha.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- **Test accuracy at memorization vs α** (MLP):
  - **MSE collapses** as α grows: 0.92 → 0.72 → **0.13** (deeper trap; *final* recovers to ~0.92
    = grokking, and the gap grows with α).
  - **CE stays ~0.74+** — no collapse, no trap.
- So *init scale is not independently decisive* — it deepens the trap **only when the loss is MSE**.

## Result 3 — mechanism: norm migration = generalization
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_norm.png" class="img-fluid rounded z-depth-1" zoomable=true %}
- MLP/MSE/α8/wd0.1: weight norm starts high (~106), holds while memorizing, then **decays to
  ~20 — and test rises exactly as it falls.** Weight decay slides the model from the high-norm
  memorizing solution to the low-norm generalizing one.

## Why MSE traps but CE doesn't
- **MSE = equality.** Pins all 10 outputs to exact 0/1. At large init the reachable fit is a
  spiky, high-norm, per-example memorizer → *manufactured trap*.
- **CE = inequality / ranking.** Only the correct logit must be largest; GD's implicit bias is
  **max-margin** (smooth, generalizing). The reachable solution *already generalizes* — no trap.

## When does grokking happen? (two conditions)
1. **A gap** — the quick fit doesn't generalize. *Intrinsic* to the task (algorithmic), or
   *manufactured* (MSE + large init).
2. **A regime** — small fixed data + overtraining + a push toward simplicity (weight decay).

- **Modular addition**: gap is intrinsic → groks across losses (even CE).
- **MNIST**: no intrinsic gap → groks only if manufactured (MSE + large init), then traversed by wd.
- **LLMs**: reported grokking is in *controlled* setups (small data, many epochs, wd) on
  *compositional* sub-tasks; vanilla pretraining (huge data, ~1 epoch) lacks condition 2.

## Takeaway
- The knob that *makes* grokking on gap-less data is the **loss**: **MSE traps, CE doesn't**.
- **Init scale is not independently decisive** — it deepens the trap *only under MSE* (under CE,
  scaling init barely moves anything: no collapse).
- **Architecture is not decisive** either — MLP and CNN both grok; the CNN's bias only **narrows
  the delay**.
- **Weight decay** traverses the trap once it exists.
- **One line:** *grokking = gap (intrinsic, or manufactured by MSE **and** large init) × push
  (weight decay) × regime (overtrain on small fixed data).*

## Code
[`ablation.ipynb`](/assets/code/posts/inducing-grokking-on-natural-image-data/ablation.ipynb)

## References
- Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023. arXiv:2210.01117.
- Power et al. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.* 2022. arXiv:2201.02177.
- Hui, Belkin. *Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy.* ICLR 2021. arXiv:2006.07322.
- Soudry et al. *The Implicit Bias of Gradient Descent on Separable Data.* JMLR 2018. arXiv:1710.10345.
- Wang et al. *Grokked Transformers are Implicit Reasoners.* 2024. arXiv:2405.15071.
