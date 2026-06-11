---
title: "Inducing grokking on natural (image) data"
tags: [grokking, generalization, mnist, weight-decay]
categories: [experiments]
layout: post
date: 2026-06-12
hidden: true
---

<!-- Rough notes. Run /blogify on this folder to turn it into a polished, published draft.
     Companion to the modular-addition post — read that for the basic mechanism. -->

## Question
On an algorithmic task (modular addition) grokking is dramatic and easy — the task itself
makes a "memorize ≠ generalize" gap. **MNIST has no such intrinsic gap**: the quick-fit
solution already generalizes, so it doesn't grok on its own. Can we *manufacture* a gap? We
hypothesized three knobs — **loss** (MSE vs CE), **architecture** (MLP vs CNN), **init scale
α** — plus **weight decay** to traverse the gap, and ran a clean ablation to see which knobs
actually matter.

## Setup
- MNIST, a **1000-image** train subset (full 10k test), AdamW lr=1e-3, **full-batch**, 40k
  steps, seed 0. Log train/test accuracy + weight L2 norm per 200 steps.
- Models: **MLP** (784-256-256-10) and a small **BN-free CNN** (BatchNorm would cancel the
  init-scale trick). Losses: **MSE** (regress to one-hot) and **CE**. Init scale **α** scales
  the initial weights.
- Two parts: a **knockout grid** (arch × loss × α∈{1,8} at wd=0.1, + a wd=0 control) and an
  **α-sweep** (MLP, MSE vs CE, α∈{1,2,4,8,16}). Ran on ciplab-r6 / `jhbyun_toy_gpuall`.

## Result 1 — the loss is decisive; the architecture is not
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_curves.png" class="img-fluid rounded z-depth-1" zoomable=true %}

Each panel is α=8, wd=0.1. **Grokking (train memorizes early, test jumps much later) appears
with MSE — for *both* the MLP and the CNN** — and **never with CE** (test rises together with
train and plateaus). The dashed gray curve (MLP, MSE, **wd=0**) shows the trap: without weight
decay the model memorizes and **stays stuck** (test ~0.24 forever).

This **overturned our guess**: we expected a CNN's image inductive bias to keep it close to the
generalizing solution and so *prevent* grokking. It doesn't — given MSE + large init + weight
decay the **CNN groks too (even more sharply: test 0.96)**. So architecture is *not* the knob
that gates grokking here; **the loss is** (and the init, below).

## Result 2 — init scale digs the trap (and weight decay climbs out)
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_alpha.png" class="img-fluid rounded z-depth-1" zoomable=true %}

For the MLP, plotting **test accuracy at the moment of memorization** vs init scale: with
**MSE** it **collapses as α grows** (0.92 → 0.21 → 0.13) — a larger init lands the memorizing
fit in a spikier, *non-generalizing* solution. The **final** test (dashed) stays ~0.92 — the
model *groks out* of the trap by the end. **The gap between "at memorization" and "final" is
the grokking magnitude, and it opens up as α grows.** With **CE** there is barely a gap at all
(it generalizes at memorization regardless of α).

## Result 3 — the mechanism: norm migration = generalization
{% include figure.liquid loading="lazy" path="assets/img/posts/inducing-grokking-on-natural-image-data/ablation_norm.png" class="img-fluid rounded z-depth-1" zoomable=true %}

For the grokking cell (MLP/MSE/α8/wd0.1): the weight norm starts high (~106), stays flat while
the model memorizes, then **decays to ~20 — and test accuracy rises exactly as the norm
falls.** Weight decay slides the model along the zero-train-loss manifold from the high-norm
memorizing solution to the low-norm generalizing one.

## Why does MSE trap but CE doesn't?
- **MSE = equality.** It pins all 10 outputs to exact 0/1 values per example. At large init the
  reachable fit is a **spiky, high-norm, per-example memorizer that fails to generalize** — a
  manufactured trap. Weight decay is then needed to migrate out of it.
- **CE = inequality / ranking.** It only needs the correct logit to be **largest** (softmax);
  the satisfying set is a region, and GD's implicit bias is toward the **max-margin** (smooth,
  generalizing) solution (Soudry et al.). So the reachable solution **already generalizes** —
  no trap to grok out of, for either architecture.

## When does grokking happen? (two conditions)
1. **A memorize-vs-generalize gap** — the quick-fit solution does not generalize. Made by the
   *task* (algorithmic / non-smooth), or *manufactured* here by **MSE + large init**.
2. **A regime that overfits then keeps going** — small fixed data + heavy overtraining + a
   force toward simplicity (**weight decay**).

This ties everything together:
- **Modular addition**: the task makes the gap intrinsically → groks across losses (even CE).
- **MNIST (natural images)**: no intrinsic gap → grokking only if you manufacture one with
  **MSE + large init**, then traverse with weight decay. CE doesn't manufacture it → no grok.
  **Architecture (MLP/CNN) is not the deciding factor** — both grok under the recipe.
- **LLMs**: "grokking" is reported in **controlled** settings (small fixed data, many epochs,
  weight decay) on **compositional/algorithmic** sub-tasks (both conditions hold); vanilla
  large-scale pretraining (huge data, ~1 epoch) lacks condition 2, so it shows gradual learning.

## Takeaway
- On a task with **no intrinsic gap (MNIST)**, grokking is **manufactured** — and the
  decisive knobs are the **loss (MSE, not CE)** and the **init scale (large α)**; the
  **architecture is not** (MLP and CNN both grok). **Weight decay** is the force that traverses
  the manufactured gap; without it the model stays trapped.
- One-line rule: **grokking = gap (made by the task, or by MSE + large init) × push (weight
  decay) × regime (overtrain on small fixed data).**

## Code

Full ablation notebook (config cell at the top; run on a GPU):
[`ablation.ipynb`](/assets/code/posts/inducing-grokking-on-natural-image-data/ablation.ipynb)

## References

- Liu, Michaud, Tegmark. *Omnigrok: Grokking Beyond Algorithmic Data.* ICLR 2023.
  arXiv:2210.01117. — grokking on MNIST via large init scale + weight decay.
- Power et al. *Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets.*
  2022. arXiv:2201.02177. — the original phenomenon (algorithmic, cross-entropy).
- Hui, Belkin. *Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy.*
  ICLR 2021. arXiv:2006.07322. — MSE for classification.
- Soudry et al. *The Implicit Bias of Gradient Descent on Separable Data.* JMLR 2018.
  arXiv:1710.10345. — CE + GD converges to the max-margin solution.
- Wang et al. *Grokked Transformers are Implicit Reasoners.* 2024. arXiv:2405.15071. —
  grokking in transformers on a controlled compositional task.
