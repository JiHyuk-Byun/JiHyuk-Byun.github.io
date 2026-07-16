---
title: "The visual-interaction tax: what a VLM web agent pays for the human GUI"
date: 2026-07-08
tags: [web-agents, multimodal, evaluation, grounding]
categories: [experiments]
layout: post
---

## Question — what does the human-facing GUI *cost* a VLM web agent?

- A web GUI is designed for **humans**; a VLM agent uses it because that's what the app happens
  to expose. Using it isn't free: the agent must **read pixels, infer what's clickable, and aim
  coordinates** before any task reasoning pays off.
- **This post measures that price.** Call it the **visual-interaction tax**: the success-rate gap
  between running a task through the human GUI and running the *same* task through an equivalent
  text interface.
- **[Webstep](https://jiwanchung.github.io/webstep/)** makes the measurement clean: each web app is
  a deterministic **MDP** (state, fixed action set, pure transitions), and the GUI is just *one
  rendering* of it. Render the same MDP as **pixels** or as **text** — same information, same
  tasks, same model; only the interface differs. The performance gap *is* the tax.

## Setup

- **Model:** `Qwen3.5-9B` (unified multimodal), one vLLM server, **greedy** decoding (temp 0).
- **Benchmark:** Webstep — deterministic MDP web tasks, **10 sites, 220 tasks**. Both conditions
  run the *identical* task set (fixed seed), same 3-step memory window, same per-episode cap.
- **GUI condition:** observation = **screenshot**; actions = **pixel coordinates**.
- **text-MDP condition:** observation = the site's `observe()` output serialized to text
  (visibility-filtered → **the same information the screen shows**); actions = semantic
  `{type, payload}` applied **directly through the MDP** (`dispatch`), no pixels, no coordinates.
- Only the **interface** differs; the model, tasks, decoding, and judge are held fixed.

## Result 1 — the tax is large: **+40pp**, on every site

{% include figure.liquid loading="lazy" path="assets/img/posts/does-a-vlm-web-agent-even-need-the-gui/tax_by_site.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- **Overall: text-MDP 68% vs GUI 28% → a +40pp tax.** The *same weights* solve ~2.4× more tasks
  when the interface is text — so a large share of GUI-condition failures were **interface
  failures, not task failures**.
- The tax is **universal, not site-specific**: text wins on **all 10 sites** (+10pp → +65pp), and
  the GUI condition **never clears 50%** anywhere.

## Result 2 — removing the tax doesn't solve the task: multi-step execution still fails

- Text-MDP reaches 68%, **not 100%**: on multi-step tasks (open several items, compare an
  attribute, commit) the 9B **loops** — re-issuing the same action until the step cap.
- So the interface is **not the whole story**: even with the GUI cost at zero, a real
  **execution/reasoning gap remains**.
- *Caveat (unverified):* the full action history is in context, and the looping looks like a small
  model failing to exploit it — plausibly a **9B capability ceiling** rather than something
  fundamental. Untested at larger scale (see Caveats).

## Anatomy of the tax — what the GUI condition bundles

- Switching to text removes the **whole GUI stack at once**: (1) **perception** (read the screen),
  (2) **affordance inference** (what's actionable), (3) **coordinate grounding** (where to click).
- So +40pp prices **the bundle**, not any single layer — a **"visual-interaction tax,"** not a
  "grounding tax." Which layer dominates is an open question (below).

## Caveats (honest)

- **Model scale:** measured at the **~9B scale, single model/seed, greedy**. Screenshot grounding
  improves with scale and grounding-specialized training, so the tax is likely **smaller for
  larger or computer-use-tuned models** — untested here (a scale run was blocked on disk). Even at
  temp 0, run-to-run variance is ~2–3pp (vLLM batching).
- **Idealized text interface:** the MDP's `observe()`/`dispatch` is a **best-case** machine
  channel that real websites don't expose; realistic stand-ins (DOM, accessibility tree) are
  noisier. Read +40pp as an **upper bound** on what a text channel could recover.
- **Fairness fix:** the text view initially didn't expose the sites' **filter/sort vocabularies**
  (which the GUI agent sees as dropdown options), so text hallucinated invalid filters. We
  surfaced the real allow-lists from the transition functions before the final run; it restored
  parity but didn't move the aggregate (the bottleneck is comparison, not filtering).

## Remaining questions

- **Which layer dominates the tax?** Ablations that move one layer at a time — text observations
  with *coordinate* actions, screenshots with *semantic* actions, Set-of-Mark-style annotated
  screenshots — would split the +40pp across perception / affordance / grounding.
- **What would an agent-friendly UI be?** The text-MDP is ground truth the wild doesn't offer.
  Among realistic channels — **DOM, accessibility tree, semantic action APIs** — which noisy
  approximation retains most of the tax refund, and at what serialization cost (token length,
  staleness)?

## Takeaway

- **At the ~9B scale, a VLM web agent pays a +40pp visual-interaction tax for the human-facing
  GUI** (28% → 68% on identical tasks, worse on all 10 sites) — a *measurement* of the interface's
  price, not a prescription about interfaces and not GUI abolition.
- And the interface isn't everything: **remove the tax and multi-step execution still fails** —
  the residual 32% is an execution/reasoning gap, possibly just model scale.

## Code

- <a href="/assets/code/posts/does-a-vlm-web-agent-even-need-the-gui/make_figure.py" download><code>make_figure.py</code></a>
