---
title: "The visual-understanding tax: a 9B web agent, text vs pixels"
date: 2026-07-08
tags: [web-agents, multimodal, evaluation, grounding]
categories: [experiments]
layout: post
hidden: true
---

## Question — is *seeing the GUI* the bottleneck, not reasoning?

- Modern LLMs are strong at math/code reasoning but still weak as **web/GUI agents**.
- Hypothesis: the blocker isn't reasoning — it's **visual understanding of the GUI**.
- Clean test: give the **same model** the *same* tasks, but hand it the environment as
  **structured text** (bypassing pixels) vs as **screenshots**, and measure the gap.

## Setup

- **Model:** `Qwen3.5-9B` (unified multimodal), one vLLM server, **greedy** decoding (temp 0).
- **Benchmark:** Webbench — deterministic MDP web tasks, **10 sites, 220 tasks**. Both conditions
  run the *identical* task set (fixed seed), same 3-step memory window, same per-episode cap.
- **text-MDP condition:** observation = the site's `observe()` output serialized to text
  (visibility-filtered → **the same information the screen shows**); actions = semantic
  `{type, payload}` applied **directly through the MDP** (`dispatch`), no pixels, no coordinates.
- **vision condition:** observation = **screenshot**; actions = **pixel coordinates**.
- Only the **modality** differs; the model, tasks, decoding, and judge are held fixed.

## Result 1 — GUI understanding, not reasoning, is the bottleneck: **+40pp**

{% include figure.liquid loading="lazy" path="assets/img/posts/the-visual-understanding-tax-a-9b-web-agent-text-vs-pixels/tax_by_site.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- **Overall: text-MDP 68% vs vision 28% → +40pp.** The *same weights* solve far more when the
  environment is handed to them as text — so the planning/reasoning was largely **already there**;
  what's missing is **perceiving and operating the GUI through pixels**.
- Text **wins on all 10 sites** (per-site tax +10pp → +65pp); **vision never clears 50%** anywhere.
- **Main finding:** in these web-agent tasks the bottleneck is **understanding the GUI visually,
  more than task reasoning** — remove the visual channel and success **~doubles**.

## Result 2 — some reasoning headroom remains — but likely a 9B-capability issue

- Text-only isn't perfect: on multi-step tasks (open several items, compare an attribute, commit)
  the 9B **loops** — re-issuing the same action and hitting the step cap instead of finishing.
- So even after vision is removed, **some web-task reasoning headroom is still on the table.**
- **But this is plausibly just the 9B's capability ceiling, not a fundamental barrier:** the full
  action history is in context, and the failure looks like a small model not exploiting it.
  Whether a **larger model** closes this gap is **untested** (see caveats) — so we read it as a
  *possible* capability limit, not a hard claim, and separate from the visual bottleneck of Result 1.

## Why — what "visual understanding" bundles

- The text condition removes the **whole GUI stack at once**: (1) **perception** (read the screen),
  (2) **affordance inference** (what's actionable), (3) **coordinate grounding** (where to click).
- So the +40pp is the cost of **that whole stack** — *not* pixel-coordinate grounding alone.
  Call it a **"visual-understanding tax,"** not a "grounding tax."

## Caveats (honest)

- **What it measures:** the whole perception + affordance + grounding stack, not coordinate
  grounding in isolation. A cleaner isolation (text obs, but *coordinate* actions) is future work.
- **Single model / seed, greedy.** *Caveat (unverified):* whether a **larger model** closes the
  gap — on the vision side or the text-comparison side — is **not yet tested** (a scale run was
  blocked on disk). Even at temp 0, run-to-run variance is ~2–3pp (vLLM batching).
- **Fairness fix:** the text view initially didn't expose the sites' **filter/sort vocabularies**
  (which the vision agent sees as dropdown options), so text hallucinated invalid filters. We
  surfaced the real allow-lists from the transition functions before the final run; it restored
  parity but didn't move the aggregate (the bottleneck is comparison, not filtering).

## Takeaway

- **In web-agent tasks, the bottleneck is visual GUI understanding, not task reasoning.** Handing
  a 9B the *same* tasks as text instead of pixels **~doubles** success (28% → 68%). What's left of
  the text agent's failures is **multi-turn execution** (looping), not perception.

## Code

- <a href="/assets/code/posts/the-visual-understanding-tax-a-9b-web-agent-text-vs-pixels/make_figure.py" download><code>make_figure.py</code></a>
