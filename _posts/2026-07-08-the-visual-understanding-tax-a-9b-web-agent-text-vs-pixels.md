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

## Result 1 — text beats pixels on *every* site: **+40pp overall**

{% include figure.liquid loading="lazy" path="assets/img/posts/the-visual-understanding-tax-a-9b-web-agent-text-vs-pixels/tax_by_site.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- **Overall: text-MDP 68% vs vision 28% → +40pp.** Removing the visual channel **~doubles** success.
- Per-site tax ranges **+10pp (ubereats) → +65pp (gcalendar)**; text wins everywhere.
- **Vision never clears 50%** on any site — the 9B can barely operate a GUI through pixels.

## Result 2 — the tax doesn't track *visual richness*

- The largest gaps are on **text-heavy** sites: **stackoverflow +55pp**, linkedin +50, github +45.
- Image-heavy shopping/listing sites aren't where vision suffers most (amazon +40, zillow +20).
- **Takeaway:** the bottleneck is **broad GUI grounding**, not any one flashy visual feature —
  the vision model is weak at *reading and acting on* interface layout across the board.

## Why — what "visual understanding" actually bundles

- The text condition removes the **whole GUI stack at once**: (1) **perception** (read the screen),
  (2) **affordance inference** (what's actionable), (3) **coordinate grounding** (where to click).
- So the +40pp is the **cost of that whole stack** — *not* pixel-coordinate grounding alone.
  Call it a **"visual-understanding tax,"** not a "grounding tax."
- **Text isn't perfect either.** It fails on **comparison / multi-turn tasks** (open N items,
  compare an attribute, pick the max): the 9B **loops** — re-issuing the same action instead of
  converging, even though its full action history is in context. So *"reasoning is fine, only
  vision is the problem"* is **too strong**: multi-turn orchestration is a genuine 9B limitation
  (worst on zillow/ubereats, where text is only 35–40%).

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

- **For a 9B web agent, going through the eyes costs ~40pp of task success (28% → 68%).**
  The planning is largely there; **wiring it to a GUI via pixels is where it breaks.**

## Code

- <a href="/assets/code/posts/the-visual-understanding-tax-a-9b-web-agent-text-vs-pixels/make_figure.py" download><code>make_figure.py</code></a>
