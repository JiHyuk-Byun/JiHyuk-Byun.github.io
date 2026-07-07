---
title: "Does a VLM web agent even need the GUI?"
date: 2026-07-08
tags: [web-agents, multimodal, evaluation, grounding]
categories: [experiments]
layout: post
---

## Question — does a web agent even need the (human-facing) GUI?

- A GUI is designed for **humans**. But a web agent is a **VLM** — it *can* see, yet is the GUI
  the right interface for it, or just an accident of the app having been built for people?
- **[Webstep](https://jiwanchung.github.io/webstep/)** lets us ask this cleanly: it models each web
  app as a deterministic **MDP** (state, fixed action set, pure transitions), where the GUI is only
  *one rendering* of that MDP.
- Decouple the GUI from the MDP and you can render the *same* MDP two ways — a **human-friendly GUI**
  (pixels) or a **VLM-friendly text UI** (state + available actions). Same information, only the
  modality differs.
- **Main question:** what is the **tax** a VLM pays for going through the human GUI instead of a
  text-only UI? Equivalently — is web-agent performance bottlenecked by **reading the GUI**, not by
  the task reasoning?

## Setup

- **Model:** `Qwen3.5-9B` (unified multimodal), one vLLM server, **greedy** decoding (temp 0).
- **Benchmark:** Webstep — deterministic MDP web tasks, **10 sites, 220 tasks**. Both conditions
  run the *identical* task set (fixed seed), same 3-step memory window, same per-episode cap.
- **text-MDP condition:** observation = the site's `observe()` output serialized to text
  (visibility-filtered → **the same information the screen shows**); actions = semantic
  `{type, payload}` applied **directly through the MDP** (`dispatch`), no pixels, no coordinates.
- **vision condition:** observation = **screenshot**; actions = **pixel coordinates**.
- Only the **modality** differs; the model, tasks, decoding, and judge are held fixed.

## Result 1 — GUI understanding, not reasoning, is the bottleneck: **+40pp**

{% include figure.liquid loading="lazy" path="assets/img/posts/does-a-vlm-web-agent-even-need-the-gui/tax_by_site.png" class="img-fluid rounded z-depth-1" zoomable=true %}

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

- **A web agent pays a large tax for the human-facing GUI.** Rendering the *same* MDP as an
  VLM-friendly **text UI** instead of pixels **~doubles** a 9B's success (28% → 68%, **+40pp**) —
  the bottleneck is **reading the GUI, not the task reasoning.** What's left of the text agent's
  failures is **multi-turn execution** (looping), possibly just a 9B capability limit.

## Code

- <a href="/assets/code/posts/does-a-vlm-web-agent-even-need-the-gui/make_figure.py" download><code>make_figure.py</code></a>
