---
title: "Upstage Products Evaluation — Model Test 1: Document Parse"
date: 2026-06-19
slug: upstage-eval-doc-parse
tags: [upstage, document-ai, figure, chart]
categories: [experiments]
layout: post
---

> **Model Test 1 / Document Parse** — a controlled capability check.

## What I tested & how
Document Parse is Upstage's document-structuring model. I fed it **three single pages from a paper** — one with a merged-cell results table, one with quantitative charts, one with a pipeline diagram — and split the evaluation into two tracks:

- **Track A · Localization** — *where* is each element? I overlay the predicted boxes on the page and
  check them against hand-counted element counts (recall).
- **Track B · Understanding** — *how* does it rewrite each element into machine-readable HTML? I compare
  the chart structuring against the original charts read by eye.

Document Parse takes **no natural-language prompt** — only parameters — so this measures the model's
built-in behavior, not prompt engineering. The concrete question: does it convert chart → data and
diagram → structure, or just detect them?

## Setup
- **Model:** `document-parse`, `base64_encoding=['table','figure','chart','equation']`
- **Dataset (3 paper pages):** table_page (merged-cell results table), chart_page (quantitative charts), diagram_page (pipeline diagram)
- **Track A — Localization:** the model predicts bounding-box regions. `category` · `coordinates` (normalized polygon bbox) · `base64_encoding` (crop) → eyeballed via a bbox overlay.
- **Track B — Understanding:** the model rewrites elements into an LLM-understandable form in `content.html` (table/chart structuring, diagrams) rather than just a bbox.
- **Prompt used:** none (Parse takes parameters only) — `model=document-parse`, `base64_encoding=['table','figure','chart','equation']`. What to detect/structure is not specified in natural language.

## Result

### Track A — Localization: it catches the regions well
{% include figure.liquid loading="lazy" path="assets/img/posts/upstage-eval-doc-parse/exp1b_trackA_localization.png" class="img-fluid rounded z-depth-1" zoomable=true %}

**It catches every primary element — tables, charts, diagrams — 6 out of 6.**
- Caption detection is inconsistent, though: it gets the table caption but classifies the diagram caption as body text (`paragraph`) and misses it (1 out of 2).
- Each element comes back with its position coordinates and a cropped image.

### Track B — Understanding: charts keep the numbers but lose the structure
{% include figure.liquid loading="lazy" path="assets/img/posts/upstage-eval-doc-parse/exp1b_trackB_chart.png" class="img-fluid rounded z-depth-1" zoomable=true %}

**The numbers are read correctly, but almost all of the chart's structure is lost.**
- The original is a dot plot comparing two models (OpenAI CUA · Qwen3.5), but Parse keeps **only one series** (a single `item_01` row).
- It misclassifies the dot plot as line/bar, and flattens the Nav/Commit/Filter skill grouping.
- The same thing happens across all four charts (series lost 4/4, type misclassified 4/4). The values themselves are accurate.
- The diagram isn't structured at all — only an image placeholder is left.

## Finding
- **It provides both capture (crop) and documentation (HTML):**
	- the same element comes back as a bbox + a cropped image + structured HTML, all at once.
- **It does "attempt" chart→data structuring** (it produces a table)
	— but fidelity is low because of **chart-type confusion (dot→line) and multi-series loss**.
- **It doesn't convert diagrams into structure** (detection + label OCR only). For diagrams that carry no data, this seems like reasonable behavior.
- For bounding boxes, region separation is clean, **but the hierarchy isn't consistent** (visible in the difference between chart10 and chart7).
- Table structuring is achieved at close to 100%.

## Appendix — Code · Dataset
**Code (single notebook):**
<a href="/assets/code/posts/upstage-eval-doc-parse/exp1_figure_table.ipynb" download><code>exp1_figure_table.ipynb</code></a>
- Shared modules: `code/upstage_eval.py` (API wrapper) · `code/metrics.py` (scoring).

**Dataset (3 paper pages):** a merged-cell table page, a quantitative-chart page, and a pipeline-diagram page — single pages from the WebStep paper, not redistributed for copyright reasons (the figures above show their content).
- Shared outputs: detection crops `figures/1b_*_crops/` · bbox overlays `figures/1b_*_overlay.png` · scores `results/scores_exp1b.json`
