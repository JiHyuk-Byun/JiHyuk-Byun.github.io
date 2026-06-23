---
title: "Upstage Products Evaluation — In-the-wild 1: Paper Review (Document Parse × Solar-pro3)"
date: 2026-06-19 15:00:00
slug: upstage-eval-paper-review
tags: [upstage, document-parse, solar, e2e, practical]
categories: [experiments]
layout: post
---

> **In-the-wild 1 / Document Parse × Solar-pro3** — an end-to-end document task.

## What I tested & how
This is the first of two experiments that **chain Upstage's document models with Solar-pro3** in one
pipeline. Here, **Document Parse** extracts a 47-page paper (tables as HTML), then **Solar-pro3** reads
that and writes a weakness-diagnosis report for one agent (OpenAI CUA), with supporting tables and quotes.

I deliberately added **no hallucination-suppressing constraints** — I wanted to see the model as-is —
and then checked two things separately:

1. **Conversion boundary** — how much of the paper's table / chart / figure actually reaches a text LLM?
2. **Grounding vs. reasoning** — does Solar **cite the numbers correctly** (checked automatically against
   the real tables), and is the **diagnosis built on those numbers sound** (checked by hand)?

## Setup
- **Task:** Parse-extracted paper (tables = HTML) → Solar diagnoses **OpenAI CUA**'s weak skills and proposes dataset augmentation, **with supporting tables and quotes**. *No hallucination-suppressing constraints — observe performance as-is.*
- **Pipeline:** `document-parse` (47 pp) → `solar-pro3` (temp 0). Figures are excluded from the text path (Solar can't see images).
- **Dataset:** `2026webstep.pdf` (47 pp) — 23 tables · 7 charts · 25 figures.
- **Input form:** tables are `<table>` HTML preserving even merged headers (rowspan/colspan), so the LLM can attribute *cell ↔ value*.

	{% include figure.liquid loading="lazy" path="assets/img/posts/upstage-eval-paper-review/structuring.png" class="img-fluid rounded z-depth-1" zoomable=true %}

- **Metric:** the **match rate of CUA-row numbers against the real table values (grounding)** + a **manual check** of semantic/reasoning hallucination.
- **Output:** `results/report/webstep_review.md`, `exp4_table_fidelity.json`.

## Result 1 — Conversion boundary: tables work, charts and figures don't
{% include figure.liquid loading="lazy" path="assets/img/posts/upstage-eval-paper-review/boundary_extract.png" class="img-fluid rounded z-depth-1" zoomable=true %}

**The only thing that reaches a text LLM intact is the table.**

| Element | What Parse produces | Can a text LLM use it |
|---|---|---|
| Table | HTML table preserving merges | ✅ usable as-is (numeric grounding 100%) |
| Chart | reads values but collapses to one row (`item_01`) | ⚠ numbers only, no series/hierarchy |
| Figure | image + OCR of text inside | ❌ the model can't see it |

## Result 2 — The table's numbers are trustworthy (grounding 100%)
**Every one of the 25 numbers the report attributed to CUA matched the real table values — zero invented.**
- It reconstructed the per-site and information-access tables with their actual values, and even cited real passages from the paper.

## Result 3 — But the reasoning hallucinates
{% include figure.liquid loading="lazy" path="assets/img/posts/upstage-eval-paper-review/hallucination_compare.png" class="img-fluid rounded z-depth-1" zoomable=true %}

**The numbers are all correct, but the judgments drawn from them are wrong.** So the value-only automatic
check (grounding 100%) catches none of the errors below.

| #      | Type     | What's wrong (vs. the table values) |
| ------ | ------ | ------------------------------------------------------------------------------------- |
| **H1** | False weakness  | Calls Search a weakness, when it's actually CUA's **strongest** skill (94.5%). The conclusion then contradicts itself, calling it "high." |
| **H2** | Mis-cited evidence | The passage cited for the weakness doesn't mention search at all. Same passage repeated 6 times. |
| **H3** | Category confusion | Table 17's "Filter" is an information-access level, but it's mistaken for the Filter **skill**. |
| **H4** | Fabricated meaning | Says "combined success rate 69.8%," but 69.8 is a Card-level score. The number is right and the **label is fake**. |
| **H5** | Repetition | Same table 3×, same quote 6×, duplicated rows in the reconstructed table. |
| H6     | (minor)   | A "(Figure 3)" reference to an image it can't see is left in. |

- **This is not a hallucination:** the Filter weakness is backed by both the actual low values (Shopping 21 · Accommodation 69) and the paper's text.

## Finding
- **Tables = the trustworthy axis.**
	- Thanks to `<table>` HTML, Solar's numeric citations are **100% grounded**, zero invented.
- **Reasoning = hallucination.**
	- Errors in **selecting** the weakness, **connecting** the evidence, and **assigning meaning** to numbers
		→ **value-checking can't catch this; you need a separate check on the validity of the reasoning.** (the core point of this post)
- **Charts/figures are unfit as data sources.**
	- charts lose the series/legend hierarchy (7/7 `item_01`); figures are images, so they don't survive the text path.
- **The boundary in one line:** tables work because their structure is **explicit in the document**; charts/figures don't, because their structure is **visual**.

## Appendix — Code · Dataset
**Code (single notebook):**
<a href="/assets/code/posts/upstage-eval-paper-review/exp3_e2e_report.ipynb" download><code>exp3_e2e_report.ipynb</code></a>
- Shared module: `code/upstage_eval.py` (`document_parse` · `solar_chat`).

**Dataset:** the WebStep benchmark paper (47 pp) — the original isn't redistributed for copyright reasons.
- Parse cache `results/parse/webstep_full.json` · report `results/report/webstep_review.md` · grounding `results/exp4_table_fidelity.json`
