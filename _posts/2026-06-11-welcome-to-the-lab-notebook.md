---
layout: post
title: "Welcome to the lab notebook"
date: 2026-06-11
description: "Why this blog exists, and a quick tour of the formatting I'll use to write up small experiments."
tags: [experiments, workflow]
categories: [meta]
toc:
  beginning: true
related_posts: false
published: false # kept as a reference/template; not built or shown on the blog. Set to true (or remove) to publish.
---

This blog is my lab notebook. Whenever I have an idea or a toy I want to
sanity-check, I run a small experiment, and then write up what I found here —
short, honest, and reproducible. Posts are not meant to be polished papers;
they are notes to my future self (and anyone curious).

This first post does double duty: it explains the format, and it shows every
building block I'll reuse in later write-ups, so I can copy it as a template.

## How a typical post is structured

Most experiment notes follow the same arc:

1. **Question** — the single thing I want to know.
2. **Setup** — dataset, model, optimizer, and what I logged.
3. **Result** — one or two figures and the takeaway.
4. **Open questions** — what I'd test next.

Keeping the arc fixed makes posts fast to write and fast to skim.

## Math

Inline math works, e.g. the $$\ell_2$$ norm $$\lVert w \rVert_2$$, and so do
display equations:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \big(f_\theta(x_i) - y_i\big)^2
+ \lambda \lVert \theta \rVert_2^2 .
$$

## Code

Code blocks are syntax-highlighted:

```python
import torch

def l2_penalty(model: torch.nn.Module) -> torch.Tensor:
    return sum(p.pow(2).sum() for p in model.parameters())
```

## Figures

Figures use the responsive `figure` include, which generates `webp` variants at
build time:

{% include figure.liquid loading="eager" path="assets/img/9.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}

Replace the `path` with a plot exported from the experiment (e.g.
`assets/img/2026-06-11-loss-curve.png`).

## Tables and quotes

| setting        | value         |
| -------------- | ------------- |
| optimizer      | AdamW         |
| learning rate  | 3e-4          |
| weight decay   | 0.1           |

> A note worth remembering goes in a blockquote — for instance a surprising
> result or a caveat about how the experiment was run.

## Open questions

- Does the effect survive a larger model?
- Is it an artifact of the logging interval?

That's the whole toolbox. The next post will be a real experiment.
