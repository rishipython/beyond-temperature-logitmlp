# Beyond Temperature — blog source

Source for the standalone research blog post **[Beyond Temperature: Rethinking fixed, global temperature and moving toward token-adaptive logit transformations](https://rishipython.github.io/beyond-temperature-logitmlp/)**, hosted on GitHub Pages.

The post introduces *LogitMLP*, a tiny learned head trained with RLVR on top of a frozen base model that picks its own piecewise-monotonic logit transform on every token. It recovers most of the gap between a temperature-optimized base model and full-parameter RLVR fine-tuning on MATH-500, and breaks the global-temperature accuracy/diversity Pareto frontier when stacked on top of an already-RLVR-tuned checkpoint.

Authors: Timothy Gao, Alex Luu, Harsha Polavaram, David Yang, Aryan Bansal, Rishi Athavale (UC Berkeley EE290).

## Layout

```
.
├── index.html                # the post (single-page)
├── styles/main.css           # all CSS
├── scripts/widgets.js        # all interactive widgets (vanilla JS + inline SVG)
├── assets/                   # PNG figures (paper + analysis)
├── data/
│   ├── qualitative_examples.json   # examples used by Figure 4
│   ├── extract_qualitative.py      # script that produced ^   (paths point to internal data)
│   └── regen_figures.py            # regenerates the divergence + per-token-T figures
└── .nojekyll                 # tells GitHub Pages to serve files as-is (no Jekyll)
```

Hand-written HTML/CSS/JS — no framework, no build step, no Jekyll. Math is rendered with [KaTeX](https://katex.org/) (CDN). Fonts are Source Serif Pro, Inter, and JetBrains Mono from Google Fonts.

## Running locally

Anything that serves static files works:

```bash
python3 -m http.server 8000
# then open http://localhost:8000/
```

KaTeX and fonts come from CDNs, so a network connection is required on first load (it caches afterwards).

## Editing

* **Prose** lives in `index.html`. Inline math uses `\(...\)` or `$...$`, display math uses `$$...$$`.
* **Styling** lives in `styles/main.css`. The page is a Distill-style centered column (`var(--col-text)`) with figures that escape to a wider column via the `wide` class.
* **Interactive widgets** live in `scripts/widgets.js`. Four widgets, each attached by `id`:
  * `#piecewise-widget`   — live `(s_1, s_2, b)` slider, with presets for temperature / top-k / top-p / anti-overconfidence and two presets taken from real high-KL tokens of the trained LogitMLP heads.
  * `#pareto-widget`      — best@1 vs. best@8 scatter with hover tooltips.
  * `#tvslm-widget`       — side-by-side global-T vs. LogitMLP comparison with live KL divergence.
  * `#qualitative-widget` — fetches `data/qualitative_examples.json` and shows top-6 candidates under base and LogitMLP for real Qwen3-1.7B rollouts.

## Reproducibility scripts

The two scripts under `data/` document how the per-token data and analysis figures were produced. They assume access to the parent research repo (`extra_layer_rlvr`) with the trained checkpoints and per-position `.pt` files, and **are not directly runnable from this public mirror** — the file paths inside them point to the original workspace. They're included here as a reference for how each artifact in the post was generated.

## License

Content of the post is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Code (HTML/CSS/JS/Python utilities) is released under the MIT license.

## Citation

```bibtex
@article{logitmlp2026,
  title  = "Beyond Temperature: Learned Token-Adaptive Logit Transformations with LogitMLP",
  author = "Gao, Timothy and Luu, Alex and Polavaram, Harsha and Yang, David and Bansal, Aryan and Athavale, Rishi",
  year   = "2026",
  note   = "UC Berkeley EE290 (Scalable AI) class project"
}
```
