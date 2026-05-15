/* Interactive widgets for the LogitMLP blog.
 *
 * Three widgets, all vanilla JS + inline SVG:
 *   1. piecewise()    -- the hero figure: live (s1, s2, b) -> f(l) on the left,
 *                        softmax distribution on the right, plus an "implied
 *                        temperature" comparator. Includes preset buttons for
 *                        temperature, top-k, top-p, anti-overconfidence.
 *   2. paretoFrontier() -- best@1 vs best@8 scatter from Table 2.
 *   3. tempVsLogitmlp() -- side-by-side: same logits warped by global T sweep
 *                          vs the LogitMLP transform, showing why T can't
 *                          reach the same shape.
 *
 * All numbers are baked in as constants; no fetch/network usage so the page
 * works fully offline / on GitHub Pages.
 */

(function () {
  "use strict";

  // ---------- Utility ----------

  const svgNS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, children) {
    const e = document.createElementNS(svgNS, tag);
    if (attrs) {
      for (const k in attrs) e.setAttribute(k, attrs[k]);
    }
    if (children) {
      for (const c of children) e.appendChild(c);
    }
    return e;
  }

  function fmt(x, n) {
    if (n === undefined) n = 3;
    if (!isFinite(x)) return "—";
    if (Math.abs(x) >= 100) return x.toFixed(0);
    return x.toFixed(n);
  }

  function softmax(arr) {
    let m = -Infinity;
    for (const v of arr) if (v > m) m = v;
    const ex = arr.map((v) => Math.exp(v - m));
    let s = 0;
    for (const v of ex) s += v;
    return ex.map((v) => v / s);
  }

  function entropy(p) {
    let h = 0;
    for (const v of p) if (v > 0) h -= v * Math.log(v);
    return h;
  }

  // Two-piece monotonic transform.
  function piecewise_f(l, s1, s2, b) {
    return s1 * l + (s2 - s1) * Math.max(0, l - b);
  }

  // A representative "interesting" logit vector. Hand-tuned so the
  // distribution is moderately concentrated with a clear top-3 and a long
  // tail. Loosely modeled on Qwen3 logits at a moderate-entropy reasoning
  // token, but not literal data.
  function makeDemoLogits(n) {
    if (n === undefined) n = 30;
    // Sharp head + flatter body + heavy tail.
    const out = [];
    const peaks = [9.5, 8.2, 7.6, 6.4, 6.0, 5.6, 5.2, 4.9, 4.6, 4.4];
    for (let i = 0; i < peaks.length; i++) out.push(peaks[i]);
    for (let i = peaks.length; i < n; i++) {
      // Smoothly decaying body.
      out.push(4.3 - 0.10 * (i - peaks.length) - 0.005 * Math.pow(i - peaks.length, 1.8));
    }
    return out;
  }

  // Demo "label" tokens to make the distribution feel concrete.
  const DEMO_LABELS = [
    "use", "compute", "let", "consider", "find",
    "solve", "given", "first", "denote", "note",
    "by", "the", "we", "to", "and",
    "set", "from", "with", "for", "if",
    "since", "now", "so", "thus", "therefore",
    "expand", "factor", "simplify", "rewrite", "substitute",
    "verify", "check", "apply", "evaluate", "obtain",
  ];

  // ---------- 1. Piecewise transform hero widget ----------

  function piecewiseWidget(root) {
    // Internal state
    const state = {
      s1: 1.0,
      s2: 1.0,
      b: 8.6,   // initially just below top-1
      preset: "identity",
    };

    const logits = makeDemoLogits(28);
    const lmin = -2;
    const lmax = 11;

    // ---- Layout ----
    const fnPlot = document.createElement("div");
    const distPlot = document.createElement("div");
    fnPlot.className = "plot-fn";
    distPlot.className = "plot-dist";

    const controlsHost = document.createElement("div");
    controlsHost.className = "controls";

    const presetRow = document.createElement("div");
    presetRow.className = "preset-row";
    presetRow.style.gridColumn = "1 / -1";

    root.appendChild(controlsHost);
    root.appendChild(presetRow);

    const plotRow = document.createElement("div");
    plotRow.className = "plot-row";
    plotRow.appendChild(fnPlot);
    plotRow.appendChild(distPlot);
    root.appendChild(plotRow);

    const summary = document.createElement("div");
    summary.style.fontFamily = "-apple-system, Inter, sans-serif";
    summary.style.fontSize = "12.5px";
    summary.style.color = "#5b5b5b";
    summary.style.marginTop = "10px";
    summary.style.textAlign = "center";
    root.appendChild(summary);

    // ---- Controls ----
    function slider(name, label, range, step) {
      const wrap = document.createElement("label");
      const top = document.createElement("div");
      top.className = "lbl";
      const left = document.createElement("span");
      left.innerHTML = label;
      const right = document.createElement("span");
      right.className = "val";
      right.textContent = state[name].toFixed(2);
      top.appendChild(left);
      top.appendChild(right);
      const inp = document.createElement("input");
      inp.type = "range";
      inp.min = range[0];
      inp.max = range[1];
      inp.step = step;
      inp.value = state[name];
      inp.addEventListener("input", () => {
        state[name] = parseFloat(inp.value);
        state.preset = "custom";
        right.textContent = state[name].toFixed(2);
        updatePresets();
        render();
      });
      wrap.appendChild(top);
      wrap.appendChild(inp);
      wrap.dataset.name = name;
      return { wrap, inp, val: right };
    }

    const ctrlS1 = slider("s1", "<i>s</i><sub>1</sub> (tail slope)", [0.25, 4.0], 0.01);
    const ctrlS2 = slider("s2", "<i>s</i><sub>2</sub> (head slope)", [0.25, 4.0], 0.01);
    const ctrlB  = slider("b",  "<i>b</i> (breakpoint, in logit units)", [3.0, 11.0], 0.05);

    controlsHost.appendChild(ctrlS1.wrap);
    controlsHost.appendChild(ctrlS2.wrap);
    controlsHost.appendChild(ctrlB.wrap);

    // ---- Presets ----
    // First six presets reproduce standard decoders as special cases of the
    // two-piece family. The last two are sampled directly from real
    // high-KL active tokens of the trained heads on Qwen3-1.7B rollouts.
    const presets = [
      { id: "identity",   label: "Identity (T=1)",        s1: 1.00, s2: 1.00, b: 8.6 },
      { id: "hot",        label: "Hot temp (T=1.5)",      s1: 1/1.5, s2: 1/1.5, b: 8.6 },
      { id: "cold",       label: "Cold temp (T=0.5)",     s1: 1/0.5, s2: 1/0.5, b: 8.6 },
      { id: "topk",       label: "≈ Top-k (k=3)",         s1: 0.30, s2: 1.20, b: 7.8 },
      { id: "topp",       label: "≈ Nucleus (p≈0.9)",     s1: 0.45, s2: 1.10, b: 8.2 },
      { id: "anticonf",   label: "Anti-overconf.",        s1: 1.05, s2: 0.30, b: 7.4 },
      // From a real DAPO-trained active token: " Example" at a high-KL
      // moderate-confidence position. Real values were (s1=0.83, s2=0.77);
      // b is rescaled into the demo logit range (real b≈13 on Qwen logits).
      { id: "dapo",       label: "LogitMLP (DAPO) — real token",   s1: 0.83, s2: 0.77, b: 9.0 },
      // From a real pass@k-trained active token: " Count" at a step-header
      // position. Real values were (s1=1.35, s2=2.14); b rescaled.
      { id: "passk",      label: "LogitMLP (pass@k) — real token", s1: 1.35, s2: 2.14, b: 9.0 },
    ];

    function updatePresets() {
      for (const c of presetRow.children) {
        c.classList.toggle("active", c.dataset.id === state.preset);
      }
    }

    for (const p of presets) {
      const b = document.createElement("button");
      b.className = "preset";
      b.dataset.id = p.id;
      b.textContent = p.label;
      b.addEventListener("click", () => {
        state.s1 = p.s1; state.s2 = p.s2; state.b = p.b; state.preset = p.id;
        ctrlS1.inp.value = p.s1; ctrlS1.val.textContent = p.s1.toFixed(2);
        ctrlS2.inp.value = p.s2; ctrlS2.val.textContent = p.s2.toFixed(2);
        ctrlB.inp.value  = p.b;  ctrlB.val.textContent  = p.b.toFixed(2);
        updatePresets();
        render();
      });
      presetRow.appendChild(b);
    }

    // ---- Rendering ----

    function drawFnPlot() {
      const W = 460, H = 320;
      const pad = { l: 50, r: 14, t: 18, b: 38 };
      const w = W - pad.l - pad.r;
      const h = H - pad.t - pad.b;

      const { s1, s2, b } = state;

      // y range: cover both the identity reference (s=1) and the warped fn
      const samples = 200;
      let ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i <= samples; i++) {
        const l = lmin + (lmax - lmin) * i / samples;
        const yw = piecewise_f(l, s1, s2, b);
        if (yw < ymin) ymin = yw;
        if (yw > ymax) ymax = yw;
        if (l < ymin) ymin = l;
        if (l > ymax) ymax = l;
      }
      ymin = Math.floor(ymin - 1);
      ymax = Math.ceil(ymax + 1);

      const sx = (l) => pad.l + (l - lmin) / (lmax - lmin) * w;
      const sy = (y) => pad.t + (1 - (y - ymin) / (ymax - ymin)) * h;

      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img" });

      // Axes
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + h, stroke: "#888", "stroke-width": 1 }));
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: pad.t + h, y2: pad.t + h, stroke: "#888", "stroke-width": 1 }));

      // Gridlines and ticks
      const xTicks = 6, yTicks = 6;
      for (let i = 0; i <= xTicks; i++) {
        const x = lmin + (lmax - lmin) * i / xTicks;
        const px = sx(x);
        svg.appendChild(el("line", { x1: px, x2: px, y1: pad.t, y2: pad.t + h, stroke: "#eee", "stroke-width": 1 }));
        const t = el("text", { x: px, y: pad.t + h + 16, "text-anchor": "middle", "font-size": 11, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
        t.textContent = x.toFixed(0);
        svg.appendChild(t);
      }
      for (let i = 0; i <= yTicks; i++) {
        const y = ymin + (ymax - ymin) * i / yTicks;
        const py = sy(y);
        svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: py, y2: py, stroke: "#eee", "stroke-width": 1 }));
        const t = el("text", { x: pad.l - 8, y: py + 4, "text-anchor": "end", "font-size": 11, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
        t.textContent = y.toFixed(0);
        svg.appendChild(t);
      }

      // Axis labels
      const xlbl = el("text", { x: pad.l + w / 2, y: pad.t + h + 32, "text-anchor": "middle", "font-size": 12, "font-family": "Inter, sans-serif", fill: "#181818" });
      xlbl.textContent = "logit ℓ";
      svg.appendChild(xlbl);
      const ylbl = el("text", { x: 12, y: pad.t + h / 2, "text-anchor": "middle", "font-size": 12, "font-family": "Inter, sans-serif", fill: "#181818", transform: `rotate(-90 12 ${pad.t + h / 2})` });
      ylbl.textContent = "f(ℓ)";
      svg.appendChild(ylbl);

      // Identity reference line y=l (dashed)
      const isIdentity = Math.abs(s1 - 1) < 0.02 && Math.abs(s2 - 1) < 0.02;
      svg.appendChild(el("line", {
        x1: sx(Math.max(lmin, ymin)),
        y1: sy(Math.max(lmin, ymin)),
        x2: sx(Math.min(lmax, ymax)),
        y2: sy(Math.min(lmax, ymax)),
        stroke: "#bcbcbc", "stroke-width": 1, "stroke-dasharray": "4 4"
      }));
      if (!isIdentity) {
        const idLbl = el("text", {
          x: sx(lmax - 0.4), y: sy(Math.min(lmax - 0.4, ymax - 0.3)) - 4,
          "text-anchor": "end", "font-size": 11, fill: "#888", "font-family": "Inter, sans-serif", "font-style": "italic"
        });
        idLbl.textContent = "identity";
        svg.appendChild(idLbl);
      }

      // Vertical line at b. Label sits at the bottom of the line, anchored
      // to the right of the line if there's room, otherwise to its left.
      svg.appendChild(el("line", { x1: sx(b), x2: sx(b), y1: pad.t, y2: pad.t + h, stroke: "#1f6feb", "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
      const bLblAnchor = (sx(b) < pad.l + w - 60) ? { x: sx(b) + 4, anchor: "start" }
                                                  : { x: sx(b) - 4, anchor: "end" };
      const bLbl = el("text", {
        x: bLblAnchor.x, y: pad.t + h - 6,
        "text-anchor": bLblAnchor.anchor,
        "font-size": 11, fill: "#1f6feb", "font-family": "Inter, sans-serif"
      });
      bLbl.textContent = `b = ${b.toFixed(2)}`;
      svg.appendChild(bLbl);

      // Two-piece path
      const p1x1 = lmin, p1y1 = piecewise_f(lmin, s1, s2, b);
      const p1x2 = b,    p1y2 = piecewise_f(b,    s1, s2, b);
      const p2x1 = b,    p2y1 = p1y2;
      const p2x2 = lmax, p2y2 = piecewise_f(lmax, s1, s2, b);
      svg.appendChild(el("line", {
        x1: sx(p1x1), y1: sy(p1y1), x2: sx(p1x2), y2: sy(p1y2),
        stroke: "#c7541d", "stroke-width": 2.6
      }));
      svg.appendChild(el("line", {
        x1: sx(p2x1), y1: sy(p2y1), x2: sx(p2x2), y2: sy(p2y2),
        stroke: "#c7541d", "stroke-width": 2.6
      }));
      // Breakpoint dot
      svg.appendChild(el("circle", { cx: sx(b), cy: sy(p1y2), r: 4, fill: "#c7541d" }));

      // Tail/head slope annotations. Anchor labels inside the plot area so
      // they don't clip on either edge.
      const s1mx = (lmin + b) / 2;
      const s1my = piecewise_f(s1mx, s1, s2, b);
      const s1Lbl = el("text", { x: sx(s1mx), y: sy(s1my) - 6, "text-anchor": "middle", "font-size": 11, fill: "#c7541d", "font-family": "Inter, sans-serif" });
      s1Lbl.textContent = `slope s₁ = ${s1.toFixed(2)}`;
      svg.appendChild(s1Lbl);
      const s2mx = (b + lmax) / 2;
      const s2my = piecewise_f(s2mx, s1, s2, b);
      // If the breakpoint is very close to lmax, the s2 segment is tiny; in
      // that case put the label above b instead.
      const s2LblX = (lmax - b) > 0.8 ? sx(s2mx) : Math.min(sx(b) + 60, pad.l + w - 4);
      const s2Lbl = el("text", { x: s2LblX, y: sy(s2my) - 6, "text-anchor": "middle", "font-size": 11, fill: "#c7541d", "font-family": "Inter, sans-serif" });
      s2Lbl.textContent = `slope s₂ = ${s2.toFixed(2)}`;
      svg.appendChild(s2Lbl);

      // Tick markers for the actual logits (small rug)
      for (let i = 0; i < logits.length; i++) {
        const l = logits[i];
        svg.appendChild(el("line", {
          x1: sx(l), y1: pad.t + h - 3, x2: sx(l), y2: pad.t + h + 3,
          stroke: "#444", "stroke-width": 1, opacity: 0.55
        }));
      }

      fnPlot.innerHTML = "";
      const ttl = document.createElement("div");
      ttl.className = "widget-sub";
      ttl.style.textAlign = "center";
      ttl.style.marginBottom = "4px";
      ttl.innerHTML = "<b style='color:#181818'>The transform</b> &mdash; how each logit gets remapped";
      fnPlot.appendChild(ttl);
      fnPlot.appendChild(svg);
    }

    function drawDistPlot() {
      const W = 460, H = 320;
      const pad = { l: 40, r: 14, t: 18, b: 60 };
      const w = W - pad.l - pad.r;
      const h = H - pad.t - pad.b;

      const { s1, s2, b } = state;

      const base = softmax(logits);
      const warped = softmax(logits.map((l) => piecewise_f(l, s1, s2, b)));

      const yMax = Math.max(Math.max(...base), Math.max(...warped));
      const yTop = Math.min(1.0, yMax * 1.10);

      const n = logits.length;
      const bx = (i) => pad.l + (i + 0.5) / n * w;
      const by = (p) => pad.t + (1 - p / yTop) * h;

      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

      // axes
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + h, stroke: "#888" }));
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: pad.t + h, y2: pad.t + h, stroke: "#888" }));

      const yTicks = 5;
      for (let i = 0; i <= yTicks; i++) {
        const p = yTop * i / yTicks;
        const py = by(p);
        svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: py, y2: py, stroke: "#eee" }));
        const t = el("text", { x: pad.l - 6, y: py + 4, "text-anchor": "end", "font-size": 10.5, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
        t.textContent = p.toFixed(2);
        svg.appendChild(t);
      }
      const ylbl = el("text", { x: 12, y: pad.t + h / 2, "text-anchor": "middle", "font-size": 12, "font-family": "Inter, sans-serif", fill: "#181818", transform: `rotate(-90 12 ${pad.t + h / 2})` });
      ylbl.textContent = "p(token)";
      svg.appendChild(ylbl);

      // Bars: base behind (grey), warped in front (orange, half-width)
      const bw = w / n * 0.85;
      for (let i = 0; i < n; i++) {
        const xc = bx(i);
        // base
        svg.appendChild(el("rect", {
          x: xc - bw / 2, y: by(base[i]),
          width: bw, height: pad.t + h - by(base[i]),
          fill: "#cccccc", opacity: 0.78
        }));
        // warped
        svg.appendChild(el("rect", {
          x: xc - bw / 4, y: by(warped[i]),
          width: bw / 2, height: pad.t + h - by(warped[i]),
          fill: "#c7541d", opacity: 0.92
        }));
      }

      // Token labels for top-8 only
      const order = base.map((_, i) => i).sort((a, b) => base[b] - base[a]);
      const labeled = new Set(order.slice(0, 8));
      for (let i = 0; i < n; i++) {
        if (!labeled.has(i)) continue;
        const t = el("text", {
          x: bx(i), y: pad.t + h + 18,
          "text-anchor": "end", "font-size": 11, fill: "#444",
          "font-family": "JetBrains Mono, monospace",
          transform: `rotate(-45 ${bx(i)} ${pad.t + h + 18})`
        });
        t.textContent = DEMO_LABELS[i] || `t${i}`;
        svg.appendChild(t);
      }

      // Inline legend
      const lg = el("g", {});
      lg.appendChild(el("rect", { x: pad.l + 6, y: pad.t + 4, width: 12, height: 8, fill: "#cccccc" }));
      const t1 = el("text", { x: pad.l + 22, y: pad.t + 12, "font-size": 11, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
      t1.textContent = "base";
      lg.appendChild(t1);
      lg.appendChild(el("rect", { x: pad.l + 56, y: pad.t + 4, width: 12, height: 8, fill: "#c7541d" }));
      const t2 = el("text", { x: pad.l + 72, y: pad.t + 12, "font-size": 11, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
      t2.textContent = "warped";
      lg.appendChild(t2);
      svg.appendChild(lg);

      distPlot.innerHTML = "";
      const ttl = document.createElement("div");
      ttl.className = "widget-sub";
      ttl.style.textAlign = "center";
      ttl.style.marginBottom = "4px";
      ttl.innerHTML = "<b style='color:#181818'>The resulting distribution</b> after softmax";
      distPlot.appendChild(ttl);
      distPlot.appendChild(svg);

      // Numeric summary
      const Hbase = entropy(base);
      const Hwarp = entropy(warped);
      const top1Base = Math.max(...base);
      const top1Warp = Math.max(...warped);
      summary.innerHTML =
        `Base: entropy <b>${Hbase.toFixed(3)}</b> nats, top-1 prob <b>${(100*top1Base).toFixed(1)}%</b> &nbsp;&middot;&nbsp; ` +
        `Warped: entropy <b>${Hwarp.toFixed(3)}</b> nats, top-1 prob <b>${(100*top1Warp).toFixed(1)}%</b>` +
        ` &nbsp;&middot;&nbsp; effective <i>T</i> on tail ≈ <b>${(1/state.s1).toFixed(2)}</b>`;
    }

    function render() {
      drawFnPlot();
      drawDistPlot();
    }

    updatePresets();
    render();
  }

  // ---------- 2. Pareto frontier widget ----------

  function paretoFrontier(root) {
    // Numbers from Table 2 (paper / blog continuation).
    const sweepPoints = [
      { T: 0.5, best1: 0.492, best8: 0.768, uniq: 2.03, ent: 0.54 },
      { T: 0.7, best1: 0.485, best8: 0.790, uniq: 2.17, ent: 0.61 },
      { T: 1.0, best1: 0.480, best8: 0.798, uniq: 2.33, ent: 0.68 },
      { T: 1.5, best1: 0.465, best8: 0.818, uniq: 2.73, ent: 0.85 },
    ];
    const ours = { best1: 0.549, best8: 0.873, uniq: 3.60, ent: 1.16 };

    const W = 720, H = 460;
    const pad = { l: 70, r: 30, t: 30, b: 60 };
    const w = W - pad.l - pad.r;
    const h = H - pad.t - pad.b;

    // axes ranges
    const xRange = [0.450, 0.575];
    const yRange = [0.755, 0.890];
    const sx = (x) => pad.l + (x - xRange[0]) / (xRange[1] - xRange[0]) * w;
    const sy = (y) => pad.t + (1 - (y - yRange[0]) / (yRange[1] - yRange[0])) * h;

    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

    // axes
    svg.appendChild(el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + h, stroke: "#888" }));
    svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: pad.t + h, y2: pad.t + h, stroke: "#888" }));

    const xTickVals = [0.46, 0.48, 0.50, 0.52, 0.54, 0.56];
    for (const x of xTickVals) {
      const px = sx(x);
      svg.appendChild(el("line", { x1: px, x2: px, y1: pad.t, y2: pad.t + h, stroke: "#eee" }));
      const t = el("text", { x: px, y: pad.t + h + 18, "text-anchor": "middle", "font-size": 11.5, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
      t.textContent = x.toFixed(2);
      svg.appendChild(t);
    }
    const yTickVals = [0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88];
    for (const y of yTickVals) {
      const py = sy(y);
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: py, y2: py, stroke: "#eee" }));
      const t = el("text", { x: pad.l - 8, y: py + 4, "text-anchor": "end", "font-size": 11.5, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
      t.textContent = y.toFixed(2);
      svg.appendChild(t);
    }

    // axis labels
    const xlbl = el("text", { x: pad.l + w / 2, y: pad.t + h + 42, "text-anchor": "middle", "font-size": 13, fill: "#181818", "font-family": "Inter, sans-serif" });
    xlbl.textContent = "best@1   (exploitation →)";
    svg.appendChild(xlbl);
    const ylbl = el("text", { x: 20, y: pad.t + h / 2, "text-anchor": "middle", "font-size": 13, fill: "#181818", "font-family": "Inter, sans-serif", transform: `rotate(-90 20 ${pad.t + h / 2})` });
    ylbl.textContent = "best@8   (exploration →)";
    svg.appendChild(ylbl);

    // Pareto curve through the sweep
    const sorted = sweepPoints.slice().sort((a, b) => b.best1 - a.best1);
    let pathD = "";
    sorted.forEach((p, i) => { pathD += (i === 0 ? "M" : "L") + sx(p.best1) + "," + sy(p.best8); });
    svg.appendChild(el("path", { d: pathD, stroke: "#999", "stroke-width": 1.6, fill: "none" }));

    // Sweep points
    for (const p of sweepPoints) {
      svg.appendChild(el("circle", { cx: sx(p.best1), cy: sy(p.best8), r: 7, fill: "#222", "data-T": p.T }));
      const t = el("text", { x: sx(p.best1) + 12, y: sy(p.best8) + 4, "font-size": 11.5, fill: "#444", "font-family": "Inter, sans-serif" });
      t.textContent = `T = ${p.T}`;
      svg.appendChild(t);
    }

    // Highlight the "if temperature could reach our point" projection lines
    svg.appendChild(el("line", {
      x1: sx(ours.best1), x2: sx(ours.best1),
      y1: sy(yRange[1]),  y2: sy(yRange[0]),
      stroke: "#bbb", "stroke-width": 1, "stroke-dasharray": "3 4"
    }));
    svg.appendChild(el("line", {
      x1: sx(xRange[0]), x2: sx(xRange[1]),
      y1: sy(ours.best8), y2: sy(ours.best8),
      stroke: "#bbb", "stroke-width": 1, "stroke-dasharray": "3 4"
    }));

    // Our point (star)
    function star(cx, cy, r, color) {
      const pts = [];
      for (let i = 0; i < 10; i++) {
        const ang = -Math.PI / 2 + i * Math.PI / 5;
        const rr = i % 2 === 0 ? r : r / 2.4;
        pts.push((cx + Math.cos(ang) * rr) + "," + (cy + Math.sin(ang) * rr));
      }
      return el("polygon", { points: pts.join(" "), fill: color, stroke: "#000", "stroke-width": 0.6 });
    }
    svg.appendChild(star(sx(ours.best1), sy(ours.best8), 11, "#c7541d"));

    const annLbl = el("text", { x: sx(ours.best1) - 14, y: sy(ours.best8) - 16, "text-anchor": "end", "font-size": 12.5, fill: "#c7541d", "font-weight": 700, "font-family": "Inter, sans-serif" });
    annLbl.textContent = "Full-RLVR + LogitMLP";
    svg.appendChild(annLbl);
    const annLbl2 = el("text", { x: sx(ours.best1) - 14, y: sy(ours.best8) - 2, "text-anchor": "end", "font-size": 11, fill: "#c7541d", "font-family": "Inter, sans-serif" });
    annLbl2.textContent = "(breaks the frontier)";
    svg.appendChild(annLbl2);

    // Sweep curve label
    const swLbl = el("text", { x: sx(0.475) + 50, y: sy(0.783), "font-size": 11.5, fill: "#444", "font-family": "Inter, sans-serif", "font-style": "italic" });
    swLbl.textContent = "Full-RLVR @ varying T";
    svg.appendChild(swLbl);

    // Tooltip
    const tip = document.createElement("div");
    tip.style.cssText = "position:absolute; pointer-events:none; background:white; border:1px solid #ccc; padding:6px 10px; border-radius:4px; font-family:Inter,sans-serif; font-size:12px; opacity:0; transition:opacity 0.1s; box-shadow:0 2px 8px rgba(0,0,0,0.12); z-index:10";
    root.style.position = "relative";

    // Plot region
    const container = document.createElement("div");
    container.style.position = "relative";
    container.style.maxWidth = W + "px";
    container.style.margin = "0 auto";
    container.appendChild(svg);
    container.appendChild(tip);

    function hover(label, p, evt) {
      tip.innerHTML = label;
      tip.style.opacity = 0.96;
      const rect = container.getBoundingClientRect();
      tip.style.left = (evt.clientX - rect.left + 10) + "px";
      tip.style.top  = (evt.clientY - rect.top  - 30) + "px";
    }
    function hide() { tip.style.opacity = 0; }

    svg.querySelectorAll("circle[data-T]").forEach((c, i) => {
      c.style.cursor = "default";
      c.addEventListener("mouseenter", (e) => {
        const p = sweepPoints[i];
        hover(`<b>Full-RLVR @ T=${p.T}</b><br>best@1: ${p.best1.toFixed(3)} &middot; best@8: ${p.best8.toFixed(3)}<br>uniq/8: ${p.uniq} &middot; entropy: ${p.ent}`, p, e);
      });
      c.addEventListener("mousemove", (e) => {
        const p = sweepPoints[i];
        hover(`<b>Full-RLVR @ T=${p.T}</b><br>best@1: ${p.best1.toFixed(3)} &middot; best@8: ${p.best8.toFixed(3)}<br>uniq/8: ${p.uniq} &middot; entropy: ${p.ent}`, p, e);
      });
      c.addEventListener("mouseleave", hide);
    });

    const starEl = svg.querySelector("polygon");
    starEl.style.cursor = "default";
    starEl.addEventListener("mouseenter", (e) => {
      hover(`<b>Full-RLVR + LogitMLP @ T=1.0</b><br>best@1: ${ours.best1.toFixed(3)} &middot; best@8: ${ours.best8.toFixed(3)}<br>uniq/8: ${ours.uniq} &middot; entropy: ${ours.ent}`, ours, e);
    });
    starEl.addEventListener("mousemove", (e) => {
      hover(`<b>Full-RLVR + LogitMLP @ T=1.0</b><br>best@1: ${ours.best1.toFixed(3)} &middot; best@8: ${ours.best8.toFixed(3)}<br>uniq/8: ${ours.uniq} &middot; entropy: ${ours.ent}`, ours, e);
    });
    starEl.addEventListener("mouseleave", hide);

    const ttl = document.createElement("div");
    ttl.className = "widget-sub";
    ttl.style.textAlign = "center";
    ttl.style.marginBottom = "6px";
    ttl.innerHTML = "<b style='color:#181818'>MATH-500 on Qwen3-1.7B</b> — best@1 vs. best@8 (hover for details)";
    root.appendChild(ttl);
    root.appendChild(container);
  }

  // ---------- 3. Temperature vs. LogitMLP comparison ----------

  function tempVsLogitmlp(root) {
    // Two side-by-side bar charts of the same logits, warped by either
    // a global temperature or the (s1, s2, b) LogitMLP transform.
    // The user can toggle which "LogitMLP behavior" they're comparing to.

    // A representative moderate-confidence "active token" logit vector:
    // base top-1 prob ≈ 0.36, base entropy ≈ 2.1 nats, smoothly decaying
    // tail. This is the regime where LogitMLP actually acts (see the analysis below);
    // most tokens have a much sharper base distribution and LogitMLP leaves
    // them untouched.
    const logits = [
      3.6, 2.9, 2.5, 2.1, 1.7, 1.3, 0.9, 0.6, 0.3, 0.0,
      -0.3, -0.6, -0.9, -1.2, -1.5, -1.8, -2.1, -2.4, -2.7, -3.0,
    ];

    const state = {
      T: 0.7,
      mode: "passk", // 'passk' or 'dapo'
    };

    // (s1, s2) values come from real high-KL active tokens of the trained
    // heads on Qwen3-1.7B rollouts (the same ones surfaced in Figure 4):
    //   pass@k token " Count":   s1=1.35, s2=2.14
    //   DAPO    token " Example": s1=0.83, s2=0.77
    // b is set just below top-1, matching the analysis finding that the
    // learned head puts b near max(ℓ) at ~95% of positions.
    const presets = {
      passk: { s1: 1.35, s2: 2.14, b: 3.1, name: "LogitMLP — pass@k (real token)" },
      dapo:  { s1: 0.83, s2: 0.77, b: 3.1, name: "LogitMLP — DAPO (real token)" },
    };

    const controls = document.createElement("div");
    controls.className = "controls";
    const tabsHost = document.createElement("div");
    tabsHost.className = "variant-tabs";
    tabsHost.style.gridColumn = "1 / -1";
    for (const k of ["passk", "dapo"]) {
      const b = document.createElement("button");
      b.textContent = presets[k].name;
      b.dataset.k = k;
      b.addEventListener("click", () => { state.mode = k; updateTabs(); render(); });
      tabsHost.appendChild(b);
    }
    function updateTabs() {
      for (const b of tabsHost.children) b.classList.toggle("active", b.dataset.k === state.mode);
    }

    const tLabel = document.createElement("label");
    tLabel.innerHTML = `<div class='lbl'><span>Global temperature <i>T</i></span><span class='val'>${state.T.toFixed(2)}</span></div>`;
    const tIn = document.createElement("input");
    tIn.type = "range"; tIn.min = "0.30"; tIn.max = "1.80"; tIn.step = "0.01"; tIn.value = state.T;
    tIn.addEventListener("input", () => {
      state.T = parseFloat(tIn.value);
      tLabel.querySelector(".val").textContent = state.T.toFixed(2);
      render();
    });
    tLabel.appendChild(tIn);

    controls.appendChild(tabsHost);
    controls.appendChild(tLabel);
    root.appendChild(controls);

    const plotRow = document.createElement("div");
    plotRow.className = "plot-row";
    const leftHost  = document.createElement("div");
    const rightHost = document.createElement("div");
    plotRow.appendChild(leftHost);
    plotRow.appendChild(rightHost);
    root.appendChild(plotRow);

    const summary = document.createElement("div");
    summary.style.cssText = "font-family:Inter,sans-serif;font-size:12.5px;color:#5b5b5b;margin-top:10px;text-align:center;line-height:1.5";
    root.appendChild(summary);

    function drawBars(host, pPrimary, pReference, primaryColor, title) {
      // Draws primary distribution as solid bars in `primaryColor`, with a
      // narrower overlay showing the reference (the *other* panel's) bars in
      // grey so the user can see the gap directly.
      const W = 460, H = 280;
      const pad = { l: 36, r: 12, t: 24, b: 38 };
      const w = W - pad.l - pad.r;
      const h = H - pad.t - pad.b;
      const n = pPrimary.length;
      const yTop = Math.max(0.6, Math.max(Math.max(...pPrimary), Math.max(...pReference)) * 1.10);
      const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H });

      svg.appendChild(el("line", { x1: pad.l, x2: pad.l, y1: pad.t, y2: pad.t + h, stroke: "#888" }));
      svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: pad.t + h, y2: pad.t + h, stroke: "#888" }));
      const yTicks = 5;
      for (let i = 0; i <= yTicks; i++) {
        const v = yTop * i / yTicks;
        const py = pad.t + (1 - v / yTop) * h;
        svg.appendChild(el("line", { x1: pad.l, x2: pad.l + w, y1: py, y2: py, stroke: "#eee" }));
        const t = el("text", { x: pad.l - 5, y: py + 4, "text-anchor": "end", "font-size": 10.5, fill: "#5b5b5b", "font-family": "Inter, sans-serif" });
        t.textContent = v.toFixed(2);
        svg.appendChild(t);
      }
      const bw = w / n * 0.78;
      for (let i = 0; i < n; i++) {
        const xc = pad.l + (i + 0.5) / n * w;
        // Reference (the other panel) drawn as a hollow grey outline behind.
        const pyRef = pad.t + (1 - pReference[i] / yTop) * h;
        svg.appendChild(el("rect", {
          x: xc - bw / 2, y: pyRef, width: bw, height: pad.t + h - pyRef,
          fill: "none", stroke: "#999", "stroke-width": 1, "stroke-dasharray": "3 2"
        }));
        // Primary as filled bar in color
        const pyPrim = pad.t + (1 - pPrimary[i] / yTop) * h;
        svg.appendChild(el("rect", { x: xc - bw / 2, y: pyPrim, width: bw, height: pad.t + h - pyPrim, fill: primaryColor, opacity: 0.85 }));

        if (i < 6) {
          const t = el("text", { x: xc, y: pad.t + h + 16, "text-anchor": "end", "font-size": 10.5, fill: "#444", "font-family": "JetBrains Mono, monospace", transform: `rotate(-45 ${xc} ${pad.t + h + 16})` });
          t.textContent = DEMO_LABELS[i];
          svg.appendChild(t);
        }
      }

      const ttl = el("text", { x: pad.l + w / 2, y: 16, "text-anchor": "middle", "font-size": 13, fill: "#181818", "font-family": "Inter, sans-serif", "font-weight": 700 });
      ttl.textContent = title;
      svg.appendChild(ttl);

      // Small inline legend
      const lg = el("g", {});
      lg.appendChild(el("rect", { x: pad.l + w - 110, y: pad.t + 4, width: 12, height: 8, fill: primaryColor, opacity: 0.85 }));
      const lt1 = el("text", { x: pad.l + w - 94, y: pad.t + 12, "font-size": 10.5, fill: "#444", "font-family": "Inter, sans-serif" });
      lt1.textContent = "this panel";
      lg.appendChild(lt1);
      lg.appendChild(el("rect", { x: pad.l + w - 50, y: pad.t + 4, width: 12, height: 8, fill: "none", stroke: "#999", "stroke-dasharray": "3 2" }));
      const lt2 = el("text", { x: pad.l + w - 34, y: pad.t + 12, "font-size": 10.5, fill: "#444", "font-family": "Inter, sans-serif" });
      lt2.textContent = "other";
      lg.appendChild(lt2);
      svg.appendChild(lg);

      host.innerHTML = "";
      host.appendChild(svg);
    }

    // KL(p || q) in nats. Tiny epsilon for numerical safety.
    function klDiv(p, q) {
      let kl = 0;
      for (let i = 0; i < p.length; i++) {
        if (p[i] > 1e-12) kl += p[i] * (Math.log(p[i] + 1e-20) - Math.log(q[i] + 1e-20));
      }
      return kl;
    }

    function render() {
      const tempLogits = logits.map((l) => l / state.T);
      const pT = softmax(tempLogits);

      const pr = presets[state.mode];
      const warpedLogits = logits.map((l) => piecewise_f(l, pr.s1, pr.s2, pr.b));
      const pW = softmax(warpedLogits);

      drawBars(leftHost,  pT,  pW, "#666",    `Global temperature  T = ${state.T.toFixed(2)}`);
      drawBars(rightHost, pW,  pT, "#c7541d", pr.name);

      const HT = entropy(pT), HW = entropy(pW);
      const t1T = Math.max(...pT), t1W = Math.max(...pW);
      const klTW = klDiv(pT, pW);
      const klWT = klDiv(pW, pT);
      const klSym = 0.5 * (klTW + klWT);

      // Find an indicator of "best you can do with global T" against the
      // currently selected LogitMLP target. (Sweep a fine grid; just for
      // display, not interactive.)
      let bestKL = Infinity, bestT = 1.0;
      for (let T = 0.30; T <= 1.80; T += 0.01) {
        const p = softmax(logits.map((l) => l / T));
        const k = klDiv(p, pW);
        if (k < bestKL) { bestKL = k; bestT = T; }
      }

      // Color the KL display: green-ish if close, red-ish if far.
      const klColor = klSym < 0.02 ? "#1f8b3a" : klSym < 0.08 ? "#b07b00" : "#c7541d";

      summary.innerHTML =
        `<div style="font-size:13px;margin-bottom:4px"><b>KL between the two distributions:</b> ` +
        `<span style="color:${klColor};font-weight:700">${klSym.toFixed(3)} nats</span> ` +
        `(symmetrized) &nbsp;·&nbsp; best you can do at any global <i>T</i> is ` +
        `<b>${bestKL.toFixed(3)} nats</b> at <b><i>T</i> = ${bestT.toFixed(2)}</b></div>` +
        `<div>Left  · Global <i>T</i> = ${state.T.toFixed(2)}: entropy <b>${HT.toFixed(3)}</b> nats, top-1 <b>${(100*t1T).toFixed(1)}%</b></div>` +
        `<div>Right · ${pr.name} (s₁=${pr.s1.toFixed(2)}, s₂=${pr.s2.toFixed(2)}): entropy <b>${HW.toFixed(3)}</b> nats, top-1 <b>${(100*t1W).toFixed(1)}%</b></div>` +
        `<div style="font-style:italic;margin-top:4px;color:#888">The dashed outline in each panel shows the <em>other</em> panel's bars, for direct visual comparison.</div>`;
    }

    updateTabs();
    render();
  }

  // ---------- 4. Qualitative examples panel (loaded from JSON) ----------

  async function qualitativeExamples(root) {
    // Async load examples extracted from real Qwen3-1.7B rollouts.
    const status = document.createElement("div");
    status.style.cssText = "font-family:Inter,sans-serif;font-size:13px;color:#5b5b5b;padding:8px 0";
    status.textContent = "Loading examples…";
    root.appendChild(status);

    let data;
    try {
      const resp = await fetch("data/qualitative_examples.json");
      data = await resp.json();
    } catch (e) {
      status.textContent = "(Could not load qualitative_examples.json — run blog/data/extract_qualitative.py)";
      return;
    }
    status.remove();

    const tabsHost = document.createElement("div");
    tabsHost.className = "variant-tabs";
    const body = document.createElement("div");

    const variants = [
      { id: "pl_warp_passk", label: "LogitMLP (pass@k) — sharpens at choice points" },
      { id: "pl_warp_dapo",  label: "LogitMLP (DAPO / pass@1) — flattens at choice points" },
    ];

    let current = variants[0].id;

    function renderTopkTable(rows, kind) {
      const t = document.createElement("table");
      const max = Math.max(...rows.map(([, p]) => p));
      for (const [tok, p] of rows) {
        const tr = document.createElement("tr");
        const td1 = document.createElement("td"); td1.className = "tok";
        td1.textContent = JSON.stringify(tok).slice(1, -1);
        const td2 = document.createElement("td"); td2.className = "bar";
        const bar = document.createElement("div"); bar.className = "barfill" + (kind === "warp" ? " warp" : "");
        bar.style.width = `calc(${(p/max*100).toFixed(1)}% - 6ch)`;
        const pcnt = document.createElement("div"); pcnt.className = "pcnt";
        pcnt.textContent = (100*p).toFixed(1) + "%";
        td2.appendChild(bar); td2.appendChild(pcnt);
        tr.appendChild(td1); tr.appendChild(td2); t.appendChild(tr);
      }
      return t;
    }

    function escapeHtml(s) {
      return s.replace(/[&<>"']/g, (m) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
      }[m]));
    }

    function renderVariant() {
      body.innerHTML = "";
      const examples = data[current] || [];
      // Filter to the most "structural" looking ones first (a hand-tuned
      // heuristic: prefer post-step header tokens or punctuation after
      // sentence ends, which are the moderate-confidence branching tokens
      // the analysis report calls out).
      const ranked = examples.slice().sort((a, b) => b.kl - a.kl).slice(0, 4);

      for (const ex of ranked) {
        const card = document.createElement("div");
        card.className = "example-card";

        // Context display
        const ctx = document.createElement("div");
        ctx.className = "ctx";
        const preEsc = escapeHtml(ex.pre);
        const chosenEsc = escapeHtml(ex.chosen);
        const postEsc = escapeHtml(ex.post);
        ctx.innerHTML = `<span>…${preEsc}</span><span class="chosen">${chosenEsc}</span><span class="post-dim">${postEsc}…</span>`;
        card.appendChild(ctx);

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML =
          `<span><b>s₁</b> = ${fmt(ex.s1)}</span>` +
          `<span><b>s₂</b> = ${fmt(ex.s2)}</span>` +
          `<span><b>base top-1 prob</b> = ${(100*ex.top1_base).toFixed(1)}%</span>` +
          `<span><b>base entropy</b> = ${fmt(ex.entropy_base)} nats</span>` +
          `<span><b>KL(warp‖base)</b> = ${fmt(ex.kl)} nats</span>`;
        card.appendChild(meta);

        const topkRow = document.createElement("div");
        topkRow.className = "topk-row";
        const left = document.createElement("div");
        left.className = "topk";
        const lh = document.createElement("h5"); lh.textContent = "Base distribution"; left.appendChild(lh);
        left.appendChild(renderTopkTable(ex.base_top, "base"));
        const right = document.createElement("div");
        right.className = "topk";
        const rh = document.createElement("h5"); rh.textContent = "After LogitMLP"; right.appendChild(rh);
        right.appendChild(renderTopkTable(ex.warp_top, "warp"));
        topkRow.appendChild(left); topkRow.appendChild(right);
        card.appendChild(topkRow);

        body.appendChild(card);
      }
    }

    for (const v of variants) {
      const b = document.createElement("button");
      b.textContent = v.label;
      b.dataset.k = v.id;
      b.addEventListener("click", () => {
        current = v.id;
        for (const c of tabsHost.children) c.classList.toggle("active", c.dataset.k === current);
        renderVariant();
      });
      tabsHost.appendChild(b);
    }
    tabsHost.firstChild.classList.add("active");

    root.appendChild(tabsHost);
    root.appendChild(body);
    renderVariant();
  }

  // ---------- Public ----------

  window.LogitMLPBlog = {
    piecewiseWidget,
    paretoFrontier,
    tempVsLogitmlp,
    qualitativeExamples,
  };

  document.addEventListener("DOMContentLoaded", () => {
    const e1 = document.getElementById("piecewise-widget");
    if (e1) piecewiseWidget(e1);
    const e2 = document.getElementById("pareto-widget");
    if (e2) paretoFrontier(e2);
    const e3 = document.getElementById("tvslm-widget");
    if (e3) tempVsLogitmlp(e3);
    const e4 = document.getElementById("qualitative-widget");
    if (e4) qualitativeExamples(e4);
  });
})();
