(function () {
    const id = "viz-root-uuid";
    const p = JSON.parse(document.getElementById("viz-data-" + id).textContent);
    const drivers = new Set(p.drivers || []);
    const box = document.getElementById("token-display-box");
    const stops = [[250, 250, 250], [255, 238, 88], [255, 160, 0], [220, 38, 38], [100, 0, 0]];
    const lerp = (a, b, t) => [0, 1, 2].map((i) => a[i] + t * (b[i] - a[i]));
    const heatColor = (h) => {
        const t = Math.max(0, Math.min(1, h));
        const x = t * (stops.length - 1);
        const i = Math.min(Math.floor(x), stops.length - 2);
        const rgb = lerp(stops[i], stops[i + 1], x - i);
        return `rgb(${rgb[0] | 0},${rgb[1] | 0},${rgb[2] | 0})`;
    };
    const base =
        "padding:2px 6px;border-radius:3px;border:2px solid transparent;" +
        "white-space:pre-wrap;font-weight:600;font-size:14px;";

    (p.tokens || []).forEach((t) => {
        const el = document.createElement("span");
        const h = t.heat ?? 0;
        el.style.cssText = base +
            (h >= 0.55 ? "color:#fff;" : "color:#111;") +
            (drivers.has(t.idx) ? "border-color:#b71c1c;" : "");
        el.textContent = t.text === " " ? "\u00b7" : t.text;
        el.style.backgroundColor = heatColor(h);
        el.title = `raw: ${Number(t.score).toExponential(3)}  ·  heat: ${h.toFixed(2)}`;
        box.appendChild(el);
    });

    if (p.completion) {
        document.getElementById("completion-text").textContent = p.completion;
        document.getElementById("completion-block").hidden = false;
    }

    const resize = () => {
        try {
            const f = window.frameElement;
            if (f) {
                f.style.height = (document.body.scrollHeight + 16) + "px";
                f.style.overflow = "hidden";
            }
        } catch (e) { /* ignore */ }
    };
    resize();
    requestAnimationFrame(resize);
})();
