"""Jupyter Token Highlighter visualization."""

from __future__ import annotations

import html
import json
import uuid
from pathlib import Path
from typing import Any, Callable

from IPython.display import HTML, display
from vllm_hook_plugins import HookLLM, analyze_with_highlighter, generate_with_highlighter

_ANALYZER = {"artifact_wait_seconds": 1.0}
_ASSETS = Path(__file__).resolve().parent / "assets"


def _rank_heat(scores: list[float]) -> list[float]:
    n = len(scores)
    if n <= 1:
        return [1.0] * n
    heats = [0.0] * n
    for rank, i in enumerate(sorted(range(n), key=lambda j: (scores[j], j))):
        heats[i] = rank / (n - 1)
    return heats


def _iframe_height(payload: dict[str, Any]) -> int:
    n = len(payload.get("tokens", []))
    return 88 + max(1, (n + 10) // 11) * 36 + (56 if payload.get("completion") else 0)


def _heatmap_iframe_html(payload: dict[str, Any], *, html_tpl: str, css: str, js_tpl: str) -> str:
    uid = f"viz-{uuid.uuid4().hex[:8]}"
    body = (
        html_tpl.replace("viz-root-uuid", uid)
        .replace("DATA_PLACEHOLDER", json.dumps(payload))
        .replace("/*SCRIPT_PLACEHOLDER*/", js_tpl.replace("viz-root-uuid", uid))
    )
    doc = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{css}</style></head>"
        f"<body style='margin:0;padding:0'>{body}</body></html>"
    )
    h = _iframe_height(payload)
    esc = html.escape(doc, quote=True)
    return (
        f'<iframe srcdoc="{esc}" style="width:100%;border:0;display:block;height:{h}px;overflow:hidden" '
        'sandbox="allow-scripts allow-same-origin" '
        "onload=\"(function(f){try{var d=f.contentDocument;if(d)f.style.height=(d.body.scrollHeight+16)+'px';}catch(e){}})(this)\">"
        "</iframe>"
    )


class TokenHighlighterVisualizer:
    def __init__(self, model: HookLLM, highlighter_config: dict[str, Any]):
        self.model = model
        self.highlighter_config = dict(highlighter_config)
        self._html = (_ASSETS / "layout.html").read_text(encoding="utf-8")
        self._css = (_ASSETS / "style.css").read_text(encoding="utf-8")
        self._js = (_ASSETS / "renderer.js").read_text(encoding="utf-8")
        self._capture_run_id: str | None = None

    def _target_ids(self, phrase: str) -> None:
        self.highlighter_config["target_token_ids"] = self.model.tokenizer.encode(
            phrase, add_special_tokens=False
        )

    def mitigate(self, prompt: str, phrase: str, beta: float, *, max_tokens: int = 32) -> str:
        cfg = dict(self.highlighter_config)
        cfg["beta"] = float(beta)
        self._target_ids(phrase)
        out = generate_with_highlighter(
            self.model,
            prompt,
            mode="mitigate",
            highlighter_config=cfg,
            run_id=self._capture_run_id or self.model._last_run_id,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        self.model.llm_engine.reset_prefix_cache()
        return out[0].outputs[0].text.strip()

    def _highlight(
        self,
        prompt: str,
        target_phrase: str,
        *,
        max_tokens: int = 32,
        analyzer_spec: dict | None = None,
    ) -> dict[str, Any]:
        self._target_ids(target_phrase)
        cap = generate_with_highlighter(
            self.model,
            prompt,
            mode="capture",
            highlighter_config=self.highlighter_config,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        analysis = analyze_with_highlighter(
            self.model,
            analyzer_spec={**_ANALYZER, **(analyzer_spec or {})},
            highlighter_config=self.highlighter_config,
        )
        if not analysis or not analysis.get("results"):
            raise ValueError("No highlighter results.")

        seq = analysis["results"][0]
        scores = [float(s) for s in seq["token_scores"]]
        heats = _rank_heat(scores)
        tok = self.model.tokenizer
        payload = {
            "drivers": list(seq.get("analysis_drivers") or seq.get("drivers", [])),
            "tokens": [
                {
                    "idx": i,
                    "text": tok.decode([int(tid)], skip_special_tokens=False),
                    "score": scores[i],
                    "heat": heats[i],
                }
                for i, tid in enumerate(seq["token_ids"])
            ],
            "completion": cap[0].outputs[0].text.strip(),
        }
        self._capture_run_id = analysis.get("run_id") or self.model._last_run_id
        self.model.llm_engine.reset_prefix_cache()
        return payload

    def _display_heatmap(self, payload: dict[str, Any], *, into: Any = None) -> None:
        markup = _heatmap_iframe_html(
            payload, html_tpl=self._html, css=self._css, js_tpl=self._js
        )
        if into is not None:
            into.value = markup
        else:
            display(HTML(markup))

    def _make_beta_slider(
        self,
        read: Callable[[], tuple[str, str]],
        *,
        max_tokens: int,
    ):
        """Return a β slider VBox (created once per visualize() call)."""
        import ipywidgets as w

        slider = w.FloatSlider(
            value=float(self.highlighter_config.get("beta", 0.1)),
            min=0.0, max=1.0, step=0.05, description="β:",
            readout_format=".2f", continuous_update=False,
            layout=w.Layout(width="95%"),
        )
        result = w.HTML(value="")

        busy = {"active": False}

        def _on(change: dict) -> None:
            # Traitlets can deliver duplicate change notifications per drag.
            if change.get("type") != "change" or change.get("name") != "value":
                return
            if busy["active"]:
                return
            p, ph = read()
            if not p or not ph:
                result.value = "<p>Enter prompt and target phrase first.</p>"
                return
            busy["active"] = True
            try:
                text = self.mitigate(p, ph, change["new"], max_tokens=max_tokens)
                esc = html.escape(text)
                result.value = (
                    f"<p><b>Mitigated (β={change['new']:.2f}):</b></p>"
                    f"<pre style='white-space:pre-wrap;margin:0'>{esc}</pre>"
                )
            except Exception as e:
                result.value = f"<p>Mitigate error: {html.escape(str(e))}</p>"
            finally:
                busy["active"] = False

        slider.observe(_on, names="value", type="change")
        return w.VBox([
            w.HTML("<b>Soft-removal β</b> — drag and release to mitigate"),
            slider,
            result,
        ])

    def visualize(
        self,
        prompt: str = "",
        target_phrase: str = "",
        *,
        max_tokens: int = 32,
        analyzer_spec: dict | None = None,
        mitigate_slider: bool = True,
        interactive: bool = True,
    ) -> None:
        from IPython.display import clear_output, display

        clear_output(wait=True)

        kw = {"max_tokens": max_tokens, "analyzer_spec": analyzer_spec}
        static_read = lambda: (prompt, target_phrase)

        if not interactive:
            self._display_heatmap(self._highlight(prompt, target_phrase, **kw))
            if mitigate_slider:
                try:
                    display(self._make_beta_slider(static_read, max_tokens=max_tokens))
                except ImportError:
                    display(HTML(
                        "<p><b>β slider:</b> <code>pip install ipywidgets</code> then restart kernel.</p>"
                    ))
            return

        try:
            import ipywidgets as w
        except ImportError:
            if not prompt.strip() or not target_phrase.strip():
                raise ImportError(
                    "ipywidgets required for editable prompt/target. pip install ipywidgets"
                ) from None
            self._display_heatmap(self._highlight(prompt, target_phrase, **kw))
            return

        layout = w.Layout(width="98%")
        prompt_w = w.Textarea(value=prompt, layout=w.Layout(width="98%", height="72px"))
        target_w = w.Text(value=target_phrase, layout=layout)
        btn = w.Button(description="Highlight", button_style="primary", icon="search")
        heat_html = w.HTML(layout=w.Layout(width="98%", overflow="visible"))
        read = lambda: (prompt_w.value.strip(), target_w.value.strip())
        run_busy = {"active": False}

        def _run(_: Any = None) -> None:
            if run_busy["active"]:
                return
            p, t = read()
            if not p or not t:
                heat_html.value = (
                    "<p style='margin:0'><i>Enter a prompt and target phrase, "
                    "then click Highlight.</i></p>"
                )
                return
            run_busy["active"] = True
            btn.disabled = True
            try:
                payload = self._highlight(p, t, **kw)
            except Exception as e:
                heat_html.value = f"<p>Highlight error: {html.escape(str(e))}</p>"
                return
            finally:
                run_busy["active"] = False
                btn.disabled = False
            self._display_heatmap(payload, into=heat_html)

        children: list[Any] = [
            w.HTML("<b>Prompt</b>"), prompt_w,
            w.HTML("<b>Target phrase</b>"), target_w,
            btn, heat_html,
        ]
        if mitigate_slider:
            children.append(self._make_beta_slider(read, max_tokens=max_tokens))

        btn.on_click(_run)
        display(w.VBox(children))
        if prompt.strip() and target_phrase.strip():
            _run()
