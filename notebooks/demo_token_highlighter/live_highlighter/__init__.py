from live_highlighter.paths import RuntimePaths, resolve_runtime_paths

__all__ = ["TokenHighlighterVisualizer", "RuntimePaths", "resolve_runtime_paths"]


def __getattr__(name: str):
    if name == "TokenHighlighterVisualizer":
        from live_highlighter.core import TokenHighlighterVisualizer
        return TokenHighlighterVisualizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
