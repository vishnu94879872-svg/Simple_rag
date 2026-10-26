"""Reusable presentation helpers for the BookRAG console."""

import base64
import html
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import streamlit as st

from app.ui.styles import BASE_CSS

_ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets" / "brand"


def escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


@lru_cache(maxsize=4)
def asset_data_uri(filename: str) -> str:
    path = _ASSETS_DIR / filename
    if not path.exists():
        return ""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def inject_theme() -> None:
    st.markdown(BASE_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <header class="br-pagehead">
          <div class="br-eyebrow">{escape(kicker)}</div>
          <h1 class="br-h1">{escape(title)}</h1>
          <p class="br-lede">{escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def pill(label: str, tone: str = "default", dot: bool = False) -> str:
    dot_html = f'<span class="br-dot br-dot--{"on" if tone == "success" else "warn" if tone == "warning" else "off"}"></span>' if dot else ""
    tone_class = "" if tone == "default" else f" br-pill--{tone}"
    return f'<span class="br-pill{tone_class}">{dot_html}{escape(label)}</span>'


def hero(
    title: str,
    description: str,
    flow: Sequence[str],
    status_pills: Iterable[str],
    eyebrow: str = "RAG Reliability Lab / Live telemetry",
) -> None:
    image = asset_data_uri("hero-lattice.jpg")
    background = f"--br-hero-image: url('{image}');" if image else ""
    flow_html = "".join(
        f'<span class="br-flow-step{" br-flow-step--accent" if index == 1 else ""}">{escape(step)}</span>'
        for index, step in enumerate(flow)
    )
    st.markdown(
        f"""
        <section class="br-hero" style="{background}">
          <div class="br-eyebrow">{escape(eyebrow)}</div>
          <h1>{escape(title)}</h1>
          <p>{escape(description)}</p>
          <div class="br-flow">{flow_html}</div>
          <div class="br-hero-status">{''.join(status_pills)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def banner(title: str, body: str, tone: str = "warning") -> None:
    st.markdown(
        f'<div class="br-banner br-banner--{escape(tone)}"><div><strong>{escape(title)}</strong> {escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, body: str, mark: str = "◎") -> None:
    st.markdown(
        f"""
        <div class="br-empty">
          <div class="br-empty-mark">{escape(mark)}</div>
          <div class="br-empty-title">{escape(title)}</div>
          <p class="br-empty-body">{escape(body)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def verdict(title: str, body: str, tone: str = "info") -> None:
    st.markdown(
        f'<div class="br-verdict br-verdict--{escape(tone)}">'
        f'<div class="br-verdict-title">{escape(title)}</div>'
        f'<div class="br-verdict-body">{escape(body)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def sidebar_brand(title: str = "BookRAG", subtitle: str = "Reliability Lab", mark: str = "BR") -> None:
    st.markdown(
        f"""
        <div class="br-brand">
          <div class="br-brand-mark">{escape(mark)}</div>
          <div>
            <div class="br-brand-name">{escape(title)}</div>
            <div class="br-brand-sub">{escape(subtitle)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_nav_label(label: str) -> None:
    st.markdown(f'<div class="br-nav-label">{escape(label)}</div>', unsafe_allow_html=True)


def _backend_descriptor(status: Dict[str, Any]) -> Tuple[str, str, str]:
    backend = status.get("backend", "local")
    degraded = bool(status.get("degraded"))
    init_error = status.get("last_error") if backend == "local" else None
    if backend == "supabase" and not degraded:
        return "Supabase", "success", "on"
    if backend == "supabase" and degraded:
        return "Supabase · degraded", "warning", "warn"
    if init_error:
        return "Local · Supabase failed", "warning", "warn"
    return "Local files", "default", "off"


def sidebar_status(status: Dict[str, Any], vector_count: int, bm25_count: int, model: str) -> None:
    label, tone, dot = _backend_descriptor(status)
    dot_class = f"br-dot--{dot}"
    st.markdown(
        f"""
        <div class="br-status">
          <div class="br-status-row">
            <span class="br-status-key">Storage</span>
            <span class="br-status-val"><span class="br-dot {dot_class}"></span>{escape(label)}</span>
          </div>
          <div class="br-status-row">
            <span class="br-status-key">Vectors</span>
            <span class="br-status-val">{int(vector_count):,}</span>
          </div>
          <div class="br-status-row">
            <span class="br-status-key">BM25 docs</span>
            <span class="br-status-val">{int(bm25_count):,}</span>
          </div>
          <div class="br-status-row">
            <span class="br-status-key">Model</span>
            <span class="br-status-val">{escape(model)}</span>
          </div>
          <div class="br-status-foot">
            <span>Backend</span>
            <span>{escape(str(status.get("backend", "local")))} · v3.0</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def storage_banner(status: Dict[str, Any]) -> None:
    """Only surface storage problems; a healthy local backend stays quiet."""
    if status.get("degraded"):
        error = status.get("last_error") or "Unknown connection error."
        banner(
            "Supabase is unreachable — running on local fallback.",
            f"Recent operations are being written to {status.get('local_dir', 'local storage')}. {error}",
            tone="warning",
        )
    elif status.get("backend") == "local" and status.get("last_error"):
        banner(
            "Supabase storage could not initialize.",
            f"Falling back to local files. {status.get('last_error')}",
            tone="warning",
        )
