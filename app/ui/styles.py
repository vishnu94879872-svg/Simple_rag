"""Design tokens and global styling for the BookRAG console."""

BASE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  --br-bg: #08090c;
  --br-bg-soft: #0b0d12;
  --br-surface: #11141c;
  --br-surface-2: #171b26;
  --br-surface-3: #1d2230;
  --br-border: #232838;
  --br-border-strong: #313a50;
  --br-text: #edeff6;
  --br-text-muted: #9aa3b8;
  --br-text-dim: #6b7489;
  --br-accent: #7c9bff;
  --br-accent-strong: #a8bbff;
  --br-accent-soft: rgba(124, 155, 255, 0.12);
  --br-success: #46d5a0;
  --br-success-soft: rgba(70, 213, 160, 0.12);
  --br-warning: #f4be5c;
  --br-warning-soft: rgba(244, 190, 92, 0.12);
  --br-danger: #ff7676;
  --br-danger-soft: rgba(255, 118, 118, 0.12);
  --br-info: #66b6ff;
  --br-info-soft: rgba(102, 182, 255, 0.12);
  --br-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace;
  --br-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  --br-radius: 14px;
  --br-radius-sm: 9px;
  --br-ease: cubic-bezier(0.4, 0, 0.2, 1);
  --br-shadow: 0 18px 40px -28px rgba(0, 0, 0, 0.9);
}

html, body, [class*="css"], .stApp, .stApp * {
  font-family: var(--br-sans);
}

/* Streamlit renders Material icon names as ligatures; keep their icon font. */
[data-testid="stIconMaterial"],
.material-symbols-rounded {
  font-family: "Material Symbols Rounded" !important;
  font-weight: normal !important;
  font-style: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  white-space: nowrap !important;
  direction: ltr !important;
  font-feature-settings: "liga" !important;
  -webkit-font-feature-settings: "liga" !important;
}

.stApp {
  background:
    radial-gradient(1000px 560px at 16% -12%, rgba(124, 155, 255, 0.10), transparent 62%),
    radial-gradient(760px 420px at 104% -4%, rgba(70, 213, 160, 0.05), transparent 58%),
    var(--br-bg);
  color: var(--br-text);
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {
  display: none !important;
}

[data-testid="stHeader"] {
  background: transparent;
  height: 0;
}

[data-testid="stAppViewBlockContainer"], .block-container {
  max-width: 1320px;
  padding: 2.1rem 3rem 4.5rem;
}

/* Bottom-docked chat input uses a wider default gutter; keep it on the same
   content edges as the page container. */
[data-testid="stBottomBlockContainer"] {
  max-width: 1320px;
  margin: 0 auto;
  padding-left: 3rem;
  padding-right: 3rem;
}

/* ---------------------------------------------------------------- sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0a0c11 0%, #08090c 100%);
  border-right: 1px solid var(--br-border);
  min-width: 288px;
  max-width: 288px;
}

section[data-testid="stSidebar"] > div {
  padding-top: 0.8rem;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding: 0 1rem 1rem;
}

section[data-testid="stSidebar"] hr {
  border-color: var(--br-border);
  margin: 0.9rem 0;
}

/* --------------------------------------------------------------- branding */
.br-brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 2px 4px;
}

.br-brand-mark {
  width: 38px;
  height: 38px;
  border-radius: 11px;
  display: grid;
  place-items: center;
  font-family: var(--br-mono);
  font-size: 13px;
  font-weight: 700;
  letter-spacing: -0.04em;
  color: var(--br-accent-strong);
  background:
    linear-gradient(145deg, rgba(124, 155, 255, 0.22), rgba(124, 155, 255, 0.04)),
    var(--br-surface-2);
  border: 1px solid rgba(124, 155, 255, 0.35);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 8px 24px -14px rgba(124, 155, 255, 0.8);
}

.br-brand-name {
  font-size: 15.5px;
  font-weight: 650;
  letter-spacing: -0.02em;
  color: var(--br-text);
  line-height: 1.2;
}

.br-brand-sub {
  font-family: var(--br-mono);
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--br-text-dim);
  margin-top: 3px;
}

.br-nav-label {
  font-family: var(--br-mono);
  font-size: 9.5px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--br-text-dim);
  margin: 6px 2px 8px;
}

/* ------------------------------------------------------------ sidebar nav */
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has([data-testid="stRadio"]),
section[data-testid="stSidebar"] [data-testid="stRadio"],
section[data-testid="stSidebar"] [data-testid="stRadio"] > div,
section[data-testid="stSidebar"] [data-testid="stRadioGroup"],
section[data-testid="stSidebar"] [data-testid="stRadioGroup"] > div {
  width: 100% !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
  gap: 2px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
  width: 100% !important;
  padding: 9px 12px !important;
  border-radius: var(--br-radius-sm);
  border: 1px solid transparent;
  margin: 0 !important;
  cursor: pointer;
  box-shadow: inset 2px 0 0 transparent;
  transition: background 0.18s var(--br-ease), border-color 0.18s var(--br-ease),
    box-shadow 0.18s var(--br-ease), color 0.18s var(--br-ease);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: var(--br-surface);
  border-color: var(--br-border);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: var(--br-accent-soft);
  border-color: rgba(124, 155, 255, 0.28);
  box-shadow: inset 2px 0 0 var(--br-accent);
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
  font-size: 13.5px !important;
  font-weight: 500;
  color: var(--br-text-muted);
  letter-spacing: -0.01em;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
  color: var(--br-text);
  font-weight: 600;
}

/* Streamlit nests the control in a wrapper div; hide only the radio circle. */
section[data-testid="stSidebar"] div[role="radiogroup"] label > div > div:first-child {
  display: none !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label > div {
  width: 100%;
  gap: 0 !important;
}

/* -------------------------------------------------------------- status box */
.br-status {
  border: 1px solid var(--br-border);
  background: var(--br-surface);
  border-radius: 12px;
  padding: 12px 13px;
  margin-top: 4px;
}

.br-status-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  padding: 4px 0;
}

.br-status-key {
  font-family: var(--br-mono);
  font-size: 9.5px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--br-text-dim);
}

.br-status-val {
  font-family: var(--br-mono);
  font-size: 10.5px;
  color: var(--br-text-muted);
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  max-width: 150px;
}

.br-status-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-top: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--br-border);
  font-family: var(--br-mono);
  font-size: 9.5px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--br-text-dim);
}

.br-status-foot span:last-child {
  font-size: 10.5px;
  letter-spacing: 0;
  text-transform: none;
  color: var(--br-text-muted);
}

.br-dot {
  width: 6px;
  height: 6px;
  border-radius: 999px;
  display: inline-block;
  margin-right: 6px;
  vertical-align: 1px;
}

.br-dot--on { background: var(--br-success); box-shadow: 0 0 0 3px rgba(70, 213, 160, 0.12); }
.br-dot--warn { background: var(--br-warning); box-shadow: 0 0 0 3px rgba(244, 190, 92, 0.12); }
.br-dot--off { background: var(--br-text-dim); box-shadow: 0 0 0 3px rgba(107, 116, 137, 0.12); }

/* ----------------------------------------------------------------- header */
.br-pagehead {
  padding-bottom: 1.15rem;
  margin-bottom: 1.5rem;
  border-bottom: 1px solid var(--br-border);
  animation: br-rise 0.45s var(--br-ease) both;
}

.br-eyebrow {
  font-family: var(--br-mono);
  font-size: 10px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--br-accent);
  display: flex;
  align-items: center;
  gap: 9px;
  margin-bottom: 10px;
}

.br-eyebrow::before {
  content: "";
  width: 22px;
  height: 1px;
  background: var(--br-accent);
  opacity: 0.65;
}

.br-h1 {
  font-size: 1.95rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.035em;
  line-height: 1.16;
  color: var(--br-text) !important;
  margin: 0 0 0.45rem !important;
  padding: 0 !important;
}

.br-lede {
  font-size: 13.5px;
  line-height: 1.6;
  color: var(--br-text-muted);
  max-width: 86ch;
  margin: 0;
}

/* ------------------------------------------------------------------- hero */
.br-hero {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--br-border);
  border-radius: 20px;
  padding: 34px 36px 28px;
  margin-bottom: 1.6rem;
  background-image:
    linear-gradient(90deg, rgba(8, 9, 12, 0.97) 0%, rgba(8, 9, 12, 0.86) 42%, rgba(8, 9, 12, 0.28) 100%),
    var(--br-hero-image);
  background-size: cover;
  background-position: center right;
  box-shadow: var(--br-shadow);
  animation: br-rise 0.5s var(--br-ease) both;
}

.br-hero::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  pointer-events: none;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
}

.br-hero .br-eyebrow { margin-bottom: 14px; }

.br-hero h1 {
  font-size: 2.55rem;
  font-weight: 680;
  letter-spacing: -0.045em;
  line-height: 1.08;
  margin: 0 0 12px;
  max-width: 20ch;
  color: #f4f6fc;
}

.br-hero p {
  font-size: 14px;
  line-height: 1.65;
  color: var(--br-text-muted);
  max-width: 58ch;
  margin: 0 0 20px;
}

.br-flow {
  display: flex;
  flex-wrap: wrap;
  gap: 7px;
  align-items: center;
  margin-bottom: 18px;
}

.br-flow-step {
  font-family: var(--br-mono);
  font-size: 9.5px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--br-text-muted);
  background: rgba(17, 20, 28, 0.72);
  border: 1px solid var(--br-border);
  border-radius: 999px;
  padding: 4px 9px;
  backdrop-filter: blur(4px);
}

.br-flow-step--accent {
  color: var(--br-accent-strong);
  border-color: rgba(124, 155, 255, 0.35);
  background: rgba(124, 155, 255, 0.12);
}

.br-hero-status {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

/* ------------------------------------------------------------------- pills */
.br-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: var(--br-mono);
  font-size: 9.5px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  padding: 4px 9px;
  border-radius: 999px;
  border: 1px solid var(--br-border);
  background: var(--br-surface);
  color: var(--br-text-muted);
  white-space: nowrap;
}

.br-pill--accent { color: var(--br-accent-strong); background: var(--br-accent-soft); border-color: rgba(124, 155, 255, 0.3); }
.br-pill--success { color: var(--br-success); background: var(--br-success-soft); border-color: rgba(70, 213, 160, 0.3); }
.br-pill--warning { color: var(--br-warning); background: var(--br-warning-soft); border-color: rgba(244, 190, 92, 0.3); }
.br-pill--danger { color: var(--br-danger); background: var(--br-danger-soft); border-color: rgba(255, 118, 118, 0.3); }
.br-pill--info { color: var(--br-info); background: var(--br-info-soft); border-color: rgba(102, 182, 255, 0.3); }

/* ------------------------------------------------------------------ banner */
.br-banner {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  border: 1px solid var(--br-border);
  border-left: 3px solid var(--br-text-dim);
  border-radius: 10px;
  background: var(--br-surface);
  padding: 10px 14px;
  margin: 0 0 1.2rem;
  font-size: 12.5px;
  line-height: 1.55;
  color: var(--br-text-muted);
}

.br-banner strong { color: var(--br-text); font-weight: 600; }
.br-banner--warning { border-left-color: var(--br-warning); background: var(--br-warning-soft); }
.br-banner--danger { border-left-color: var(--br-danger); background: var(--br-danger-soft); }
.br-banner--info { border-left-color: var(--br-info); background: var(--br-info-soft); }

/* ------------------------------------------------------------- empty state */
.br-empty {
  border: 1px dashed var(--br-border-strong);
  background: rgba(17, 20, 28, 0.45);
  border-radius: var(--br-radius);
  padding: 26px 22px;
  text-align: center;
}

.br-empty-mark {
  font-family: var(--br-mono);
  font-size: 16px;
  color: var(--br-accent);
  margin-bottom: 6px;
}

.br-empty-title {
  font-size: 13.5px;
  font-weight: 600;
  color: var(--br-text);
  margin-bottom: 4px;
}

.br-empty-body {
  font-size: 12.5px;
  line-height: 1.6;
  color: var(--br-text-dim);
  max-width: 52ch;
  margin: 0 auto;
}

/* ------------------------------------------------------------------ cards */
.chunk-card {
  background: var(--br-surface) !important;
  border: 1px solid var(--br-border) !important;
  border-radius: 12px !important;
  padding: 16px 18px !important;
  word-break: break-word;
  overflow-wrap: break-word;
  transition: border-color 0.18s var(--br-ease);
}

.chunk-card:hover { border-color: var(--br-border-strong) !important; }

.chunk-head {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  flex-wrap: wrap;
}

.chunk-id {
  font-family: var(--br-mono);
  font-size: 11.5px;
  font-weight: 600;
  line-height: 1.5;
  color: var(--br-accent-strong);
  overflow-wrap: anywhere;
}

.chunk-tags {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-wrap: wrap;
}

.chunk-meta {
  margin-top: 8px;
  font-size: 11.5px;
  line-height: 1.6;
  color: var(--br-text-muted);
}

.chunk-match { margin-top: 10px; }

.source-tag {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: var(--br-accent-soft) !important;
  color: var(--br-accent-strong) !important;
  border: 1px solid rgba(124, 155, 255, 0.3) !important;
  padding: 3px 10px !important;
  border-radius: 999px !important;
  font-family: var(--br-mono);
  font-size: 10.5px !important;
  font-weight: 500 !important;
  margin-right: 6px !important;
  margin-top: 6px !important;
}

.br-verdict {
  border: 1px solid var(--br-border);
  border-left: 3px solid var(--br-text-dim);
  border-radius: 12px;
  background: var(--br-surface);
  padding: 15px 18px;
  margin-bottom: 1.2rem;
}

.br-verdict-title {
  font-size: 14px;
  font-weight: 650;
  letter-spacing: 0.02em;
  color: var(--br-text);
}

.br-verdict-body {
  font-size: 12.5px;
  color: var(--br-text-muted);
  margin-top: 4px;
}

.br-verdict--success { border-left-color: var(--br-success); background: var(--br-success-soft); }
.br-verdict--success .br-verdict-title { color: var(--br-success); }
.br-verdict--warning { border-left-color: var(--br-warning); background: var(--br-warning-soft); }
.br-verdict--warning .br-verdict-title { color: var(--br-warning); }
.br-verdict--danger { border-left-color: var(--br-danger); background: var(--br-danger-soft); }
.br-verdict--danger .br-verdict-title { color: var(--br-danger); }

.br-stage {
  border: 1px solid var(--br-border);
  border-left: 3px solid var(--br-border-strong);
  border-radius: 10px;
  background: var(--br-surface);
  padding: 11px 15px;
  margin-bottom: 8px;
}

.br-doc-id {
  font-family: var(--br-mono);
  font-size: 11px;
  line-height: 1.45;
  color: var(--br-text);
  overflow-wrap: anywhere;
}

.br-stage-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 6px;
  flex-wrap: wrap;
}

.br-stage-name {
  font-size: 13px;
  font-weight: 600;
  color: var(--br-text);
}

.br-stage-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: var(--br-mono);
  font-size: 10px;
  color: var(--br-text-dim);
}

.br-stage-time {
  background: var(--br-surface-3);
  color: var(--br-accent-strong);
  border-radius: 5px;
  padding: 2px 7px;
  font-weight: 500;
}

.br-track {
  width: 100%;
  height: 5px;
  background: var(--br-surface-3);
  border-radius: 999px;
  overflow: hidden;
}

.br-track > span {
  display: block;
  height: 100%;
  border-radius: 999px;
}

/* --------------------------------------------------- chunk card internals */
.chunk-issue {
  margin: 12px 0 0;
  padding: 10px 12px;
  border-radius: 9px;
}

.chunk-issue--plain { background: var(--br-surface-2); }
.chunk-issue .br-verdict-title { font-size: 12px; }
.chunk-issue .br-verdict-body { font-size: 11.5px; margin-top: 3px; }

.chunk-evidence {
  margin-top: 4px;
  font-family: var(--br-mono);
  font-size: 10.5px;
  line-height: 1.6;
  color: var(--br-info);
  overflow-wrap: anywhere;
}

.chunk-recommendation {
  margin-top: 5px;
  font-size: 11px;
  line-height: 1.55;
  color: var(--br-text-dim);
}

.chunk-retrieval {
  margin: 12px 0 0;
  padding: 10px 12px;
  border-radius: 9px;
  background: var(--br-surface-2);
}

.chunk-retrieval-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

.chunk-retrieval-status {
  font-size: 11.5px;
  color: var(--br-text-muted);
}

.chunk-queries {
  margin-top: 4px;
  font-size: 10.5px;
  line-height: 1.6;
  color: var(--br-text-dim);
}

.chunk-causality {
  margin-top: 4px;
  font-size: 10.5px;
  color: var(--br-text-dim);
}

/* --------------------------------------------------------------- metrics */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.018), transparent 45%), var(--br-surface);
  border: 1px solid var(--br-border);
  border-radius: var(--br-radius);
  padding: 15px 17px 13px;
  min-height: 113px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
  transition: border-color 0.2s var(--br-ease), transform 0.2s var(--br-ease);
}

[data-testid="stMetric"]:hover {
  border-color: var(--br-border-strong);
  transform: translateY(-1px);
}

[data-testid="stMetricLabel"] p {
  font-family: var(--br-mono) !important;
  font-size: 9.5px !important;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  color: var(--br-text-dim) !important;
  line-height: 1.35;
}

[data-testid="stMetricValue"] {
  font-family: var(--br-sans) !important;
  font-size: 1.62rem !important;
  font-weight: 650 !important;
  letter-spacing: -0.035em;
  color: var(--br-text) !important;
  overflow-wrap: anywhere;
  line-height: 1.25;
}

[data-testid="stMetricDelta"] {
  font-family: var(--br-mono) !important;
  font-size: 10px !important;
  color: var(--br-text-dim);
}

/* --------------------------------------------------------------- buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button,
button[kind="secondary"],
button[kind="primary"] {
  border-radius: var(--br-radius-sm);
  border: 1px solid var(--br-border-strong);
  background: var(--br-surface-2);
  color: var(--br-text);
  font-size: 12.5px;
  font-weight: 550;
  padding: 0.48rem 0.95rem;
  transition: all 0.18s var(--br-ease);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.035);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
button[kind="secondary"]:hover {
  background: var(--br-surface-3);
  border-color: rgba(124, 155, 255, 0.5);
  color: var(--br-text);
}

button[kind="primary"],
[data-testid="stBaseButton-primary"] {
  background: var(--br-accent);
  border-color: var(--br-accent);
  color: #0a0c12;
  font-weight: 650;
}

button[kind="primary"]:hover,
[data-testid="stBaseButton-primary"]:hover {
  background: var(--br-accent-strong);
  border-color: var(--br-accent-strong);
  color: #080a0f;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
button[kind="primary"]:focus-visible {
  outline: 2px solid var(--br-accent);
  outline-offset: 2px;
}

[data-testid="stPopover"] { width: 100%; }

[data-testid="stPopoverButton"] { height: 40px; }

[data-testid="stPopoverButton"] > div {
  flex: 1;
  justify-content: space-between;
}

[data-testid="stPopoverButton"] > div > div:first-child {
  justify-content: flex-start;
}

.stButton > button:disabled,
.stDownloadButton > button:disabled,
[data-testid="stBaseButton-secondary"]:disabled,
[data-testid="stBaseButton-primary"]:disabled {
  background: var(--br-surface) !important;
  border-color: var(--br-border) !important;
  color: var(--br-text-dim) !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
  box-shadow: none !important;
}

/* ----------------------------------------------------------------- inputs */
[data-baseweb="input"],
[data-baseweb="select"] > div,
[data-baseweb="textarea"],
[data-baseweb="base-input"] {
  background: var(--br-surface) !important;
  border-color: var(--br-border) !important;
  border-radius: var(--br-radius-sm) !important;
  transition: border-color 0.18s var(--br-ease), box-shadow 0.18s var(--br-ease);
}

[data-baseweb="input"]:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"]:focus-within {
  border-color: var(--br-accent) !important;
  box-shadow: 0 0 0 3px var(--br-accent-soft) !important;
}

input, textarea {
  color: var(--br-text) !important;
  font-size: 13px !important;
}

[data-testid="stWidgetLabel"] p {
  font-size: 11.5px !important;
  font-weight: 550;
  color: var(--br-text-muted) !important;
}

[data-testid="stCaptionContainer"] p {
  font-size: 11.5px !important;
  color: var(--br-text-dim) !important;
  line-height: 1.55;
}

/* ------------------------------------------------------------- tabs/cards */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: 4px;
  background: var(--br-surface);
  border: 1px solid var(--br-border);
  border-radius: 11px;
  padding: 4px;
  width: fit-content;
  max-width: 100%;
  flex-wrap: wrap;
}

[data-testid="stTabs"] button[role="tab"] {
  border-radius: 7px;
  padding: 6px 13px;
  height: auto;
  min-height: 32px;
  color: var(--br-text-muted);
  font-size: 12px;
  font-weight: 550;
  transition: background 0.18s var(--br-ease), color 0.18s var(--br-ease);
}

[data-testid="stTabs"] button[role="tab"]:hover {
  background: var(--br-surface-2);
  color: var(--br-text);
}

[data-testid="stTabs"] button[aria-selected="true"] {
  background: var(--br-surface-3);
  color: var(--br-text);
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
  display: none;
}

[data-testid="stExpander"] details {
  background: var(--br-surface);
  border: 1px solid var(--br-border);
  border-radius: 12px;
  overflow: hidden;
}

[data-testid="stExpander"] summary {
  padding: 11px 16px;
  font-size: 12.5px;
  font-weight: 550;
  color: var(--br-text-muted);
  transition: background 0.18s var(--br-ease), color 0.18s var(--br-ease);
}

[data-testid="stExpander"] summary:hover {
  background: var(--br-surface-2);
  color: var(--br-text);
}

[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
  padding: 4px 16px 16px;
}

[data-testid="stVerticalBlockBorderWrapper"] {
  border-color: var(--br-border) !important;
  border-radius: var(--br-radius) !important;
  background: rgba(17, 20, 28, 0.4);
}

/* --------------------------------------------------------------- dataframes */
[data-testid="stDataFrame"] {
  border: 1px solid var(--br-border);
  border-radius: 11px;
  overflow: hidden;
}

[data-testid="stDataFrame"] [role="columnheader"] {
  background: var(--br-surface-2) !important;
}

/* ------------------------------------------------------------------- chat */
[data-testid="stChatMessage"] {
  background: var(--br-surface);
  border: 1px solid var(--br-border);
  border-radius: var(--br-radius);
  padding: 14px 16px;
  margin-bottom: 4px;
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
  font-size: 14px;
  line-height: 1.68;
  color: var(--br-text);
}

[data-testid="stChatMessageAvatarUser"],
[data-testid="chatAvatarIcon-user"] {
  background: var(--br-accent-soft) !important;
  color: var(--br-accent) !important;
  border: 1px solid rgba(124, 155, 255, 0.35) !important;
}

[data-testid="stChatMessageAvatarAssistant"],
[data-testid="chatAvatarIcon-assistant"] {
  background: var(--br-surface-2) !important;
  color: var(--br-success) !important;
  border: 1px solid var(--br-border) !important;
}

[data-testid="stChatInput"] {
  background: var(--br-surface) !important;
  border: 1px solid var(--br-border-strong) !important;
  border-radius: 12px !important;
}

[data-testid="stChatInput"]:focus-within {
  border-color: var(--br-accent) !important;
  box-shadow: 0 0 0 3px var(--br-accent-soft) !important;
}

/* ------------------------------------------------------------------ alerts */
[data-testid="stAlert"] {
  border-radius: 10px;
  border: 1px solid var(--br-border);
  background: var(--br-surface);
  font-size: 12.5px;
}

/* -------------------------------------------------------------- code/blocks */
pre {
  background: #0a0c12 !important;
  border: 1px solid var(--br-border) !important;
  border-radius: 10px !important;
}

code {
  font-family: var(--br-mono) !important;
  font-size: 0.82em !important;
}

pre, code {
  white-space: pre-wrap !important;
  word-break: break-word !important;
}

/* ---------------------------------------------------------------- headings */
h1, h2, h3, h4 {
  color: var(--br-text) !important;
  letter-spacing: -0.025em;
}

h2 { font-size: 1.35rem !important; font-weight: 650 !important; margin-top: 0.6rem !important; }
h3 { font-size: 1.02rem !important; font-weight: 620 !important; margin-top: 1.2rem !important; }
h4 { font-size: 0.92rem !important; font-weight: 600 !important; }

a { color: var(--br-accent-strong) !important; }

hr {
  border-color: var(--br-border) !important;
  margin: 1.15rem 0 !important;
}

/* ---------------------------------------------------------------- scrollbar */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #262c3d; border-radius: 999px; border: 2px solid var(--br-bg); }
::-webkit-scrollbar-thumb:hover { background: #343c52; }

/* ------------------------------------------------------------------ motion */
@keyframes br-rise {
  from { opacity: 0; transform: translateY(7px); }
  to { opacity: 1; transform: translateY(0); }
}

@media (prefers-reduced-motion: reduce) {
  .br-pagehead, .br-hero { animation: none; }
  * { transition-duration: 0.01ms !important; }
}

@media (max-width: 900px) {
  [data-testid="stAppViewBlockContainer"], .block-container {
    padding: 1.4rem 1.1rem 3rem;
  }
  .br-hero { padding: 24px 22px; }
  .br-hero h1 { font-size: 1.95rem; }
  .br-h1 { font-size: 1.6rem !important; }
}
</style>
"""
