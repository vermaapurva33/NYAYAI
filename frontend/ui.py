"""
NyayAI Frontend v4.1 — Industry-grade multi-page Streamlit application.

Pages (via session_state.page):
  home        → Landing / product introduction
  login       → Sign in
  signup      → Create account
  dashboard   → Bento feature cards
  checker     → 3-panel error checking interface
"""

import os
import io
import base64
import requests
import pandas as pd
import streamlit as st
from PIL import Image
from pathlib import Path

SVG_TRUE  = Path(__file__).parent / "logo_true.svg"
LOGO_FILE = Path(__file__).parent / "finalised_logo.png"

def _get_logo_b64():
    if LOGO_FILE.exists():
        with open(LOGO_FILE, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# ── Configuration ─────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
TIMEOUT_S   = 300
LOGO_PATH   = Path(__file__).parent / "logo.png"

st.set_page_config(
    page_title="NyayAI — Legal Intelligence Platform",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject Google Fonts + favicon
def _inject_head():
    favicon_tag = ""
    if SVG_TRUE.exists():
        b64 = base64.b64encode(SVG_TRUE.read_bytes()).decode()
        favicon_tag = f'<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;base64,{b64}">'
    st.markdown(f"""
    {favicon_tag}
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
_inject_head()

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base font ── */
  html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
    background-color: #ffffff;
    color: #0a0a0a;
  }
  .stApp { background-color: #ffffff; }

  /* ── Typography ── */
  h1 { font-size: 2.6rem; font-weight: 700; letter-spacing: -0.03em; color: #0a0a0a; }
  h2 { font-size: 1.8rem; font-weight: 600; letter-spacing: -0.02em; color: #0a0a0a; }
  h3 { font-size: 1.2rem; font-weight: 600; color: #0a0a0a; }
  p  { font-size: 1rem; line-height: 1.7; color: #3a3a3a; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    padding-left: 3rem !important;
    padding-right: 3rem !important;
    max-width: 1400px;
  }

  /* ── Buttons — all selectors needed for Streamlit's shadow DOM ── */
  .stButton > button,
  .stButton > button:visited,
  button[data-testid="baseButton-secondary"],
  button[data-testid="baseButton-primary"] {
    background-color: #0a0a0a !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.55rem 1.4rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: background 0.18s ease !important;
    cursor: pointer;
    width: 100%;
  }
  /* Button inner p / span — the actual text node */
  .stButton > button p,
  .stButton > button span,
  button[data-testid="baseButton-secondary"] p,
  button[data-testid="baseButton-primary"] p {
    color: #ffffff !important;
    font-weight: 500 !important;
  }
  .stButton > button:hover,
  button[data-testid="baseButton-secondary"]:hover {
    background-color: #1a6fc4 !important;
    box-shadow: 0 2px 8px rgba(26, 111, 196, 0.25) !important;
  }
  .stButton > button:focus { outline: 2px solid #1a6fc4 !important; outline-offset: 2px !important; }
  /* Outline variant */
  .btn-outline .stButton > button,
  .btn-outline button[data-testid="baseButton-secondary"] {
    background-color: transparent !important;
    color: #0a0a0a !important;
    border: 1.5px solid #000000 !important;
  }
  .btn-outline .stButton > button p,
  .btn-outline button[data-testid="baseButton-secondary"] p {
    color: #0a0a0a !important;
  }
  .btn-outline .stButton > button:hover {
    background-color: #f0f6ff !important;
    border-color: #1a6fc4 !important;
  }
  .btn-outline .stButton > button:hover p {
    color: #1a6fc4 !important;
  }
  
  /* Disable Streamlit column margins for aligned buttons */
  .st-key-hero_signin > div { margin-top: 0 !important; }
  
  /* Disabled */
  .stButton > button:disabled {
    background-color: #f0f0f0 !important;
    color: #bbb !important;
    cursor: not-allowed !important;
  }
  .stButton > button:disabled p { color: #bbb !important; }


  /* Auth split-screen: force equal height columns */
  div[data-testid="stHorizontalBlock"]:has(.auth-left) {
    align-items: stretch !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.auth-left) > div[data-testid="column"],
  div[data-testid="stHorizontalBlock"]:has(.auth-right) > div[data-testid="column"] {
    display: flex;
    flex-direction: column;
  }
  .auth-left {
    background: #0d0d12;
    flex: 1;
    height: 100%;
    padding: 3.5rem 3rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    border-radius: 0 14px 14px 0;
    box-sizing: border-box;
  }
  .auth-left-tag {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    color: rgba(255,255,255,0.6);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 0.28rem 0.8rem;
    border-radius: 20px;
    margin-bottom: 2.5rem;
    border: 1px solid rgba(255,255,255,0.12);
  }
  .auth-left-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.2;
    margin-bottom: 1.2rem;
    letter-spacing: -0.01em;
  }
  .auth-left-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.45);
    line-height: 1.75;
    font-weight: 300;
  }
  .auth-left-quote {
    margin-top: 2.5rem;
    padding: 1rem 1.2rem;
    background: rgba(255,255,255,0.04);
    border-left: 2px solid rgba(255,255,255,0.15);
    border-radius: 0 6px 6px 0;
    font-family: 'Playfair Display', Georgia, serif;
    font-style: italic;
    font-size: 0.92rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.6;
  }
  .auth-left-stats {
    display: flex;
    gap: 2rem;
    margin-top: 2.5rem;
    padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.08);
  }
  .auth-left-stat-val  { font-size: 1.3rem; font-weight: 700; color: #ffffff; }
  .auth-left-stat-label { font-size: 0.72rem; color: rgba(255,255,255,0.35); margin-top: 0.1rem; }
  .auth-right {
    flex: 1;
    height: 100%;
    padding: 3.5rem 3rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-sizing: border-box;
    max-width: 560px;
  }
  .auth-right-logo { font-size: 0.9rem; font-weight: 700; color: #0a0a0a; letter-spacing: -0.01em; margin-bottom: 3rem; }
  .auth-right-title { font-size: 1.8rem; font-weight: 700; color: #0a0a0a; margin-bottom: 0.4rem; letter-spacing: -0.02em; }
  .auth-right-sub   { font-size: 0.88rem; color: #888; margin-bottom: 2rem; line-height: 1.5; }


  /* ── Input fields ── */
  .stTextInput > div > div > input,
  .stTextInput > div > div > input:focus {
    border: 1.5px solid #d0d0d0;
    border-radius: 6px;
    background-color: #ffffff;
    color: #0a0a0a;
    font-size: 0.95rem;
    padding: 0.5rem 0.8rem;
  }
  .stTextInput > div > div > input:focus {
    border-color: #1a6fc4;
    box-shadow: 0 0 0 3px rgba(26, 111, 196, 0.12);
  }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #e8e8e8; margin: 1.5rem 0; }

  /* ── Card ── */
  .card {
    background: #ffffff;
    border: 1.5px solid #e0eaf5;
    border-radius: 12px;
    padding: 1.8rem;
    height: 100%;
    transition: border-color 0.2s, box-shadow 0.2s;
  }
  .card:hover {
    border-color: #1a6fc4;
    box-shadow: 0 4px 16px rgba(26, 111, 196, 0.1);
  }
  .card-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1a6fc4;
    margin-bottom: 0.5rem;
  }
  .card-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 0.4rem; color: #0a0a0a; }
  .card-desc  { font-size: 0.9rem; color: #5a5a5a; line-height: 1.6; }

  /* ── Bento (dashboard cards) ── */
  .bento {
    background: #ffffff;
    border: 1.5px solid #e0eaf5;
    border-radius: 14px;
    padding: 2rem;
    min-height: 220px;
    transition: all 0.22s;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .bento:hover { border-color: #1a6fc4; box-shadow: 0 6px 24px rgba(26, 111, 196, 0.1); }
  .bento-tag {
    display: inline-block;
    background: #f0f6ff;
    color: #1a6fc4;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 1rem;
  }
  .bento-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #0a0a0a; }
  .bento-desc  { font-size: 0.88rem; color: #5a5a5a; line-height: 1.55; flex-grow: 1; }
  .bento-coming {
    display: inline-block;
    background: #f5f5f5;
    color: #999;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
  }

  /* ── Topbar ── */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.75rem 0;
    border-bottom: 1px solid #e8e8e8;
    margin-bottom: 2rem;
  }
  .topbar-brand { font-size: 1.1rem; font-weight: 700; color: #0a0a0a; letter-spacing: -0.01em; }
  .topbar-nav   { display: flex; gap: 2rem; align-items: center; }
  .topbar-link  { font-size: 0.9rem; color: #5a5a5a; text-decoration: none; font-weight: 450; }

  /* ── Pill badge ── */
  .pill {
    display: inline-block;
    background: #f0f6ff;
    color: #1a6fc4;
    border: 1px solid #cce0f5;
    border-radius: 20px;
    padding: 0.25rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
  }

  /* ── Feature panel (3-panel layout) ── */
  .panel-header {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #1a6fc4;
    padding: 0.5rem 0;
    border-bottom: 1.5px solid #e0eaf5;
    margin-bottom: 1rem;
  }
  .panel-wrap {
    border: 1.5px solid #e0eaf5;
    border-radius: 10px;
    padding: 0;
    height: 600px;
    background: #fafbff;
    overflow-y: auto;
    overflow-x: hidden;
  }
  .panel-wrap-inner {
    padding: 1rem;
  }

  /* ── Metric strip ── */
  .metric-strip {
    display: flex; gap: 1.5rem; align-items: center;
    padding: 0.8rem 1.2rem;
    background: #f7f9ff;
    border: 1px solid #e0eaf5;
    border-radius: 8px;
    margin-bottom: 1rem;
    font-size: 0.85rem; color: #3a3a3a;
  }
  .metric-val { font-weight: 700; color: #0a0a0a; font-size: 1rem; }

  /* ── Error table ── */
  .err-row { padding: 0.6rem 0; border-bottom: 1px solid #f0f0f0; }
  .err-type-SPELL { color: #b8860b; font-weight: 600; font-size: 0.8rem; }
  .err-type-GRAM  { color: #cc5500; font-weight: 600; font-size: 0.8rem; }
  .err-type-SEM   { color: #c0392b; font-weight: 600; font-size: 0.8rem; }
  .err-word { font-weight: 600; font-size: 0.9rem; color: #0a0a0a; }
  .err-suggestion { font-size: 0.85rem; color: #5a5a5a; }

  /* ── Auth form ── */
  .auth-wrap {
    max-width: 420px;
    margin: 4rem auto;
    padding: 2.5rem;
    border: 1.5px solid #e0eaf5;
    border-radius: 14px;
    background: #ffffff;
  }
  .auth-title { font-size: 1.5rem; font-weight: 700; margin-bottom: 0.25rem; }
  .auth-sub   { font-size: 0.9rem; color: #5a5a5a; margin-bottom: 2rem; }

  /* ── Scrollable page panel ── */
  div[data-testid="stVerticalBlock"] { gap: 0.75rem; }

  /* ── Info highlight ── */
  .highlight-box {
    background: #f7f9ff;
    border-left: 3px solid #1a6fc4;
    padding: 0.9rem 1.2rem;
    border-radius: 0 8px 8px 0;
    font-size: 0.9rem;
    color: #3a3a3a;
    margin: 1rem 0;
  }
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "page":       "home",
        "user":       None,
        "users_db":   {"demo@nyayai.in": "demo1234"},
        "result":     None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

def go(page): st.session_state.page = page; st.rerun()

# ── Logo helper — transparent PNG ───────────────────────────────────────────
def _logo(width=52):
    if LOGO_PATH.exists():
        b64 = base64.b64encode(LOGO_PATH.read_bytes()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" width="{width}" style="display:block;">',
            unsafe_allow_html=True,
        )

# ── Backend helper ────────────────────────────────────────────────────────────
@st.cache_data(ttl=20, show_spinner=False)
def _health():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=4)
        return r.ok, r.json() if r.ok else {}
    except Exception:
        return False, {}


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    # Topbar
    st.markdown("""
    <div class="topbar">
      <span class="topbar-brand">NyayAI</span>
      <span class="topbar-nav">
        <span class="topbar-link">Platform</span>
        <span class="topbar-link">Research</span>
        <span class="topbar-link">About</span>
      </span>
    </div>""", unsafe_allow_html=True)

    # Hero — use HTML for button alignment so both sit on same baseline
    col_hero, col_img = st.columns([3, 2], gap="large")
    with col_hero:
        st.markdown('<div class="pill">AI for Indian Legal Documents</div>', unsafe_allow_html=True)
        st.markdown("# Verify Legal Documents with Precision")
        st.markdown("""
        <p>NyayAI applies fine-tuned language models to detect spelling, grammar,
        and semantic errors in Indian legal PDFs — FIRs, petitions, judgments, and contracts.
        Every word is checked. Every error is located. Every filing made clean.</p>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        # Align buttons using flex
        st.markdown("""
        <div style="display:flex;gap:15px;align-items:center;">
        """, unsafe_allow_html=True)
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("Get Started", key="hero_start", use_container_width=True):
                go("signup")
        with b2:
            st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
            if st.button("Sign In", key="hero_signin", use_container_width=True):
                go("login")
            st.markdown('</div>', unsafe_allow_html=True)
            
        st.markdown("</div>", unsafe_allow_html=True)

    with col_img:
        logo_b64 = _get_logo_b64()
        if logo_b64:
            st.markdown(
                f'<div style="display:flex; justify-content:center; align-items:center; height:100%; min-height:280px;"><img src="data:image/png;base64,{logo_b64}" width="280" style="object-fit:contain;"/></div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr>", unsafe_allow_html=True)

    # How it works
    st.markdown("## How it works")
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    steps = [
        ("01", "Upload", "Submit any Indian legal PDF — scanned or digital."),
        ("02", "OCR & Extract", "Surya OCR extracts every word with its precise location on the page."),
        ("03", "AI Analysis", "InLegalBERT classifies each word as correct or erroneous."),
        ("04", "Review", "Errors are highlighted on the document with correction suggestions."),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-label">Step {num}</div>
              <div class="card-title">{title}</div>
              <div class="card-desc">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # What it detects
    st.markdown("## What NyayAI detects")
    st.markdown("<br>", unsafe_allow_html=True)
    d1, d2, d3 = st.columns(3, gap="medium")
    types = [
        ("Spelling Errors", "Misspelt names, legal terms, and place names that bypass standard spell checkers trained on general English."),
        ("Grammar Errors", "Subject-verb disagreement, incorrect tense, and malformed sentence structures that alter legal meaning."),
        ("Semantic Errors", "Wrong IPC/CrPC section numbers, incorrect legal citations, and misreferenced clauses — the highest-risk category."),
    ]
    for col, (t, d) in zip([d1, d2, d3], types):
        with col:
            st.markdown(f"""
            <div class="card">
              <div class="card-title">{t}</div>
              <div class="card-desc">{d}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Stats row
    st.markdown("## Built on robust foundations")
    st.markdown("<br>", unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, (val, label) in zip([s1, s2, s3, s4], [
        ("120,000", "Training Examples"),
        ("92.1%", "F1 Score on Test Set"),
        ("2.58M", "Legal Sentences Available"),
        ("InLegalBERT", "Base Model"),
    ]):
        with col:
            st.markdown(f"""
            <div style="text-align:center;padding:1.5rem;border:1.5px solid #e0eaf5;border-radius:10px;">
              <div style="font-size:1.8rem;font-weight:800;color:#0a0a0a;">{val}</div>
              <div style="font-size:0.85rem;color:#7a7a7a;margin-top:0.3rem;">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;">
      <div style="font-size:0.78rem;color:#aaa;">
        NyayAI — Legal Intelligence Platform &nbsp;|&nbsp; law-ai/InLegalBERT &nbsp;|&nbsp; Surya OCR
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _auth_left_panel(tag: str, title: str, sub: str, quote: str, stats: list):
    """Returns HTML string for the dark left panel."""
    logo_html = ""
    logo_b64 = _get_logo_b64()
    if logo_b64:
        logo_html = f'<div style="margin-bottom:2.5rem;"><img src="data:image/png;base64,{logo_b64}" height="32" style="filter:invert(1) brightness(2); object-fit:contain;"/></div>'

    stats_html = "".join([
        f'<div><div class="auth-left-stat-val">{v}</div><div class="auth-left-stat-label">{l}</div></div>'
        for v, l in stats
    ])

    return f"""
    <div class="auth-left">
      <div>
        {logo_html}
        <div class="auth-left-tag">{tag}</div>
        <div class="auth-left-title">{title}</div>
        <div class="auth-left-sub">{sub}</div>
        <div class="auth-left-quote">{quote}</div>
        <div class="auth-left-stats">{stats_html}</div>
      </div>
      <div style="font-size:0.7rem;color:rgba(255,255,255,0.18);margin-top:2rem;">
        law-ai/InLegalBERT &nbsp;&middot;&nbsp; Surya OCR
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SIGN IN
# ══════════════════════════════════════════════════════════════════════════════
def page_login():
    # Left panel rendered as HTML — right panel uses Streamlit columns
    panel_html = _auth_left_panel(
        tag="AI Legal Platform",
        title="The precision tool Indian courts need.",
        sub="NyayAI catches spelling, grammar, and semantic errors in legal documents before they reach the bench — automatically, accurately, at scale.",
        quote="A single misquoted section can cost a client their case. NyayAI makes sure it never happens.",
        stats=[("92.1%","F1 Score"),("120K","Training examples"),("2.58M","Legal sentences")],
    )

    left, right = st.columns([5, 7], gap="medium")

    with left:
        st.markdown(panel_html, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-right">', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-logo">NyayAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-title">Welcome back</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-sub">Sign in to continue to your dashboard</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        email    = st.text_input("Email address", placeholder="you@firm.com", key="li_email")
        password = st.text_input("Password", type="password", placeholder="Password", key="li_pass")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Sign In", key="li_btn", use_container_width=True):
            db = st.session_state.users_db
            if email in db and db[email] == password:
                st.session_state.user = email
                go("dashboard")
            else:
                st.error("Invalid email or password.")

        st.markdown('<p style="text-align:center;font-size:0.85rem;color:#888;margin-top:1.5rem;">Don\'t have an account?</p>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2, gap="small")
        with col_a:
            if st.button("Create account", key="li_to_signup", use_container_width=True):
                go("signup")
        with col_b:
            if st.button("Back to home", key="li_home", use_container_width=True):
                go("home")

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SIGN UP
# ══════════════════════════════════════════════════════════════════════════════
def page_signup():
    panel_html = _auth_left_panel(
        tag="Get Started — Free",
        title="Built for advocates. Trusted by courts.",
        sub="Join legal professionals using NyayAI to eliminate document errors before filing — from district courts to the Supreme Court of India.",
        quote="Trained on 2.58 million Indian legal sentences. Powered by InLegalBERT.",
        stats=[("3","Error types"),("95.1%","Precision"),("89.2%","Recall")],
    )

    left, right = st.columns([5, 7], gap="medium")

    with left:
        st.markdown(panel_html, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="auth-right">', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-logo">NyayAI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-title">Create your account</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-right-sub">Start verifying legal documents with AI</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        name     = st.text_input("Full name", placeholder="Advocate Arjun Sharma", key="su_name")
        email    = st.text_input("Email address", placeholder="you@firm.com", key="su_email")
        password = st.text_input("Password", type="password", placeholder="Min. 8 characters", key="su_pass")
        confirm  = st.text_input("Confirm password", type="password", placeholder="Repeat password", key="su_conf")
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("Create Account", key="su_btn", use_container_width=True):
            if not name or not email or not password:
                st.error("All fields are required.")
            elif password != confirm:
                st.error("Passwords do not match.")
            elif len(password) < 8:
                st.error("Password must be at least 8 characters.")
            elif email in st.session_state.users_db:
                st.error("An account with this email already exists.")
            else:
                st.session_state.users_db[email] = password
                st.session_state.user = email
                go("dashboard")


        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Sign In", key="su_to_login", use_container_width=True):
                go("login")
        with col_b:
            if st.button("Back to home", key="su_home", use_container_width=True):
                go("home")

        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def page_dashboard():
    # Topbar
    user = st.session_state.get("user", "")
    col_brand, col_user = st.columns([6, 1])
    with col_brand:
        st.markdown(f"""
        <div class="topbar">
          <span class="topbar-brand">NyayAI</span>
          <span class="topbar-nav">
            <span class="topbar-link">Dashboard</span>
            <span class="topbar-link">History</span>
          </span>
        </div>""", unsafe_allow_html=True)
    with col_user:
        if st.button("Sign Out", key="db_signout"):
            st.session_state.user = None
            go("home")

    # Welcome
    st.markdown(f"## Platform")
    st.markdown(f'<p style="color:#5a5a5a;margin-top:-0.8rem;">Select a feature to get started.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Bento grid — row 1
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="bento">
          <div>
            <div class="bento-tag">Available</div>
            <div class="bento-title">Document Error Checker</div>
            <div class="bento-desc">
              Upload a legal PDF and detect spelling, grammar, and semantic errors.
              The model highlights each error with a bounding box and suggests corrections.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Open Feature", key="db_checker", use_container_width=True):
            st.session_state.result = None
            go("checker")

    with c2:
        st.markdown("""
        <div class="bento">
          <div>
            <div class="bento-coming">Coming Soon</div>
            <div class="bento-title">Case Law Reference Validator</div>
            <div class="bento-desc">
              Cross-check cited judgments and section references against the Supreme Court
              and High Court database to flag invalid or misquoted citations.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("Coming Soon", key="db_validator", use_container_width=True, disabled=True):
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="bento">
          <div>
            <div class="bento-coming">Coming Soon</div>
            <div class="bento-title">Contract Clause Analyzer</div>
            <div class="bento-desc">
              Identify missing, ambiguous, or non-standard clauses in contracts and
              agreements under Indian contract law.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="btn-outline">', unsafe_allow_html=True)
        if st.button("Coming Soon", key="db_clauses", use_container_width=True, disabled=True):
            pass
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bento grid — row 2
    c4, c5 = st.columns([2, 1], gap="medium")

    with c4:
        st.markdown("""
        <div class="bento">
          <div>
            <div class="bento-coming">Coming Soon</div>
            <div class="bento-title">Multilingual Support — Hindi & Regional Languages</div>
            <div class="bento-desc">
              Extend error detection to Hindi, Marathi, Tamil, and other vernacular
              legal documents using multilingual LegalBERT variants.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="bento" style="background:#f7f9ff;">
          <div>
            <div class="bento-tag">Model</div>
            <div class="bento-title">InLegalBERT</div>
            <div class="bento-desc">
              F1 Score &nbsp;<strong>92.1%</strong><br>
              Precision &nbsp;<strong>95.1%</strong><br>
              Recall &nbsp;<strong>89.2%</strong><br>
              120,000 training examples
            </div>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — 3-PANEL ERROR CHECKER
# ══════════════════════════════════════════════════════════════════════════════
def page_checker():
    # Top bar with exit
    col_title, col_exit = st.columns([8, 1])
    with col_title:
        st.markdown("""
        <div style="padding:0.6rem 0; border-bottom:1px solid #e8e8e8; margin-bottom:1.5rem;">
          <span style="font-size:1rem;font-weight:600;color:#0a0a0a;">Document Error Checker</span>
          <span style="font-size:0.85rem;color:#aaa;margin-left:1rem;">
            Upload a PDF to analyse
          </span>
        </div>""", unsafe_allow_html=True)
    with col_exit:
        if st.button("Exit", key="chk_exit"):
            st.session_state.result = None
            go("dashboard")

    # Upload widget (shown before analysis)
    if st.session_state.result is None:
        st.markdown('<div style="max-width:500px;margin:3rem auto;">', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.95rem;color:#3a3a3a;margin-bottom:0.5rem;">Upload a legal PDF document</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="chk_upload")
        st.markdown("</div>", unsafe_allow_html=True)

        if uploaded:
            ok, _ = _health()
            if not ok:
                st.error(f"Backend is not reachable at {BACKEND_URL}. Start the backend first.")
                return

            with st.spinner("Running OCR and error detection — this may take up to 60 seconds on first run..."):
                try:
                    resp = requests.post(
                        f"{BACKEND_URL}/analyze-full",
                        files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                        timeout=TIMEOUT_S,
                    )
                    if not resp.ok:
                        st.error(f"Backend returned {resp.status_code}: {resp.text}")
                        return
                    data = resp.json()
                    data["_original_bytes"] = uploaded.getvalue()
                    st.session_state.result = data
                    st.rerun()
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The backend is still processing — try again.")
                except Exception as e:
                    st.error(f"Error: {e}")
        return

    # ── Results: 3-panel layout ───────────────────────────────────────────────
    data   = st.session_state.result
    errors = data.get("errors", [])
    pages  = data.get("pages_b64", [])

    # Metric strip
    summary   = data.get("summary", {})
    n_words   = data.get("total_words", 0)
    n_errors  = len(errors)
    ocr_t     = data.get("ocr_time_s", 0)
    inf_t     = data.get("inference_time_s", 0)

    st.markdown(f"""
    <div class="metric-strip">
      <span>Pages: <span class="metric-val">{data.get("page_count", 0)}</span></span>
      <span>Words: <span class="metric-val">{n_words:,}</span></span>
      <span>Errors: <span class="metric-val">{n_errors}</span></span>
      <span>Spelling: <span class="metric-val">{summary.get("SPELL", 0)}</span></span>
      <span>Grammar: <span class="metric-val">{summary.get("GRAM", 0)}</span></span>
      <span>Semantic: <span class="metric-val">{summary.get("SEM", 0)}</span></span>
      <span style="margin-left:auto;color:#aaa;font-size:0.8rem;">
        OCR {ocr_t}s &nbsp;| Inference {inf_t}s
      </span>
    </div>""", unsafe_allow_html=True)

    # Page selector (compact)
    page_count = data.get("page_count", 1)
    if page_count > 1:
        sel_page = st.slider("Page", 1, page_count, 1, key="chk_page") - 1
    else:
        sel_page = 0

    st.markdown("<br>", unsafe_allow_html=True)

    # ── FIXED: 3-panel layout with Internal Scrolling ─────────────────────────
    # We use a standard ratio. col2 (Annotated) is usually the most important.
    left, mid, right = st.columns([1, 1.2, 0.8], gap="small")

    # Set a consistent height for all panels so they align perfectly
    PANEL_HEIGHT = 700 

    # LEFT — original document
    with left:
        st.markdown('<div class="panel-header">Original Document</div>', unsafe_allow_html=True)
        # container(height=...) creates the fixed scrollable box
        with st.container(height=PANEL_HEIGHT, border=True):
            orig_bytes = data.get("_original_bytes")
            if orig_bytes:
                try:
                    import fitz
                    doc = fitz.open(stream=orig_bytes, filetype="pdf")
                    pix = doc.load_page(sel_page).get_pixmap(matrix=fitz.Matrix(1.2, 1.2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    st.image(img, use_column_width=True)
                    doc.close()
                except Exception as e:
                    st.caption(f"Error: {e}")

    # MIDDLE — annotated document
    with mid:
        st.markdown('<div class="panel-header">Annotated Output</div>', unsafe_allow_html=True)
        with st.container(height=PANEL_HEIGHT, border=True):
            if pages and sel_page < len(pages):
                img_bytes = base64.b64decode(pages[sel_page])
                img = Image.open(io.BytesIO(img_bytes))
                st.image(img, use_column_width=True)
                st.markdown("""
                <div style="font-size:0.7rem; color:#888; text-align:center; margin-top:10px;">
                  🟡 Spelling &nbsp; 🟠 Grammar &nbsp; 🔴 Semantic
                </div>""", unsafe_allow_html=True)

    # RIGHT — error list
    with right:
        st.markdown('<div class="panel-header">Errors & Suggestions</div>', unsafe_allow_html=True)
        with st.container(height=PANEL_HEIGHT, border=True):
            page_errors = [e for e in errors if e.get("page", 0) == sel_page]
            if not page_errors:
                st.info("No errors on this page.")
            else:
                for err in sorted(page_errors, key=lambda x: (x.get("y0", 0))):
                    et = err.get("error_type", "SPELL")
                    # Visual styling based on error type
                    color = {"SPELL": "#b8860b", "GRAM": "#cc5500", "SEM": "#c0392b"}.get(et, "#000")
                    
                    with st.expander(f"**{err.get('word')}**", expanded=True):
                        st.markdown(f"<span style='color:{color}; font-weight:bold;'>{et}</span>", unsafe_allow_html=True)
                        st.write(f"Fix: {err.get('suggestion')}")
                        st.caption(f"Confidence: {err.get('confidence', 0):.0%}")
        st.markdown('</div></div>', unsafe_allow_html=True)

        # Download
        if errors:
            st.markdown("<br>", unsafe_allow_html=True)
            rows = []
            for e in errors:
                et = e.get("error_type", "?")
                rows.append({
                    "Page":       int(e.get("page", 0)) + 1,
                    "Word":       e.get("word", ""),
                    "Type":       {"SPELL": "Spelling", "GRAM": "Grammar", "SEM": "Semantic"}.get(et, et),
                    "Suggestion": e.get("suggestion", ""),
                    "Confidence": f"{e.get('confidence', 0):.0%}",
                })
            csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button(
                "Download Full Report (CSV)",
                csv,
                file_name="nyayai_errors.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
page = st.session_state.get("page", "home")

if page == "home":
    page_home()
elif page == "login":
    page_login()
elif page == "signup":
    page_signup()
elif page == "dashboard":
    if st.session_state.user is None:
        go("login")
    else:
        page_dashboard()
elif page == "checker":
    if st.session_state.user is None:
        go("login")
    else:
        page_checker()
else:
    page_home()