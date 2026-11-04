"""
ArthaMart — Dashboard branding (Phase 12 polish)
==================================================
Visual identity: brand palette (light + dark), custom CSS, sidebar shell
with logo, page navigation and the dark/light toggle.
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "logo.png"

# ---------------------------------------------------------------------------
# Brand palettes
# ---------------------------------------------------------------------------

def palette(dark: bool) -> dict:
    if dark:
        return {
            "primary": "#3B82F6", "primary_dark": "#2563EB",
            "accent": "#22C55E", "warn": "#F59E0B", "danger": "#EF4444",
            "ink": "#E7ECF7", "muted": "#94A3B8",
            "bg": "#0B1220", "card": "#121B30", "line": "#243452",
            "hero_from": "#1E3A8A", "hero_to": "#4338CA",
        }
    return {
        "primary": "#2563EB", "primary_dark": "#1E40AF",
        "accent": "#16A34A", "warn": "#D97706", "danger": "#DC2626",
        "ink": "#0F172A", "muted": "#64748B",
        "bg": "#F7F8FC", "card": "#FFFFFF", "line": "#E2E8F0",
        "hero_from": "#1E40AF", "hero_to": "#4338CA",
    }


CARD_KEYS = {
    "URGENT_MARKDOWN": "danger",
    "INVEST_REORDER": "primary",
    "HOLD": "muted",
    "BUNDLE": "accent",
}

PAGES = ["Decision Dashboard", "Order Queue", "OCR Intake",
         "Personalisation", "Model Performance", "Simulation Results",
         "AI Assistant"]

NAV_GROUPS = [
    ("OPERATIONS", ["Decision Dashboard", "Order Queue", "OCR Intake"]),
    ("CUSTOMERS", ["Personalisation"]),
    ("INTELLIGENCE", ["Model Performance", "Simulation Results",
                      "AI Assistant"]),
]

PAGE_ICONS = {"Decision Dashboard": "🎯", "Order Queue": "📋",
              "OCR Intake": "🧾", "Personalisation": "👥",
              "Model Performance": "📈", "Simulation Results": "⚖️",
              "AI Assistant": "🤖"}


# ---------------------------------------------------------------------------
# streamlit theme switching
# ---------------------------------------------------------------------------

# NOTE: Streamlit locks ALL theme.* options to config/CLI in current
# versions, so the runtime toggle drives the whole look through the CSS
# layer below instead (surfaces, text, widgets, cards, charts).


def apply_css(dark: bool) -> None:
    b = palette(dark)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, Segoe UI, Roboto, sans-serif;
        }}
        /* broad text recolour for the selected mode */
        p, li, ul, ol, h1, h2, h3, h4, label, span,
        [data-testid="stMarkdownContainer"], .stMarkdown,
        [data-testid="stCaptionContainer"] {{
            color: {b['ink']};
        }}
        .stApp, section.main {{ background: {b['bg']}; }}
        .block-container {{ padding-top: 1.2rem; max-width: 1200px; }}

        /* hero */
        .hero {{
            background: linear-gradient(120deg, {b['hero_from']} 0%,
                        {b['hero_to']} 100%);
            color:#fff; border-radius:18px; padding:1.3rem 1.8rem 1.1rem;
            margin-bottom:1.1rem; box-shadow:0 10px 30px rgba(37,99,235,.25);
        }}
        .hero h1 {{ color:#fff; font-weight:800; font-size:1.7rem;
                    margin:0 0 .15rem; letter-spacing:-.02em; }}
        .hero .sub {{ color:rgba(255,255,255,.85); font-size:.92rem; }}
        .hero .pill {{ display:inline-block; background:rgba(255,255,255,.16);
            border:1px solid rgba(255,255,255,.25); padding:.12rem .65rem;
            border-radius:999px; font-size:.72rem; margin-right:.35rem;
            margin-top:.5rem; font-weight:600; }}

        /* sidebar shell — fixed, non-scrolling on normal screens */
        section[data-testid="stSidebar"] {{
            background:{b['card']}; border-right:1px solid {b['line']};
            overflow:hidden; }}
        @media (max-height: 660px) {{
            section[data-testid="stSidebar"] {{ overflow-y:auto; }} }}
        section[data-testid="stSidebar"]
          [data-testid="stVerticalBlock"] {{ min-height:100%; }}
        /* the spacer element grows, pinning theme+account to the bottom */
        section[data-testid="stSidebar"]
          [data-testid="stElementContainer"]:has(.sb-spacer) {{
            flex:1 1 auto; min-height:10px; }}
        .sb-spacer {{ height:1px; }}
        .sb-rule {{ height:1px; background:{b['line']}; margin:10px 0 14px; }}

        /* signed-in user chip */
        .uchip {{ display:flex; align-items:center; gap:10px;
            background:{b['bg']}; border:1px solid {b['line']};
            border-radius:12px; padding:8px 10px; margin:.4rem 0 .8rem; }}
        .uava {{ width:36px; height:36px; border-radius:50%;
            background:linear-gradient(135deg, {b['primary']}, #7C3AED);
            color:#fff; font-weight:800; font-size:.85rem;
            display:flex; align-items:center; justify-content:center;
            flex-shrink:0; }}
        .uname {{ font-weight:700; font-size:.88rem; color:{b['ink']};
            line-height:1.1; }}
        .umail {{ font-size:.7rem; color:{b['muted']}; }}
        /* nav/theme block buttons: left-aligned, product-menu look */
        section[data-testid="stSidebar"] .stButton > button {{
            justify-content:flex-start; text-align:left;
            border-radius:10px; font-weight:600; }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            background:{b['bg']}; color:{b['ink']}; }}
        section[data-testid="stSidebar"] .brand-name {{
            font-weight:800; font-size:1.08rem; letter-spacing:-.01em;
            color:{b['ink']}; }}
        section[data-testid="stSidebar"] .brand-tag {{
            font-size:.68rem; color:{b['muted']}; font-weight:600; }}

        /* metrics */
        [data-testid="stMetric"] {{
            background:{b['card']}; border:1px solid {b['line']};
            border-left:4px solid {b['primary']}; border-radius:14px;
            padding:12px 15px; box-shadow:0 2px 10px rgba(0,0,0,.06); }}
        [data-testid="stMetricLabel"] p {{ color:{b['muted']};
            font-weight:600; }}
        [data-testid="stMetricValue"] {{ color:{b['ink']}; font-weight:800; }}

        /* decision cards */
        .dcard {{ background:{b['card']}; border:1px solid {b['line']};
            border-radius:14px; padding:13px 15px; margin-bottom:11px;
            box-shadow:0 2px 8px rgba(0,0,0,.06);
            border-top:3px solid var(--dc, {b['primary']}); }}
        .dcard .dhead {{ display:flex; justify-content:space-between;
            align-items:center; margin-bottom:5px; gap:10px; }}
        .dcard .dtitle {{ font-weight:700; font-size:.98rem;
            color:{b['ink']}; }}
        .dcard .dval {{ font-weight:800; font-size:1.02rem; color:var(--dc);
            white-space:nowrap; }}
        .dcard .drow {{ font-size:.83rem; color:{b['muted']}; margin-top:3px; }}
        .dcard .drow b {{ color:{b['ink']}; font-weight:600; }}

        /* buttons */
        .stButton > button {{ background:{b['primary']}; color:#fff;
            border:none; border-radius:10px; padding:.5rem 1.1rem;
            font-weight:600; transition:.15s; }}
        .stButton > button:hover {{ background:{b['primary_dark']};
            color:#fff; }}

        /* tables */
        [data-testid="stDataFrame"] {{ border-radius:12px; overflow:hidden;
            border:1px solid {b['line']}; }}

        .foot {{ text-align:center; color:{b['muted']}; font-size:.78rem;
            margin-top:1.6rem; padding-top:.9rem;
            border-top:1px solid {b['line']}; }}

        #MainMenu, footer, header {{ visibility:hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# layout chrome
# ---------------------------------------------------------------------------

def sidebar(user: dict | None = None) -> tuple[str, bool]:
    """Professional fixed sidebar.

    Layout (top → bottom): brand row · grouped navigation · flex spacer ·
    theme toggle · account chip. The spacer + overflow rules keep the panel
    pinned (no scroll) on normal screens.
    """
    with st.sidebar:
        # ---- brand row (compact logo + wordmark) ----------------------
        bc1, bc2 = st.columns([1, 3.2], vertical_alignment="center")
        if LOGO_PATH.exists():
            bc1.image(str(LOGO_PATH), use_container_width=True)
        bc2.markdown('<div class="brand-name">ArthaMart</div>'
                     '<div class="brand-tag">Intelligent Retailing '
                     "Suite</div>", unsafe_allow_html=True)
        st.markdown('<div class="sb-rule"></div>',
                    unsafe_allow_html=True)

        # ---- navigation (top): clean block buttons — no radio circles,
        #      no checkboxes; the active page is a filled block ----------
        if "page" not in st.session_state:
            st.session_state.page = PAGES[0]
        sel = st.session_state.page
        for title, opts in NAV_GROUPS:
            st.caption(title)
            for p in opts:
                if st.button(f"{PAGE_ICONS[p]}  {p}", key=f"nav_{p}",
                             use_container_width=True,
                             type="primary" if p == sel else "secondary"):
                    st.session_state.page = p
                    st.rerun()

        # ---- flex spacer pins the rest to the bottom ------------------
        st.markdown('<div class="sb-spacer"></div>',
                    unsafe_allow_html=True)

        # ---- bottom: theme (a button, not a checkbox) + account -------
        if "dark" not in st.session_state:
            st.session_state.dark = st.query_params.get("theme") == "dark"
        dark = st.session_state.dark
        if st.button("🌙 Dark mode" if not dark else "☀️ Light mode",
                     key="btn_theme", use_container_width=True,
                     type="primary" if dark else "secondary"):
            st.session_state.dark = not dark
            st.query_params["theme"] = "dark" if not dark else "light"
            st.rerun()

        if user:
            initials = "".join(w[0] for w in user["name"].split()[:2]).upper()
            prov = "· Google" if "google" in user.get("provider", "") else ""
            st.markdown(
                f"""
                <div class="uchip">
                  <div class="uava">{initials}</div>
                  <div>
                    <div class="uname">{user['name']}</div>
                    <div class="umail">{user['email']}{prov}</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
            if st.button("Sign out", key="btn_logout",
                         use_container_width=True):
                st.session_state.pop("auth_user", None)
                st.rerun()
    return st.session_state.page, dark




def hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🛒 ArthaMart</h1>
          <div class="sub">Intelligent Retailing — Forecasting · Pricing ·
          Inventory · Personalisation</div>
          <span class="pill">v1.1</span>
          <span class="pill">AI-powered decisions</span>
          <span class="pill">6 modules</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_css(v) -> str:
    """Pandas-Styler callback for match-status tables — must return a CSS
    *declaration* (or ''), never a bare hex colour, otherwise Styler raises
    ValueError the moment a slip is uploaded."""
    c = {"MATCHED": "#c8e6c9", "REVIEW": "#fff9c4",
         "UNMATCHED": "#ffcdd2"}.get(v)
    return f"background-color: {c}" if c else ""


def render_card(card, dark: bool) -> str:
    b = palette(dark)
    color = b[CARD_KEYS.get(card.card_type, "primary")]
    return f"""
    <div class="dcard" style="--dc:{color}">
      <div class="dhead">
        <span class="dtitle">{card.icon} {card.title} — {card.product_name}</span>
        <span class="dval">₹{card.impact_inr:,.0f}</span>
      </div>
      <div class="drow"><b>Do:</b> {card.action}</div>
      <div class="drow"><b>Why:</b> {card.reason}</div>
    </div>
    """


def footer(db_rows: int | None = None, sim_delta_pct: float | None = None,
           dark: bool = False) -> None:
    """Professional two-column footer card (brand / system)."""
    b = palette(dark)
    from ..config import get as _get
    sys_bits = f"v{_get('project.version', '1.2.1')} · 12 phases · 161 tests"
    if db_rows is not None:
        sys_bits += f" · {db_rows:,} DB rows"
    if sim_delta_pct is not None:
        sys_bits += f" · system +{sim_delta_pct:.1f}% profit"
    st.markdown(
        f"""
        <div class="footcard">
          <div>
            <div class="fbrand">🛒 ArthaMart</div>
            <div>Intelligent Retailing Application — Reinventing Retail
            Operations with Intelligent Forecasting and
            Personalisation.</div>
          </div>
          <div>
            <div class="fhead">SYSTEM</div>
            {sys_bits}<br>
            Random Forest · K-Means · FP-Growth · EOQ/ROP<br>
            © 2026 ArthaMart
          </div>
        </div>
        <style>
        .footcard {{ display:grid; grid-template-columns:1.6fr 1fr;
            gap:26px; background:{b['card']}; border:1px solid {b['line']};
            border-radius:16px; padding:18px 22px; margin-top:2rem;
            color:{b['muted']}; font-size:.78rem; line-height:1.5; }}
        .fbrand {{ font-weight:800; font-size:1.05rem; color:{b['ink']};
            margin-bottom:.3rem; }}
        .fhead {{ font-weight:800; font-size:.7rem; letter-spacing:.08em;
            color:{b['primary']}; margin-bottom:.3rem; }}
        @media (max-width: 800px) {{ .footcard {{ grid-template-columns:1fr; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )
