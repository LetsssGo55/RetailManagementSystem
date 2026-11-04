"""
ArthaMart — authentication screen (login / sign-up / Google).
Rendered when no session user exists; returns the signed-in user dict.
"""
from __future__ import annotations

import uuid

import streamlit as st

from . import google_auth, store
from ..dashboard import branding


def _card_css(dark: bool) -> None:
    b = branding.palette(dark)
    st.markdown(f"""
    <style>
    .auth-wrap {{ max-width: 430px; margin: 4vh auto; }}
    .auth-card {{ background:{b['card']}; border:1px solid {b['line']};
        border-radius:18px; padding:26px 28px;
        box-shadow:0 12px 40px rgba(0,0,0,.12); text-align:center; }}
    .auth-title {{ font-weight:800; font-size:1.25rem; color:{b['ink']};
        margin:.6rem 0 .2rem; }}
    .auth-sub {{ color:{b['muted']}; font-size:.85rem; margin-bottom:1.2rem; }}
    </style>
    """, unsafe_allow_html=True)


def render(conn, dark: bool = False) -> dict | None:
    """Full-screen auth gate. Returns user dict after successful login.
    Already-signed-in sessions pass straight through without re-rendering
    the login card."""
    if st.session_state.get("auth_user"):
        return st.session_state["auth_user"]
    store.ensure_table(conn)
    _card_css(dark)

    # Google OAuth redirect callback lands here with ?code=
    if google_auth.is_configured():
        prof = google_auth.handle_callback()
        if prof and prof.get("email"):
            user = store.upsert_oauth_user(conn, prof["email"], prof["name"],
                                           "google")
            st.session_state["auth_user"] = user
            st.rerun()

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    col_l, col_c, col_r = st.columns([1, 2.2, 1])
    with col_c:
        if branding.LOGO_PATH.exists():
            st.image(str(branding.LOGO_PATH), width=92)
        st.markdown('<div class="auth-title">Welcome to ArthaMart</div>'
                    '<div class="auth-sub">Sign in to open the retail '
                    "intelligence dashboard</div>",
                    unsafe_allow_html=True)

        # ---- Google ----
        if google_auth.is_configured():
            if st.button("G  Continue with Google", use_container_width=True):
                st.session_state["oauth_state"] = uuid.uuid4().hex
                st.markdown(
                    f'<meta http-equiv="refresh" content="0;url='
                    f'{google_auth.login_url(st.session_state["oauth_state"])}'
                    '">', unsafe_allow_html=True)
                st.link_button(
                    "Open Google sign-in",
                    google_auth.login_url(st.session_state["oauth_state"]),
                    use_container_width=True)
                st.stop()
        else:
            if st.button("G  Continue with Google (demo)",
                         use_container_width=True):
                st.session_state["auth_user"] = store.upsert_oauth_user(
                    conn, **{**google_auth.demo_google_user(),
                             "provider": "google-demo"})
                st.rerun()
            st.caption("Demo mode — set GOOGLE_CLIENT_ID / "
                       "GOOGLE_CLIENT_SECRET to enable real Google OAuth.")

        st.divider()

        # ---- local login / signup ----
        tab_login, tab_signup = st.tabs(["Login", "Sign up"])
        with tab_login:
            le = st.text_input("Email", key="login_email",
                               placeholder="you@example.com")
            lp = st.text_input("Password", type="password",
                               key="login_pass")
            if st.button("Sign in", use_container_width=True,
                         key="btn_login"):
                em = (le or "").strip().lower()
                if store.is_locked(em):
                    st.error(f"Too many attempts. Try again in "
                             f"{store.lock_remaining(em)}s.")
                else:
                    user = store.check_local_login(conn, em, lp or "")
                    if user:
                        store.clear_fails(em)
                        st.session_state["auth_user"] = user
                        st.rerun()
                    else:
                        store.record_fail(em)
                        st.error("Invalid email or password.")
        with tab_signup:
            sn = st.text_input("Full name", key="su_name",
                               placeholder="Anadi Singh")
            se = st.text_input("Email", key="su_email",
                               placeholder="you@example.com")
            sp = st.text_input("Password (min 6 chars)", type="password",
                               key="su_pass")
            sp2 = st.text_input("Confirm password", type="password",
                                key="su_pass2")
            if st.button("Create account", use_container_width=True,
                         key="btn_signup"):
                if sp != sp2:
                    st.error("Passwords do not match.")
                else:
                    try:
                        user = store.create_local_user(conn, se or "",
                                                       sn or "", sp or "")
                        st.session_state["auth_user"] = user
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.caption("Local accounts are stored in `app_users` with "
                   "PBKDF2-HMAC-SHA256 salted hashes.")
        from ..config import get as _get
        st.caption(f"ArthaMart v{_get('project.version', '1.2.1')}")
    st.markdown("</div>", unsafe_allow_html=True)
    return st.session_state.get("auth_user")
