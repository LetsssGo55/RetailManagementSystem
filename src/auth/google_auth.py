"""
ArthaMart — Google Sign-In (OAuth 2.0 authorization-code flow).

Real mode  : set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars
             (Google Cloud Console → Credentials → OAuth client, with
             redirect URI = your dashboard URL, e.g. http://localhost:8501).
Demo mode  : without credentials a clearly-labelled demo button signs in a
             sandbox Google identity so the flow can be presented in a viva.
"""
from __future__ import annotations

import os
import urllib.parse

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"
SCOPES = "openid email profile"


def is_configured() -> bool:
    return bool(os.environ.get("GOOGLE_CLIENT_ID")
                and os.environ.get("GOOGLE_CLIENT_SECRET"))


def _redirect_uri() -> str:
    # the URL the user is browsing (works for localhost and the e2b preview)
    import streamlit as st
    headers = st.context.headers if hasattr(st, "context") else None
    host = (headers or {}).get("Host") if headers else None
    if not host:
        host = "localhost:8501"
    return f"http://{host}/"


def login_url(state: str) -> str:
    params = {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": SCOPES,
        "state": state,
        "prompt": "select_account",
    }
    return AUTH_URL + "?" + urllib.parse.urlencode(params)


def handle_callback() -> dict | None:
    """Exchange ?code=… for a profile. Returns None when not a callback."""
    import streamlit as st
    code = st.query_params.get("code")
    if not code:
        return None
    st.query_params.clear()                       # one-time use
    import requests
    try:
        tok = requests.post(TOKEN_URL, data={
            "code": code,
            "client_id": os.environ.get("GOOGLE_CLIENT_ID", ""),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", ""),
            "redirect_uri": _redirect_uri(),
            "grant_type": "authorization_code",
        }, timeout=10).json()
        if "access_token" not in tok:
            st.error(f"Google OAuth failed: {tok.get('error', 'unknown')}")
            return None
        info = requests.get(
            USERINFO_URL,
            headers={"Authorization": f"Bearer {tok['access_token']}"},
            timeout=10).json()
        return {"email": info.get("email", ""),
                "name": info.get("name", "Google User"),
                "provider": "google"}
    except Exception as e:                          # network down etc.
        st.error(f"Could not complete Google sign-in: {e}")
        return None


def demo_google_user() -> dict:
    return {"email": "anadi.demo@gmail.com", "name": "Anadi (Google Demo)",
            "provider": "google-demo"}
