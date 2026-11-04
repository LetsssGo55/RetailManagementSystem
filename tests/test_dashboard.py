"""Phase 10 — dashboard smoke tests (Streamlit AppTest, headless)."""
import uuid

import pytest
from streamlit.testing.v1 import AppTest

from src import config

APP = str(config.ROOT / "src" / "dashboard" / "app.py")

TEST_USER = {"email": "tester@example.com", "name": "Test User",
             "provider": "local"}


@pytest.fixture(scope="module")
def app():
    if not config.DB_PATH.exists():
        pytest.skip("run setup first")
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["auth_user"] = TEST_USER      # pre-authenticated session
    at.run()
    return at


def test_app_runs_without_exceptions(app):
    assert not app.exception, \
        [e.value for e in app.exception] if app.exception else ""


def test_sidebar_navigation_seven_block_buttons(app):
    labels = [b.label for b in app.sidebar.button]
    for page in ("Decision Dashboard", "Personalisation", "AI Assistant",
                 "Simulation Results"):
        assert any(page in lbl for lbl in labels), page
    # no radio circles / checkboxes anywhere in the sidebar
    assert len(app.sidebar.radio) == 0
    assert len(app.sidebar.toggle) == 0
    assert len(app.sidebar.checkbox) == 0


def test_dark_mode_is_button_not_checkbox(app):
    labels = [b.label for b in app.sidebar.button]
    assert any("Dark mode" in lbl or "Light mode" in lbl for lbl in labels)


def test_user_chip_rendered(app):
    chip = [m for m in app.sidebar.markdown if "Test User" in m.value]
    assert chip, "signed-in user chip missing"


def test_branded_hero_present(app):
    hero = [m for m in app.markdown if '<div class="hero">' in m.value]
    assert hero, "branded hero banner missing"


def test_decision_metrics_rendered(app):
    assert len(app.metric) >= 4


def test_footer_card_present(app):
    foot = [m for m in app.markdown if '<div class="footcard">' in m.value]
    assert foot, "professional footer missing"


# ---------------------------------------------------------------------------
# auth gate: signup through the UI
# ---------------------------------------------------------------------------

def test_auth_gate_and_local_signup():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception
    # not signed in → login card, no hero
    assert not [m for m in at.markdown if '<div class="hero">' in m.value]
    assert at.button(key="btn_login") is not None

    email = f"u{uuid.uuid4().hex[:8]}@example.com"
    at.text_input(key="su_name").set_value("Signup Tester")
    at.text_input(key="su_email").set_value(email)
    at.text_input(key="su_pass").set_value("secret1")
    at.text_input(key="su_pass2").set_value("secret1")
    at.button(key="btn_signup").click()
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    # signed in now → hero renders
    assert [m for m in at.markdown if '<div class="hero">' in m.value]


def test_auth_bad_login_rejected():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.text_input(key="login_email").set_value("ghost@example.com")
    at.text_input(key="login_pass").set_value("wrongpass")
    at.button(key="btn_login").click()
    at.run()
    assert at.error                                # invalid credentials msg
    assert not [m for m in at.markdown if '<div class="hero">' in m.value]


def test_status_css_returns_valid_declarations():
    """Regression: pandas Styler raises ValueError unless the mapper returns
    full CSS declarations (or ''), never bare hex colours."""
    import pandas as pd
    from src.dashboard import branding

    df = pd.DataFrame({"status": ["MATCHED", "REVIEW", "UNMATCHED", "???"]})
    # would raise ValueError before the fix
    html = df.style.map(branding.status_css, subset=["status"]).to_html()
    assert "background-color: #c8e6c9" in html
    assert branding.status_css("???") == ""

def test_assistant_answers_from_data(app):
    from src.dashboard import assistant
    ans = assistant.answer("which products need reordering")
    assert ans.strip()
    ans2 = assistant.answer("how much profit did the system make")
    assert "₹" in ans2
    ans3 = assistant.answer("help")
    assert "reorder" in ans3.lower() and "bundle" in ans3.lower()
    # any product mention now yields a full brief, never a dead end
    ans4 = assistant.answer("tell me about Parle-G")
    assert "Parle" in ans4 and "₹" in ans4
