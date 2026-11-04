"""Built-in web UI — API smoke tests against a live in-process server."""
import json
import threading
import urllib.request

import pytest

from src import config
from src.webui import server as ws


@pytest.fixture(scope="module")
def port():
    httpd = ws.make_server(0)                      # ephemeral port
    p = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield p
    httpd.shutdown()


def call(port, path, method="GET", body=None, token=None, ctype=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    if ctype:
        req.add_header("Content-Type", ctype)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_index_served(port):
    st, body = call(port, "/")
    assert st == 200 and b"ArthaMart" in body


def test_health(port):
    st, body = call(port, "/api/health")
    assert st == 200
    assert json.loads(body)["status"] == "ok"


def test_auth_and_protected(port):
    st, _ = call(port, "/api/cards")
    assert st == 401                                # protected without token

    import uuid
    st, body = call(port, "/api/auth/signup", "POST",
                    json.dumps({"email": f"webui-{uuid.uuid4().hex[:8]}@x.com",
                                "password": "secret1",
                                "name": "WebUI Test"}).encode(),
                    ctype="application/json")
    assert st == 200, body
    tok = json.loads(body)["token"]

    for ep in ("kpis", "cards", "orders", "inventory", "customers",
               "models", "simulation", "forecast"):
        st, body = call(port, f"/api/{ep}", token=tok)
        assert st == 200, ep
        assert len(body) > 10


def test_google_demo_login(port):
    st, body = call(port, "/api/auth/google-demo", "POST")
    assert st == 200
    assert json.loads(body)["user"]["provider"] == "google-demo"


def test_parity_endpoints(port):
    st, body = call(port, "/api/auth/google-demo", "POST")
    tok = json.loads(body)["token"]

    st, body = call(port, "/api/assistant", "POST",
                    json.dumps({"q": "stock of potato"}).encode(),
                    token=tok, ctype="application/json")
    assert st == 200 and "Potato" in json.loads(body)["answer"]

    st, body = call(port, "/api/recommend/1", token=tok)
    assert st == 200 and len(json.loads(body)) >= 1

    st, body = call(port, "/api/pending", token=tok)
    assert st == 200 and isinstance(json.loads(body), list)

    st, body = call(port, "/api/order/3004", token=tok)
    assert st == 200 and len(json.loads(body)) >= 1


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run setup first")
def test_ocr_endpoint(port):
    st, body = call(port, "/api/auth/google-demo", "POST")
    tok = json.loads(body)["token"]
    slip = config.ROOT / "assets" / "sample_slips" / "slip_clean.png"
    if not slip.exists():
        pytest.skip("no sample slip")
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
    except Exception:
        pytest.skip("tesseract not installed")
    st, body = call(port, "/api/ocr", "POST", slip.read_bytes(),
                    token=tok, ctype="image/png")
    assert st == 200
    lines = json.loads(body)
    assert sum(1 for x in lines if x["status"] == "MATCHED") >= 5
