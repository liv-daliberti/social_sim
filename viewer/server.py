#!/usr/bin/env python3
"""
Unified viewer hub — port 5050.

Reverse-proxies:
  /exp1/*  →  exp1 prospective viewer        (localhost:5051)
  /exp2/*  →  exp2 biased-news viewer         (localhost:5052)  Simulator + Results
  /exp3/*  →  exp3 elections (static + train) (localhost:5053)

All HTML responses get a fetch() interceptor injected that rewrites
  fetch('/api/...')  →  fetch('/expN/api/...')
so sub-app JavaScript works through the hub without modification.
"""

import os
from pathlib import Path

import requests as _requests
from flask import Flask, Response, request, send_file

app = Flask(__name__)
_HUB_HTML = Path(__file__).resolve().parent / "hub.html"

_UPSTREAMS = {
    "exp1": "http://127.0.0.1:5051",
    "exp2": "http://127.0.0.1:5052",
    "exp3": "http://127.0.0.1:5053",
}

_SKIP_HEADERS = {
    "content-encoding",
    "transfer-encoding",
    "content-length",
    "connection",
}


def _inject_interceptor(html: str, prefix: str) -> str:
    snippet = (
        f"<script>(function(){{var P='{prefix}',o=window.fetch;"
        f"window.fetch=function(u,x){{if(typeof u==='string'&&u.startsWith('/api/'))u=P+u;"
        f"return o.call(this,u,x);}};}})()</script>"
    )
    if "</head>" in html:
        return html.replace("</head>", snippet + "</head>", 1)
    return snippet + html


def _proxy(prefix: str, path: str) -> Response:
    upstream = _UPSTREAMS[prefix]
    url = upstream.rstrip("/") + "/" + path.lstrip("/")

    try:
        upstream_resp = _requests.request(
            method=request.method,
            url=url,
            params=request.args,
            data=request.get_data(),
            headers={k: v for k, v in request.headers if k.lower() != "host"},
            allow_redirects=False,
            timeout=30,
        )
    except _requests.exceptions.ConnectionError:
        return Response(
            f"<h3 style='font-family:monospace;color:#ef4444'>"
            f"Cannot connect to {prefix} upstream ({upstream}). "
            f"Is the server running?</h3>",
            status=502,
            content_type="text/html",
        )

    ct = upstream_resp.headers.get("Content-Type", "")
    body = upstream_resp.content
    out_headers = {
        k: v for k, v in upstream_resp.headers.items() if k.lower() not in _SKIP_HEADERS
    }

    if "text/html" in ct:
        html = _inject_interceptor(upstream_resp.text, f"/{prefix}")
        return Response(
            html, status=upstream_resp.status_code, content_type=ct, headers=out_headers
        )

    return Response(
        body, status=upstream_resp.status_code, content_type=ct, headers=out_headers
    )


# ── routes ─────────────────────────────────────────────────────────────────────


@app.route("/")
def hub():
    return send_file(str(_HUB_HTML))


@app.route("/exp1/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/exp1/<path:path>", methods=["GET", "POST"])
def exp1_proxy(path):
    return _proxy("exp1", path or "")


@app.route("/exp2/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/exp2/<path:path>", methods=["GET", "POST"])
def exp2_proxy(path):
    return _proxy("exp2", path or "")


@app.route("/exp3/", defaults={"path": ""}, methods=["GET", "POST"])
@app.route("/exp3/<path:path>", methods=["GET", "POST"])
def exp3_proxy(path):
    return _proxy("exp3", path or "")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"\n  Unified Viewer  →  http://0.0.0.0:{port}")
    print(f"  Exp 1 proxied from  {_UPSTREAMS['exp1']}")
    print(f"  Exp 2 proxied from  {_UPSTREAMS['exp2']}")
    print(f"  Exp 3 proxied from  {_UPSTREAMS['exp3']}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
