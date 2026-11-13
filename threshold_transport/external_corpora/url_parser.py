"""URL parsing for the SpaPhish adapter and the seven URL-structure features.

The parser in spaphish_adapter_and_scoring.py returns no URL for any SpaPhish message:
  URL_RE = re.compile(r"https?://[^\\s,'\\"<>\\\\)\\\\]]+")  (double-escaped in a raw string: the
  character class closes early and the pattern demands a literal ']' after one character), and
  ast.literal_eval cannot read SpaPhish's unquoted "[url1],[url2]" format.

Only the URL splitting differs here. clean_domain, subdomain_depth and the feature definitions
are the same as in spaphish_adapter_and_scoring.py, so the seven URL features keep their meaning.
"""
from __future__ import annotations

import ast
import re
from urllib.parse import urlparse

import numpy as np

URL_RE = re.compile(r"https?://[^\s'\"<>\[\]]+", re.IGNORECASE)
SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _split_bracketed(raw: str) -> list[str] | None:
    """SpaPhish stores URLs as "[u1],[u2],...". Return the items, or None if not that format."""
    if not (raw.startswith("[") and raw.endswith("]")):
        return None
    inner = raw[1:-1]
    parts = inner.split("],[")
    return [p.strip() for p in parts]


def parse_urls(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "[]"}:
        return []
    items: list[str] = []
    if raw.startswith("[") and raw.endswith("]"):
        try:  # quoted Python or JSON list, as in the development interface
            v = ast.literal_eval(raw)
            if isinstance(v, (list, tuple)):
                items = [str(x) for x in v if x]
        except Exception:
            items = _split_bracketed(raw) or []
    candidates = items or URL_RE.findall(raw)
    out = []
    for x in candidates:
        x = x.strip().strip(",.'\"<>[]()")
        if SCHEME_RE.match(x):  # case-insensitive: SpaPhish contains "hTtps://"
            out.append(x)
    return out


# Same definitions as in spaphish_adapter_and_scoring.py.
def clean_domain(v: str) -> str:
    v = str(v or "").strip().lower().rstrip(".")
    if "@" in v:
        v = v.rsplit("@", 1)[-1]
    if ":" in v:
        v = v.split(":", 1)[0]
    return v


def subdomain_depth(domain: str) -> int:
    d = clean_domain(domain)
    if not d:
        return 0
    labels = [x for x in d.split(".") if x]
    return max(0, len(labels) - 2)


def url_features(value) -> dict:
    urls = parse_urls(value)
    lengths = [len(u) for u in urls]
    domains = []
    for u in urls:
        try:
            domains.append(urlparse(u).netloc)
        except Exception:
            pass
    depths = [subdomain_depth(d) for d in domains if d]
    return {
        "url_indicator": int(bool(urls)),
        "url_count_raw": len(urls),
        "url_count_parsed": len(urls),
        "url_length_max": float(max(lengths) if lengths else 0.0),
        "url_length_avg": float(np.mean(lengths) if lengths else 0.0),
        "url_subdom_max": float(max(depths) if depths else 0.0),
        "url_subdom_avg": float(np.mean(depths) if depths else 0.0),
    }
