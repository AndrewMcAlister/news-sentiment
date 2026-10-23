"""NewsAPI query building and fetching.

The first attempt used `(Microsoft OR MSFT) AND (stock OR shares ...)` searched across title
and description, and half the results were off-topic: an AWS migration announcement, a
Veeam release note, an Apple home-device piece, a Gary Marcus essay on the AI bubble. All of
them mention Microsoft in the body, none of them is news about Microsoft's equity.

Two rules fix that:

1. SEARCH THE TITLE ONLY. A story about a company's shares names the company in its
   headline. A body match catches every passing mention, and passing mentions are most of
   what a news index contains.

2. REQUIRE BOTH HALVES IN THE QUERY - the company AND a market term - rather than relying on
   the search engine's relevance ranking. NewsAPI's `q` supports AND/OR/parentheses and
   quoted phrases, so the constraint can be expressed rather than hoped for.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

# Words that mark an article as being about the equity rather than the company's products.
# "cloud" was in the first attempt and pulled in every AWS and Azure infrastructure story.
MARKET_TERMS = (
    "stock", "stocks", "shares", "share price", "earnings", "revenue", "guidance",
    "analyst", "analysts", "price target", "upgrade", "downgrade", "valuation",
    "dividend", "buyback", "market cap", "quarterly", "results", "forecast",
)

# Sources that publish package releases, gadget reviews and link blogs. Dropping them by
# source is cheaper and more reliable than trying to describe them in the query.
NOISE_SOURCES = (
    "pypi.org", "geeksaresexy", "tom's hardware", "pluralistic.net", "biztoc",
    "github.com", "medium.com", "dev.to",
)


def query_for(company: str, ticker: str | None = None) -> str:
    """A tight NewsAPI `q` for one listed company.

    Both halves are required: the company name (or its ticker) AND a market term. Searched
    in the title only, by `fetch` below, because a headline names what the story is about.
    """
    names = [f'"{company}"']
    if ticker:
        names.append(ticker.upper())
    entity = "(" + " OR ".join(names) + ")"
    market = "(" + " OR ".join(f'"{t}"' for t in MARKET_TERMS) + ")"
    return f"{entity} AND {market}"


def fetch(company: str, ticker: str | None = None, *, api_key: str,
          page_size: int = 50, days: int = 7) -> list[dict]:
    """Return articles whose TITLE matches the tight query, noise sources removed."""
    from datetime import datetime, timedelta, timezone

    params = {
        "q": query_for(company, ticker),
        "searchIn": "title",          # rule 1
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": str(min(page_size, 100)),
        "from": (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d"),
        "apiKey": api_key,
    }
    url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "news-sentiment/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}")

    out = []
    for article in data.get("articles") or []:
        source = ((article.get("source") or {}).get("name") or "").lower()
        if any(noise in source for noise in NOISE_SOURCES):
            continue
        title = (article.get("title") or "").strip()
        if not title:
            continue
        out.append({
            "title": title,
            "description": (article.get("description") or "").strip(),
            "source": (article.get("source") or {}).get("name"),
            "published": article.get("publishedAt"),
            "url": article.get("url"),
        })
    return out
