"""MSFT sentiment from NewsAPI headlines, scored by Strata.

Two fixes over the first probe:
  * The query was 'MSFT OR "Microsoft"', which matches any page mentioning the word -
    PyPI packages, a flight-sim review, a spyware piece. Narrowed to financial terms.
  * stdout is forced to UTF-8; the default cp1252 crashed on the first curly quote.
"""
import json, os, sys, urllib.error, urllib.parse, urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, r"D:\news-sentiment")
import sentiment

KEY = os.environ.get("NEWSAPI_KEY", "")
if not KEY:
    sys.exit("NEWSAPI_KEY not set")

# Financial terms only, searched in the title: an article about Microsoft's stock names it
# in the headline. A body match catches every passing mention instead.
QUERY = '(Microsoft OR MSFT) AND (stock OR shares OR earnings OR revenue OR analyst OR cloud)'

params = {"q": QUERY, "searchIn": "title,description", "language": "en",
          "sortBy": "publishedAt", "pageSize": "20", "apiKey": KEY}
url = "https://newsapi.org/v2/everything?" + urllib.parse.urlencode(params)
req = urllib.request.Request(url, headers={"User-Agent": "news-sentiment/0.1"})
with urllib.request.urlopen(req, timeout=30) as r:
    data = json.loads(r.read().decode("utf-8"))

arts = data.get("articles") or []
print(f"  query       {QUERY}")
print(f"  returned    {len(arts)} articles of {data.get('totalResults')}")
print()

# Skip the obvious non-financial noise rather than scoring a package release.
SKIP = ("pypi.org", "geeksaresexy", "tom's hardware", "pluralistic")
texts = []
for a in arts:
    src = ((a.get("source") or {}).get("name") or "").lower()
    if any(s in src for s in SKIP):
        continue
    title = (a.get("title") or "").strip()
    desc = (a.get("description") or "").strip()
    if not title:
        continue
    texts.append((title, f"{title}. {desc}".strip()))

print(f"  scoring     {len(texts)} after dropping non-financial sources")
print()
results = []
for i, (title, text) in enumerate(texts, 1):
    try:
        r = sentiment.score(text)
    except Exception as e:
        print(f"  [{i}] FAILED {type(e).__name__}: {e}")
        continue
    results.append(r["score"])
    print(f"  [{i}] {r['label']:<8}{r['score']:+.2f} {r['confidence']:<6} {r['elapsed_ms']:>6.0f} ms")
    print(f"      {title[:110]}")

print()
if results:
    import statistics
    print(f"  scored      {len(results)}")
    print(f"  mean score  {statistics.mean(results):+.2f}")
    print(f"  median      {statistics.median(results):+.2f}")
    print(f"  range       {min(results):+.2f} .. {max(results):+.2f}")
    neg = sum(1 for s in results if s < 0)
    pos = sum(1 for s in results if s > 0)
    print(f"  negative {neg}  positive {pos}  neutral {len(results) - neg - pos}")
