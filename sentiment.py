"""Score news articles for sentiment against Strata's local model.

Everything here was measured against the running engine, and the three settings that
matter are not obvious:

1. `chat_template_kwargs: {"enable_thinking": False}` is NOT optional. Without it the
   model reasons before answering and a small `max_tokens` budget is spent entirely on
   the reasoning: `content` comes back as null with 32 tokens of `reasoning_content`.
   A pipeline that does not check would record every article as unclassifiable. This one
   refuses rather than records nothing, because a silent null is indistinguishable from
   a neutral article once it is in a table.

2. The system prompt is FIXED, and that is what makes repeated requests cheap. With a
   warm prompt cache an identical prefix is served in ~70 ms against ~2,300 ms cold. The
   cache is why the prompt is a module constant rather than a parameter: changing it
   invalidates every cached prefix.

3. Requests are SERIALISED. Measured: a concurrent request took 10,100 ms against 370 ms
   for the same call alone, and it evicted the cache for the calls that followed.

Run it directly to score the articles in a JSON file:

    python sentiment.py articles.json
    python sentiment.py articles.json --url http://127.0.0.1:8080

An articles file is a JSON list of strings, or of objects with a "text" key.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "strata"

# Fixed on purpose. See the module docstring: a changing prefix is a cache miss on every
# request, and this prompt is the whole cached head of every call.
SYSTEM_PROMPT = (
    "You are a financial news sentiment classifier. Given one news article, answer with "
    "exactly one line: LABEL|SCORE|CONFIDENCE where LABEL is positive, negative or neutral, "
    "SCORE is a number from -1.00 to 1.00, and CONFIDENCE is high, medium or low. "
    "Judge only the effect on the company's equity value. No explanation, no preamble."
)

LABELS = ("positive", "negative", "neutral")
CONFIDENCES = ("high", "medium", "low")

# Sentiment is a classification, not a composition: nine tokens was the measured reply
# length and sixteen leaves room without letting a runaway generation bill by the second.
MAX_TOKENS = 16


class SentimentError(RuntimeError):
    """The model answered, but not with something this pipeline can record."""


def parse_reply(text: str) -> dict:
    """Turn `LABEL|SCORE|CONFIDENCE` into a record, or raise.

    Deliberately strict. A parser that guesses turns a malformed answer into a plausible
    number, and a sentiment table full of plausible numbers is worse than an empty one
    because nobody can tell which rows were real.
    """
    if not text:
        raise SentimentError("empty reply")
    parts = [p.strip() for p in text.strip().split("|")]
    if len(parts) != 3:
        raise SentimentError(f"expected 3 fields, got {len(parts)}: {text!r}")

    label, score_text, confidence = parts
    label = label.lower()
    confidence = confidence.lower()
    if label not in LABELS:
        raise SentimentError(f"label {label!r} is not one of {LABELS}")
    if confidence not in CONFIDENCES:
        raise SentimentError(f"confidence {confidence!r} is not one of {CONFIDENCES}")
    try:
        score = float(score_text)
    except ValueError:
        raise SentimentError(f"score {score_text!r} is not a number") from None
    if not -1.0 <= score <= 1.0:
        raise SentimentError(f"score {score} is outside -1..1")
    return {"label": label, "score": score, "confidence": confidence}


def score(article: str, *, url: str = DEFAULT_URL, model: str = DEFAULT_MODEL,
          timeout: float = 300.0) -> dict:
    """Score one article. One request, no retry, no concurrency."""
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article},
        ],
        # See the docstring. Without this the reply has no content at all.
        "chat_template_kwargs": {"enable_thinking": False},
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise SentimentError(f"HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise SentimentError(f"could not reach {url}: {exc.reason}") from None
    elapsed_ms = (time.monotonic() - started) * 1000.0

    choices = payload.get("choices") or []
    if not choices:
        raise SentimentError(f"no choices in the response: {str(payload)[:200]}")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not content:
        # This is the failure the whole `enable_thinking` setting exists to prevent, so
        # name it rather than reporting a missing score.
        reasoning = message.get("reasoning_content") or ""
        raise SentimentError(
            "no content in the reply"
            + (f"; the model reasoned instead ({reasoning[:120]!r})" if reasoning else "")
        )

    usage = payload.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    record = parse_reply(content)
    record.update(
        reply=content.strip(),
        elapsed_ms=round(elapsed_ms, 1),
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
        cached_tokens=cached,
    )
    return record


def score_all(articles: list[str], **kwargs) -> list[dict | None]:
    """Score a list, one at a time. A failure is recorded as None, not raised.

    Serialised on purpose - see the module docstring. One bad article must not lose the
    rest of a batch, but a failure must also stay visible, so it is a None in place rather
    than a guess or a dropped row.
    """
    results: list[dict | None] = []
    for index, article in enumerate(articles, 1):
        try:
            result = score(article, **kwargs)
        except SentimentError as exc:
            print(f"  [{index}/{len(articles)}] FAILED: {exc}", file=sys.stderr)
            results.append(None)
            continue
        results.append(result)
        print(
            f"  [{index}/{len(articles)}] {result['label']:<8}{result['score']:+.2f} "
            f"{result['confidence']:<6} {result['elapsed_ms']:>7.0f} ms  "
            f"cached={result['cached_tokens']}  out={result['completion_tokens']}",
            file=sys.stderr,
        )
    return results


def load_articles(path: str) -> list[str]:
    # utf-8-sig, not utf-8: a file written by PowerShell's Set-Content -Encoding UTF8
    # carries a BOM, and json.loads rejects it with a message about column 1 that does
    # not obviously mean "this file has a byte order mark at the front of it".
    with open(path, encoding="utf-8-sig") as handle:
        data = json.loads(handle.read())
    if not isinstance(data, list):
        raise SystemExit(f"{path}: expected a JSON list")
    articles = []
    for entry in data:
        if isinstance(entry, str):
            articles.append(entry)
        elif isinstance(entry, dict) and isinstance(entry.get("text"), str):
            articles.append(entry["text"])
        else:
            raise SystemExit(f"{path}: each entry must be a string or an object with 'text'")
    return articles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("articles", help="JSON file: a list of strings or of {text: ...}")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)

    articles = load_articles(args.articles)
    results = score_all(articles, url=args.url, model=args.model)

    ok = [r for r in results if r is not None]
    if args.json:
        print(json.dumps(results, indent=2))
    if ok:
        times = [r["elapsed_ms"] for r in ok]
        cached = sum(1 for r in ok if r["cached_tokens"])
        print()
        print(f"  scored    {len(ok)} of {len(results)}")
        print(f"  median    {statistics.median(times):.0f} ms")
        print(f"  fastest   {min(times):.0f} ms")
        print(f"  slowest   {max(times):.0f} ms")
        print(f"  cached    {cached} of {len(ok)} carried a cached prefix")
    return 0 if len(ok) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
