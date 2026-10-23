# News sentiment against Strata

Scores news articles for effect on equity value, one article at a time, against the
local model. Everything in here was measured against the running engine rather than
assumed, and three of the findings are not obvious from the documentation.

## Use

```
python sentiment.py articles.json
```

`articles.json` is a JSON list — of strings, or of objects with a `text` key. A file
written by PowerShell's `Set-Content -Encoding UTF8` carries a BOM; the loader reads
`utf-8-sig` so that is not a failure.

Each result is `{"label", "score", "confidence", "reply", "elapsed_ms", ...}`, or `null`
for an article that could not be scored. **A null stays visible rather than being dropped
or guessed**, because a missing row and a neutral article are not the same thing.

## The output is `LABEL|SCORE|CONFIDENCE`

```
negative|-0.85|high
positive|+0.85|high
neutral|+0.00|high
```

The parser is deliberately strict — three fields, a known label, a known confidence, a
score in -1..1 — and raises rather than guessing. A sentiment table full of plausible
numbers is worse than an empty one, because nobody can tell which rows were real.

## Three settings that are not optional

### `chat_template_kwargs: {"enable_thinking": false}`

Without it the model **reasons before answering**, and a small `max_tokens` budget is
consumed entirely by the reasoning — `content` comes back **null** with 32 tokens of
`reasoning_content`. Measured across four variants:

| request | time | completion | content |
|---|---|---|---|
| plain, `max_tokens: 32` | 502 ms | 32 | **null** |
| `enable_thinking: false` | 2392 ms | 9 | `negative\|-0.85\|high` |
| `/no_think` in the prompt | 2687 ms | 32 | **null** |
| `reasoning_budget: 0` | 2617 ms | 32 | **null** |

Only the first works. A pipeline that did not check would record every article as
unclassifiable, and the null would look like a data problem rather than a request one.

### A fixed system prompt

The prompt is a module constant because a warm prefix cache serves an identical prompt
in **~70 ms** against **~2,300 ms** cold — measured, a 34x difference. Changing the
prompt invalidates every cached prefix, so it is not a parameter.

### Serialised requests

Measured: a concurrent request took **10,100 ms** against 370 ms for the same call alone,
and it **evicted the cache** for the calls that followed. The engine serves one at a
time; a pipeline that fans out is slower than one that queues.

## The latency problem, and the config that fixes it

The running engine's config has two settings that make short requests expensive:

| setting | stock | why it costs |
|---|---|---|
| `--prefill auto` | 8192 | "the largest chunk up to 8192" — a 121-token article does 8,192 tokens of work |
| `--prompt-cache-every` | 16384 | a prompt under 16k never checkpoints, so `0 reused` on every request |

Measured consequence:

```
prompt  121 tokens = 0 reused +  121 read in 2064 ms (58.6 tok/s)
prompt 7440 tokens = 0 reused + 7440 read in 10461 ms (711.2 tok/s)
```

**121 tokens costs 2,064 ms; 7,440 costs 10,461 ms.** There is roughly **2,000 ms of
fixed cost per request** that does not scale with the article, and for a sentiment
pipeline every article pays it.

`strata-sentiment.json` in the Strata folder sets:

```
--prefill 256              instead of 8192
--prompt-cache-every 128   instead of 16384, so a shared prefix checkpoints
--max-context 32768        instead of 131072 - articles are short, and the 131k held VRAM
                           that the engine itself warned about
--vram-reserve-mib 976     what the engine asked for: "236 MiB of VRAM free - LOW"
--vision removed           a text classifier does not need a second engine and a 900 MB
                           projector
```

**This has not been measured yet** — it needs a restart, and the engine was in use. Run
`sentiment.py articles-example.json` after restarting on that config and compare the
median against 2,343 ms.

## What is not done

- The config's effect is unmeasured; the numbers above are from the stock config.
- No retry. A transport failure is a `null` in the output and a line on stderr.
- No batching beyond serial iteration. The engine takes one prompt at a time.
- No article fetching. This scores text you already have.
