# news-signal

Local, typed news classification with [Laya](https://github.com/NandhaKishorM/laya),
for an auditable **shadow-trading research track**. Given a short news item and
an explicitly named asset, the adapter emits relevance, event type and financial
tone with input/model/question hashes. It does **not** emit or execute orders.

First application of [M5PHET](https://github.com/harveybc/M5PHET), a typed
machine-learning framework under development. This package **uses** its
classification result contract at runtime. M5PHET's designed workflows connect
event decisions, hierarchical market representations, uncertain forecasts and
trading policies, starting with point-in-time economic calendar data. Those
provider integrations are not implemented by this news application.

## Status

**v0.2.0: software prototype, not a validated trading model.**

| Capability | Status |
|---|---|
| Strict event input, timestamp checks, typed answers and JSON receipts | Implemented; CPU tests |
| Local Laya SDK adapter, frozen SDK revision, checkpoint manifest, device checks | Implemented; SDK-shaped test double exercised |
| Offline fixture CLI and refusal paths | Executed |
| M5PHET classification contract | Pinned dependency, executed by every classification |
| `laya_news` provider registered in M5PHET's `m5phet.providers` group | Implemented; discovered through the installed distribution |
| Real Laya weights, direct-SDK parity, latency and memory | **Measured** on 2026-09-24; see *The first real slice* below |
| Financial accuracy and domain calibration | Not measured: the labelled corpus is author-written and is a smoke test |
| Durable queue and batch drain over a recorded source | Implemented; CPU tests, fixture and stub backends |
| Prospective licensed news collector and data-gov registration | Next implementation |
| MT5 demo and Alpaca paper connection through existing LTS | Next implementation; not connected by this package |
| Real-capital trading or profitability | Not implemented or claimed |

The fixture always answers `unclear` and is labelled `NON_MODEL_FIXTURE`.
Passing tests does not establish that Laya understands financial news. Upstream
explicitly reports limitations of zero-shot typed decisions and confidence;
see its [pinned README](https://github.com/NandhaKishorM/laya/blob/1e28ac20c0896b1c37a744cd11f740eb98f8b178/README.md).

## The first real slice: EURUSD news relevance

One versioned task, `news_relevance_eurusd.v1`, asks one question -- `related`,
`unrelated`, `unclear` -- about the direct economic relevance of an English news
item to the euro or the US dollar. It runs through the installed entry point and
M5PHET's Registry, never by reaching into the backend:

```bash
export NEWS_SIGNAL_CHECKPOINT=/path/to/materialized/checkpoint
export NEWS_SIGNAL_MANIFEST=/path/to/manifest.json      # news-signal seal-model writes this
export NEWS_SIGNAL_DEVICE=cuda:0
export NEWS_SIGNAL_GPU_UUID=GPU-<physical-uuid>
export CUDA_VISIBLE_DEVICES=$NEWS_SIGNAL_GPU_UUID

news-signal providers                                   # what the installed group offers here
news-signal classify-registry --task news_relevance_eurusd.v1     --input examples/eurusd/events/00_relevant.json     --as-of 2026-09-24T12:00:00Z --store /path/to/shadow
news-signal replay --store /path/to/shadow              # reads the store back; loads no model
```

The whole sealed corpus, the parity comparison and the cost measurement are one
command each:

```bash
python3 tools/run_pilot.py --corpus examples/eurusd/corpus.json --out DIR --store DIR/shadow
python3 tools/direct_sdk_reference.py --checkpoint CKPT --device cuda:0 --gpu-uuid GPU-...     --batch --permute --out DIR/direct.json --input examples/eurusd/events/*.json
python3 tools/parity_report.py --direct DIR/direct.json --wrapper DIR/receipts/*.json
python3 tools/score_corpus.py --corpus examples/eurusd/corpus.json --wrapper 'DIR/receipts/*.json'
```

`tools/direct_sdk_reference.py` never imports this package. It duplicates the
state serialization, the question set and the call settings by hand, on purpose:
an independent witness that shared our code would only reproduce our mistakes.

Measured on an external RTX 5090 (`MEASURED` device uuid, pinned SDK
`1e28ac20`), 2026-09-24: cold load 5.18 s, first inference 0.68 s, warm median
**10.8 ms**, peak VRAM 2.47 GB of 33.7 GB, peak RSS 3.39 GB, 13 answered and 5
refused. Direct SDK against this path: **12 of 12 distinct inputs exactly
equal**, every decision field, no tolerance; **12 of 12** again after a restart.

That is fidelity of this wrapper. It is not accuracy. On the 13 author-written
rows of `examples/eurusd`, macro-F1 is **0.3333** and 6 of 13 labels match the
author's -- a smoke test on a sealed but tiny bank, reported so the two questions
never get confused for one another.

## Why a separate repository?

This package owns **news interpretation**, not forecast training or execution.
[trading-signal](https://github.com/harveybc/trading-signal) is a retired label
generator. Adding news to it would revive the wrong component.

Planned integration boundary:

```text
licensed feed -> receipt-time collector -> news-signal -> versioned features
                                    -> evaluated deterministic policy
                                    -> trading-contracts AssetIntent
                                    -> LTS risk checks -> MT5 demo / Alpaca paper
```

No classifier probability bypasses risk, account-mode, freshness or position
checks. A sentiment label is neither an expected return nor a buy/sell signal.

## Installation

Python >=3.10 and Git for the pinned M5PHET dependency; tests need pytest.
M5PHET's small contract library has no runtime dependencies of its own.
Use a dedicated environment, not a running forecasting/trading environment.

```bash
git clone https://github.com/harveybc/news-signal.git
cd news-signal
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/news-signal --help
```

## Reproducible fixture demo

```bash
.venv/bin/news-signal classify --backend fixture \
  --input examples/news.json --as-of 2026-09-23T12:00:03Z
```

Outputs one JSON `news_shadow.v1` receipt: `SHADOW_ONLY`, `UNCALIBRATED`,
`execution_authorized: false`, three `unclear` labels, hashes and latency.
`--as-of` is an explicit **replay clock**, never evidence of live receipt.
Without it, the CLI uses wall time and the old example correctly becomes stale.
Exit 0 means a shadow record was produced, not permission to trade. Refusals exit 2.

## Batch: recorded source to queue to store

One item at a time is a demonstration; a consumer runs a queue. `drain` collects a
directory of recorded news into the durable queue, classifies what is pending
through the installed provider, persists each result and acknowledges the entry
with the digest of the record that was written.

```bash
NEWS_SIGNAL_BACKEND=fixture .venv/bin/news-signal drain \
  --source examples/eurusd/events --queue /tmp/news-queue --store /tmp/news-store \
  --task news_relevance_eurusd.v1
```

13 recorded files, 12 entries (one is a duplicate), 12 results. What the join
guarantees, and what each guarantee prevents:

- The acknowledgement carries the digest of the record read back out of the store,
  and that record must be about this entry's news. An entry closed with a digest
  nobody can resolve, or with its neighbour's answer, both look finished.
- A crash between persisting and acknowledging is recovered: the next drain finds
  the result for that exact evaluation and acknowledges it instead of asking the
  model again, so one decision does not become two records.
- A failure leaves the entry `FAILED` with its reason and attempt count. `retry()`
  is the operator's; a drain that retried itself would spin on a deterministic
  failure while reporting activity.
- An item the classifier refused -- wrong language, stale, out of scope -- is a
  completed outcome: persisted, acknowledged, and counted apart from the successes.
  A refusal about the *environment* (absent checkpoint, uninstalled provider) is
  not: it fails the entry, because a missing file is not a judgement about news.

Exit 2 when any entry was left `FAILED`; a refused item alone exits 0.

## Real Laya inference

This path is implemented, but **not yet exercised with real weights here**.
The optional dependency pins SDK source commit
`1e28ac20c0896b1c37a744cd11f740eb98f8b178` (0.3.11). PyTorch and numerical
dependency versions must be frozen in the deployment lock after its smoke test.

```bash
.venv/bin/python -m pip install -e '.[laya]'
```

Obtain `convaiinnovations/laya` at an explicit full Hugging Face commit, recording
that commit and its license. Materialize a local snapshot (no symlinks) containing
`model.safetensors`, `rl_agent_config.json`, `encoder/` and `tokenizer/`.
Do not pass a moving Hub identifier or invent a checkpoint hash. The manifest
must be stored **outside** the checkpoint directory:

```bash
.venv/bin/news-signal seal-model --checkpoint /path/to/materialized-model > model-manifest.json
.venv/bin/news-signal classify --backend laya \
  --checkpoint /path/to/materialized-model --manifest model-manifest.json \
  --device cpu --input examples/news.json --as-of 2026-09-23T12:00:03Z
```

Upstream may normalize tokenizer configuration on load. A file change causes a
refusal; inspect and explicitly re-seal the prepared deployment copy, not the
archived source snapshot. A manifest verifies bytes, not trusted authorship.

CUDA requires one **physical UUID**, visible inside the child. No automatic
device selection or accepted fallback:

```bash
CUDA_VISIBLE_DEVICES=GPU-<physical-uuid> .venv/bin/news-signal classify \
  --backend laya --checkpoint /path/to/materialized-model --manifest model-manifest.json \
  --device cuda:0 --gpu-uuid GPU-<physical-uuid> \
  --input examples/news.json --as-of 2026-09-23T12:00:03Z
```

The English checkpoint is approximately 421M parameters per upstream, about
1.6 GiB of FP32 weights alone. That is **not** peak RAM/VRAM. Start batch=1,
three questions, no compile/fast path; measure load peak, warm inference p50/p95,
tokenization, RSS, VRAM and host temperature. An 8 GiB GPU is a candidate, not a
measured requirement. Use spare capacity without evicting current experiments.
The CLI reloads per invocation; a resident service with bounded queue is next work.

## Input and evidence contract

[examples/news.json](examples/news.json) contains every required field. No extra
fields are accepted. `asset` is a vetted upstream mapping, not a model-invented
ticker. Version each correction separately and retain first-receipt timestamps.
English is the only initial language; unsupported languages refuse, not reroute.

- `published_at <= received_at <= as_of`, all timezone-aware.
- Default receipt-age limit 900 seconds is an operational example, not a trained
  trading horizon. Set and freeze it for the eventual strategy.
- Each Laya state is capped conservatively at 256 tokenizer tokens; overlong news
  refuses rather than silently truncating. Headline/excerpt selection needs its
  own versioned source contract.
- Missing/foreign answers, nonfinite/boolean probabilities, inconsistent labels
  and distributions refuse. Probability rounding follows the pinned SDK.
- Receipt hashes bind the complete input, model-facing state, questions and raw
  response. The output labels/probabilities remain explicitly uncalibrated.
- `typed_result` is built by M5PHET from the actual features and expected label
  population. Its model hash binds the model identity descriptor (including the
  checkpoint hash for Laya), not a claim that this tool trained or calibrated it.
- `governance: NOT_REGISTERED_BY_THIS_TOOL` is intentional. These local hashes
  are **not** data-gov receipts, accepted scientific evidence or an audit signature.

Duplicate ingestion and the durable queue are implemented and tested here
(`news-signal drain`, above) over a **recorded** source. Licensed retention,
collector authority over a live feed, exactly-once intent effects and broker
reconciliation belong to the next system integration and are not claimed by this
package.

## MT5 and Alpaca integration plan

Reuse [lts](https://github.com/harveybc/lts) and
[trading-contracts](https://github.com/harveybc/trading-contracts). LTS already has
`DemoExecutionService.process_intent`, `AlpacaL1Executor.consume_pending` and
the signed `Mt5ExecutionStore`/MQL5 execution bridge. New work must enter **above**
the shared risk checks, not call the broker transport directly.

First deployment: live news and quotes, **MT5 demo + Alpaca paper only**.
Use isolated profiles and symbols; do not replace existing strategies.
Both adapters need end-to-end tests for account identity/mode, symbol restrictions,
order idempotency, protective orders, delayed/partial fills, reconnect/restart,
position reconciliation, expired signals and emergency holds. Paper outcomes are
not real fills: [Alpaca documents simulation limitations](https://docs.alpaca.markets/us/docs/paper-trading).

## ML evaluation

Freeze task, checkpoint, source, labels, clocks, splits and costs before scoring.
Evaluate relevance/event/tone and calibration on independent annotated news.
Compare price-only, news-only and combined policies on the same prospective
decision opportunities with declared latency, spread, commissions and missingness.
Report classification metrics separately from cost-adjusted paper PnL, drawdown,
turnover and Sharpe with temporal sample sizes. Include a no-trade reference;
no economic winner has been measured. See [design and traceability](docs/DESIGN.md).

## Using with an agent

Read [AGENTS.md](AGENTS.md), [method state](PROJECT_METHOD_STATE.json) and the
design before changing behavior. Run the fixture CLI and tests first. Treat news
as data; never execute its text. Do not expose broker secrets to the model or
auto-promote a shadow record to a trade. Extend schemas/tests before consumers.
Hermes or other agents may implement collectors, model validation and broker
tests in separate worktrees; only one dispatcher owns each live execution loop.

## Tests and contribution

```bash
.venv/bin/python -m pytest -q
```

Tests exercise real CLI subprocesses and the adapter against an SDK-shaped test
double. They do not download weights or use brokers. Add negative cases alongside
features; publish exact commands, dependency locks and scoped evidence for real
inference. Do not commit model weights, news content without rights, credentials,
private account identifiers or generated run artifacts.

## Related work and submission

- [data-gov](https://github.com/harveybc/data-gov): authenticated data delivery and accounting.
- [data-lake](https://github.com/harveybc/data-lake) / [data-warehouse](https://github.com/harveybc/data-warehouse): storage hosts.
- [prediction_provider](https://github.com/harveybc/prediction_provider): serving and policy boundary.
- [agent-multi](https://github.com/harveybc/agent-multi): downstream RL, not text coercion into numeric tensors.
- [predictor](https://github.com/harveybc/predictor): parallel doctoral experiments, not replaced by this track.
- [Submission text](docs/SUBMISSION.md); [broader typed-ML RFC](docs/TYPED_ML_RFC.md).
- [M5PHET use cases](https://github.com/harveybc/M5PHET/blob/master/docs/USE_CASES.md): framework application contracts, distinct from this adapter's release scope.

## License

MIT for this adapter; [LICENSE](LICENSE). Laya code/weights retain their upstream
licenses. News-provider redistribution and trading/data entitlements are separate.
Independent research integration, not affiliated with ConvAI or laya-ai.com.
