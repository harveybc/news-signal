# NEWS-LIVE: typed news features, not a trading authority

## Decision and scope

Create a separately installable `news_signal` package. `trading-signal` is a
retired label generator; predictor trains offline numeric models. Neither is the
owner of streaming news interpretation. Existing trading repositories retain
broker routing and risk ownership. No second broker implementation here.

Actors: collector, typed model, deterministic policy, execution adapter, auditor.
Input: one licensed news revision, one externally resolved asset, publication
and actual first-receipt timestamps. Output: a content-addressed SHADOW record.
No funds, broker credentials, order methods, capital allocation or return forecast.
Model text is untrusted data, never shell instructions or tool calls.

## Requirements and tests declared before implementation

| ID | Acceptance | Unit/integration/system evidence |
|---|---|---|
| N01 | Strict news schema; UTC-aware times; reject future/unreceived/stale news before inference | `test_news_validation`, spy backend never called |
| N02 | English-only initial task, explicit asset, no inferred trading symbol | input and unsupported-language tests |
| N03 | Complete typed choices, finite probabilities, normalized distribution, matching argmax | malformed SDK response tests |
| N04 | Never authorize an order; probabilities explicitly UNCALIBRATED | receipt and CLI tests, including hostile news text |
| N05 | Link full input, question schema, model bytes and response by hashes | changed-input and checkpoint mutation tests |
| N06 | Real Laya SDK adapter is distinct from fixture backend | SDK-injection integration test; fixture marked NON_MODEL_FIXTURE |
| N07 | Reject implicit device fallback and overlong tokenized state | fake-agent negative tests, real weight smoke pending |
| N08 | CLI prints parseable evidence and nonzero status on refusal | subprocess smoke and stale example |
| RP148-1 | Every collected item carries our receipt clock, stamped here and unsupplied by the source; a source that declares no publication clock is recorded as declaring none and never given the receipt clock in its place | `test_rp148_collection_boundary.py` receipt-clock section; `test_cl18_collector.py::test_an_item_may_not_supply_the_moment_we_received_it` |
| RP148-2 | The same item twice is a duplicate; changed text is a revision bound to the first and rewrites nothing; `known_at(T)` never includes an item received after T | `test_rp148_collection_boundary.py` dedup/`known_at` section, swept over five values of T; `test_cl20_queue_integrity.py` for cross-source identity |
| RP148-3 | An item accepted for processing survives a restart, an acknowledgement is bound to the persisted record, and an interrupted collection or drain replays without double-counting | `test_cl18_collector.py::test_the_queue_survives_a_new_process`, `test_cl23_pipeline.py::test_a_crash_between_persisting_and_acknowledging_is_recovered_without_asking_again`, `test_rp148_collection_boundary.py::test_a_collection_interrupted_mid_batch_replays_without_double_counting` |
| RP148-4 | Staleness is a declared policy: an item past the declared maximum age, and an item whose age cannot be measured, are refused by name; a source handing us nothing is `AWAITED`, distinguishing `SILENT_SINCE` from `NEVER_PRODUCED`, and no empty batch or neutral reading is invented | `test_rp148_collection_boundary.py` staleness and silence sections, including the CLI in a new process |

The collection boundary (RP148) needs no model weights and no feed entitlement, and
was built without either: `collector` stamps the receipt clock, records a missing
publication clock as missing, decides duplicate against revision, answers
`known_at(T)`, enforces a declared maximum age and reports `AWAITED` for a silent
source. What an entitlement adds is the arrival of real bytes; licensed retention,
authority over a live wire and data-gov registration of collected inputs remain
unimplemented and unclaimed.

Architecture: `core` validates and builds receipts; `backends` wraps the pinned
SDK or a disclosed fixture; `cli` selects the backend and serializes JSON.
Checkpoint manifests inventory all regular model files and reject added/changed
files. They prove identity, NOT authenticity, calibration or governance acceptance.
SDK source is pinned separately. Receipts do not claim data-gov registration.

System tests use synthetic text and a fake SDK with the exact `system_one` response
shape. They are not financial accuracy tests. Alpha with real frozen weights,
licensed prospective news, accepted governance receipts, hardware resource
measurements, and both demo brokers is assigned to Satoshi as independent work.
No historical benchmark or profit claim follows from this software release.

## Next experiment, separate from software tests

Freeze checkpoint/task/languages, news source rights and receipt-time capture.
Human-labelled disjoint train/calibration/test news: relevance/event/tone macro-F1,
per-class precision/recall, confusion, Brier/NLL/ECE with sample sizes and a
majority/rule baseline; FinBERT only under a matched task. Tone is not a return.
Prospective price-only/news-only/combined policies share eligible decisions and
costs. Report executed paper fills separately from simulated/shadow outcomes.
No future news revisions, publication-time backdating, or training on evaluation.
Predeclare horizon, cost, chronological blocks and release schedule. Inspect
duplicates, calibration, latency and failure modes before tuning model capacity.

MT5 demo and Alpaca paper are distinct acceptance targets. Real-capital release
is not part of this version. Broader typed-ML/RL/causal RFC is design-only and must
not be advertised as a capability measured by this adapter.
