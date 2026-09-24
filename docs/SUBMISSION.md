# Laya directory submission draft

Form inspected 2026-09-24 UTC: https://laya-ai.com/submit?type=integration
This is an independent community directory with review, not an upstream
development contract. No submission has been sent.

| Field | Value |
|---|---|
| Type | Integration |
| Project / source URL | https://github.com/harveybc/news-signal |
| Name (optional) | Harvey Bastidas |
| Email (optional) | Leave blank, or enter the owner's chosen contact address |
| Website | Leave blank. This input is hidden, aria-hidden and tabindex=-1; it is a honeypot, not the project URL. |

## Description

news-signal is an open-source prototype integrating local Laya with an auditable
news-to-features pipeline for trading research. It classifies short English news
for an explicitly supplied asset into relevance, event type and financial tone.
The Python CLI validates timestamps and typed responses, hashes inputs/questions/
checkpoint files, and emits shadow-only receipts. It pins the Laya SDK revision,
rejects device fallback and oversized tokenized inputs, and never sends orders.

The repository includes a runnable offline fixture, tests, a local-checkpoint SDK
adapter, and an explicit evaluation and integration plan. Real-weight financial
accuracy, calibration and latency have not yet been measured; tests currently
exercise a disclosed fixture and SDK-shaped test double, not a trading model.
Next work connects licensed prospective news and the existing governed execution
stack to MT5 demo and Alpaca paper, comparing news-only, price-only and combined
policies. Those broker connections are planned, not claimed as operational here.
We welcome review of the integration and discussion of domain adaptation. No
profitability or real-money readiness is claimed.
