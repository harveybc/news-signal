# Integration design: typed decisions within M5PHET

**M5PHET** is a typed machine-learning framework, not a research-program label.
Its [canonical use cases](https://github.com/harveybc/M5PHET/blob/master/docs/USE_CASES.md)
and [provider interfaces](https://github.com/harveybc/M5PHET/blob/master/docs/INTERFACES.md)
define structured task input/output across five task families. news-signal is
its first application and consumes the pinned classification contract at runtime.
Do not duplicate Laya's classifier: use its SDK directly where no broader
composition is needed, and adapt supported primitives where it is.

The next shared data input is the actual economic dataset: schedule and consensus
vintages, actual releases and revisions, with field-level availability. Numerical
values are source data, not inferred by Laya. Hierarchical market states and
calendar-aware forecasting/RL use those features through their own domain engines.
See the [calendar contract](https://github.com/harveybc/M5PHET/blob/master/docs/ECONOMIC_CALENDAR.md).
Geopolitical text remains a later source-specific extension. No new provider
runtime or financial results are claimed by this documentation revision.

Design-only summary of the owner's typed-ML proposal. Nothing below is claimed
as implemented Laya capability or demonstrated trading advantage.

Common boundary: `Task(State, Schema) -> TypedResult`, carrying event/receipt
time, feature order/units, missingness, model and source versions, uncertainty
and refusal. Different engines remain appropriate for different mathematical
problems: text classification, regression, representation/OOD, forecasting,
RL, causal estimation and optimization. A text classifier is not automatically
a causal estimator or a temporal numeric model.

The first concrete integration is news-signal. It can later supply a measured
feature branch to a frozen forecasting/RL interface or an evaluated final
decision head. Each placement is a separate ablation, with training/serving
parity, loss, sample support and cost declared. DOIN may search bounded validated
configurations; the model never assumes broker or governance authority.

Questions for upstream collaborators: supported financial adaptation recipes,
checkpoint/task calibration, label-order sensitivity, domain OOD and abstention,
embedding versus typed-output transfer, and CPU/GPU deployment benchmarks.
The project directory submission points to runnable software; a request to
collaborate on these broader ideas should be sent separately. No external
proposal has been submitted by this repository's authors through this work.
