# Proposal: typed decisions as one engine in a broader ML stack

The owner has named this broader program **M5PHET**. Its canonical proposal and
contracts now live at https://github.com/harveybc/M5PHET. news-signal is its first
application and consumes the pinned classification contract at runtime. The
five operational fronts are classification, regression/forecasting,
representation/unsupervised, RL and causal; optimization is cross-cutting.
The original proposal text below remains context, not a claim of five engines.

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
