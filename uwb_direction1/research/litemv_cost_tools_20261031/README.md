# Guarded synthetic cost tools; no actual policy or timing selected yet

This is executable preparation while OS-CNN's full audit is active. The four
tests in TESTS.log pass pure cost arithmetic, ordered budget selection,
memory/no-feasible-budget closure and refusal before policy reads/output
creation when prior audit closure is missing. These tests are not a formal
LITEMV resource profile or accuracy experiment.

profile_cost.py requires an explicit future policy, output path and previous
experiment path. Prior full audit/termination/private complete-backup/gate
checks run before reading the new policy. No actual POLICY.json is supplied
in this preparation directory. No real epoch, batch, seed, preprocessing or
resource decision has been selected by these files. Test values are synthetic
arithmetic fixtures only. A prior gate pass requires source confirmation and
cannot trigger replacement by LITEMV.

Once prior closure permits a next method, the caller must prospectively supply
one policy specifying an ordered epoch-budget list, one batch/seed/input
adaptation, CPU and memory bounds. The policy bytes/hash and utility hash are
written before timing. Only then does the tool import the verified full
author LITEMV/native Keras engine. All profiled inputs/labels are generated
synthetic fixtures, not UWB. Eight native training batches and four validation
batches include first-call overhead. Three original `.keras` saves and three
full model/Adam/BN/callback/RNG snapshot saves measure serialization cost.

The estimate uses measured maxima, all4700 training records, all1000 validation
and1200 final diagnostic records, conservative per-epoch checkpoint/snapshot
counts, a factor2 and120s fixed allowance. Memory must also pass. It picks only
from the prospective ordered epoch budgets; it never changes batch, width,
seed or filters to rescue a failure. Source architecture remains full
621680 trainable/889624 total parameters. An infeasible policy closes for
resources and does not measure accuracy or disprove an unrun original budget.

Timing itself has an internal590/600s CPU limit and610s alarm. A future actual
invocation must additionally use the existing independent bounded supervisor
to retain exit/timeout/CPU receipts. Produced checkpoint fixtures remain local
and out of Git. This is not a real UWB training launcher; a selected feasible
policy still requires the actual runner/auditor, fixed source-only protocol,
remote preregistration and exact readback before UWB values/labels/fitting.
Users7–8 remain excluded and all8 historical development stays exploratory.

The source/engine dependencies and GPL-3.0-only license are retained in the
existing LITEMV preparation/engine directories. No source or global runtime
was modified here. Reproduce the non-timing checks:

```bash
python /workspace/uwb-litemv-cost-tools-20261031/test_cost_tools.py
```
