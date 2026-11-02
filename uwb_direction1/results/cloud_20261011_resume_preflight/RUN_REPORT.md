# Resume-time cloud preflight

Observed: 2026-10-11 18:02:55 UTC, executor spec revision 633.

## Current executor

- Executor is running and connected.
- Exposed capabilities: none; runtime variables, secrets, and outbound identities are absent.
- HTTP policy is enforced/restricted with no task-configured egress hosts; shell network access is not an available data-ingress route.
- Python 3.12.14; PyTorch 2.14.0+cpu; CUDA unavailable; 5 CPU cores.
- `/workspace` is 32 GB total with 712 MB free.
- Restored X4 data exists at `/workspace/uwb-restored-20260930`.
- Process inspection found no active Python/training process; existing completed result directories remain untouched.

## Decision

The current environment is the existing restore, so no Drive recovery or archive fetch is needed. No candidate method has passed both novelty and validation gates. With no GPU, only 712 MB free, restricted shell egress, no acquisition hardware, and processed range-time data without raw complex echoes or calibrated motion truth, a new external-cohort run or phase/kinematics study cannot be started reproducibly here.

No arrays, labels, or checkpoint contents were read. No data download, fit, training, or checkpoint/source mutation occurred. This is an executor/feasibility preflight, not an experiment or contribution result.
