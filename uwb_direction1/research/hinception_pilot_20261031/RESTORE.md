# Scope of model preservation

The initial/first insurance is not the final model set. After a completed
audited run, the complete bundle contains all protocol-produced checkpoints,
original selected/last native Keras models, exact IDs/partitions/logs and audit.
Reconstruct dependency text paths by enumerating PROTOCOL.dependencies_text:
payload/dependencies/text/<index>_<basename> maps to that original absolute
cloud path. Vendor/helper/runtime manifests preserve their own directory keys.
Restore run01 and study files under this protocol's workspace. Dependency
wheels/base Python/NumPy/Torch are separately pinned; recover the isolated
Keras runtime using its prior restore helper and requirement hashes. Raw data
and RMS maps use the existing private migration indices and immutable hashes;
do not request user uploads or place binary inputs in Git.

Use make/restore helpers with the matching identity/settings for tested native
state recovery. Epoch-boundary continuation is tested, but no automatic full
job or mid-epoch launcher exists. The finalprediction state is an inference
audit state (best weights and last optimizer), not the best optimization
trajectory. For that trajectory use best.pt with its own full native state.
Never duplicate an active process or silently retry a scientific configuration.
