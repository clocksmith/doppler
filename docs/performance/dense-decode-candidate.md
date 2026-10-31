# Dense decode specialization screening

The dense-decode candidate tool builds an explicit inline kernel path. It does
not change runtime defaults or promote a kernel. Selection policy lives in the
canonical matmul rule map; the existing rule-bundle generator supplies its mirror.

Input contains kernelPath, bindings and capabilities. Each binding names one
declared decode operation and its section (layer or postLayer), actual
materialized weight/activation/output dtypes, transpose layout and positive
integer M/N/K. Capabilities explicitly include hasF16, hasSubgroups and
hasSubgroupId. Capture these from the executing device rather than inferring them
from a vendor or model filename.

For M=1 row-major projections, the opt-in policy offers existing full-F32 subgroup
GEMV or F16-weight/F32-activation subgroup GEMV. It retains the declared program
when shape or capabilities do not qualify, recording a reason. Storage/output or
explicit precision contradictions fail before rewriting. Prefill steps stay
unchanged, including the prefill vocabulary head. Output records program
identity, canonical digest and registry constants and remains unqualified.

Graphs with layer overrides must first be expanded into independently
materialized per-layer plans. The builder refuses them instead of applying a
shared dtype assumption to mixed-format layers. Missing, duplicate or unknown
projection bindings also fail. These are intake limits, not fallback support.

Run the focused contract test, then physically qualify the emitted programs and
full model using unchanged weights, reference tolerances and output delivery.
Keep diagnostic instrumentation separate from uninstrumented paired timing.
CPU admission proves selection and boundary handling, not numerics or speed.
Do not promote a conversion manifest before reference and performance gates pass.

In the b72ad041 35B product source, CPU reconstruction selects subgroup
GEMV for linear QKV, Z and output projections. The vocabulary-head step selects
the tiled F16-weight/F32-activation kernel, and F32 controls select tiled F32.
Do not generalize the vocabulary-head issue to every linear projection without
an actual selected-variant receipt.

This source reconstruction is separate from the 405409c9 measured execution
receipt and from the native packed-format 5d2c13bf lane. Use actual per-operation
receipts before attributing a selected entry point or cost to either run.
