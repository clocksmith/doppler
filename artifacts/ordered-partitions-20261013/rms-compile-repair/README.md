# Paired normalization compiler repair

Component: Doppler paired RMSNorm. Intent preserved. Boundary effects: no
wrapper, layout, resource, precision, scalar operand or evaluation-order change.

The mean and reciprocal share one division callsite outside the unchanged
paired square-root refinement. The symbolic scalar operation trace is identical.
intel-host's actual native provider passes all six original seeded independent F64
cases at unchanged 2e-12, maximum error 5.809e-15, including pre/post residual
and offset paths. Both distinct F32 specializations compile within the fixed
300-second diagnostic; queue, device, host and process settlement pass. Peak
cgroup memory is 4,853,059,584 bytes, with no limit events. Different constants
from the earlier full-model first-use samples prevent a matched speedup claim.

Mac independently passes the same original six cases on the exact candidate
shader. The first Mac command started before the source copy finished; its
hash identifies it as an old-shader control and it is retained separately.
Native startup initially preceded file transfer; no GPU opened. The next probe
correctly rejected boolean specialization constants in the native binding;
its teardown passes. The corrected adapter uses numeric 0/1 values, preserving
the declared booleans. These diagnostic failures remain retained.

Kernel/digest/style checks and wrapper resource-ownership controls pass. This
is operator evidence, not the 55-prefix model gate or a deployment. Full-model
cold preparation still needs complete specialization coverage before readiness.
