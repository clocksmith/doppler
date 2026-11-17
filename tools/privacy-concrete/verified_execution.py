"""Complete and close a fixed verified route before client-local numerical checks.

Integrity, cancellation and runtime failures still terminate immediately. This
phase ordering removes reference-acceptance control over transport completion;
it does not provide constant-time execution or a reviewed leakage guarantee.
"""
from full_status import ExecutionRejected

VALIDATION_BOUNDARY='after-route-close/v1'


def complete_verified_route(bridge,steps,execute,validate):
    """Retain attempt-local outputs; release none before every check passes."""
    try:
        try:
            if type(steps) is not int or steps<=0 or not callable(execute) or not callable(validate):
                raise ValueError('Explicit fixed execution and validation phases required')
            results=[execute(step) for step in range(steps)]
            bridge.finish()
        finally:
            # Closure must succeed before any comparison, including on remote SSH.
            bridge.close()
    except BaseException as error:
        # A checked route records integrity failures client-locally. An evaluator
        # that was caught must not be able to turn that into a transport or
        # runtime failure by ending the route early: the recorded integrity
        # failure outranks whatever ended the attempt.
        failed=getattr(bridge,'integrity_failed',None)
        if failed is not None and failed():raise ExecutionRejected('integrity-check-failed') from error
        raise
    # On a clean finish the same outcome is reported only here, after the fixed
    # transcript is complete and the transport is closed.
    require=getattr(bridge,'require_integrity',None)
    if require is not None:require()
    for step,result in enumerate(results):validate(step,result)
    return results
