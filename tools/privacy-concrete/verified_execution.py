"""Complete and close a fixed verified route before client-local numerical checks.

Integrity, cancellation and runtime failures still terminate immediately. This
phase ordering removes reference-acceptance control over transport completion;
it does not provide constant-time execution or a reviewed leakage guarantee.
"""

VALIDATION_BOUNDARY='after-route-close/v1'


def complete_verified_route(bridge,steps,execute,validate):
    """Retain attempt-local outputs; release none before every check passes."""
    try:
        if type(steps) is not int or steps<=0 or not callable(execute) or not callable(validate):
            raise ValueError('Explicit fixed execution and validation phases required')
        results=[execute(step) for step in range(steps)]
        bridge.finish()
    finally:
        # Closure must succeed before any comparison, including on remote SSH.
        bridge.close()
    for step,result in enumerate(results):validate(step,result)
    return results
