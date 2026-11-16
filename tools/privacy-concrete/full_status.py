"""Construct bounded client status before observers; exception text never crosses this boundary."""
import math

FAILURE_CODES=frozenset({'cancelled','deadline-exceeded','resource-exhausted','contract-rejected',
                         'ciphertext-verification-failed','token-reference-mismatch','component-reference-mismatch','execution-failed'})


class ExecutionRejected(ValueError):
    def __init__(self,code):
        if code not in FAILURE_CODES:
            raise ValueError('Unknown public execution failure code')
        self.code=code
        super().__init__(code)


def failure_event(error,elapsed_ms):
    if isinstance(error,ExecutionRejected) and isinstance(error.code,str) and error.code in FAILURE_CODES:code=error.code
    elif isinstance(error,(InterruptedError,KeyboardInterrupt)):code='cancelled'
    elif isinstance(error,TimeoutError):code='deadline-exceeded'
    elif isinstance(error,MemoryError):code='resource-exhausted'
    elif isinstance(error,ValueError):code='contract-rejected'
    else:code='execution-failed'
    if type(elapsed_ms) not in (int,float) or not math.isfinite(elapsed_ms) or elapsed_ms<0:
        raise ValueError('Finite elapsed execution time required')
    return dict(event='run-failed',failureCode=code,elapsedMs=elapsed_ms)
