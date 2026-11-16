"""Actual phase owner: identical completion transcript for numerical acceptance/rejection."""
from types import SimpleNamespace
import unittest
from verified_execution import complete_verified_route


class VerifiedExecutionTests(unittest.TestCase):
    def attempt(self,reject=None,execution_error=None,finish_error=None,close_error=None):
        trace=[];wire=[]
        def execute(step):
            trace.append(('execute',step));wire.append(('ciphertext',step))
            if step==2 and execution_error is not None:raise execution_error
            return step+10
        def finish():
            trace.append(('finish',));wire.append(('finish',))
            if finish_error is not None:raise finish_error
        def close():
            trace.append(('close',));wire.append(('close',))
            if close_error is not None:raise close_error
        def validate(step,result):
            self.assertEqual(trace[4:6],[('finish',),('close',)])
            self.assertEqual(result,step+10);trace.append(('validate',step))
            if step==reject:raise ValueError('client-local numerical rejection')
        bridge=SimpleNamespace(finish=finish,close=close)
        return trace,wire,lambda:complete_verified_route(bridge,4,execute,validate)

    def test_all_fixed_steps_finish_and_close_before_any_validation(self):
        trace,wire,run=self.attempt();self.assertEqual(run(),[10,11,12,13])
        self.assertEqual(trace,[('execute',i) for i in range(4)]+[('finish',),('close',)]+[('validate',i) for i in range(4)])

    def test_every_numerical_failure_has_the_successful_completion_transcript_and_no_result(self):
        _,baseline,run=self.attempt();run()
        for failed_step in range(4):
            trace,wire,run=self.attempt(reject=failed_step)
            with self.assertRaises(ValueError):run()
            self.assertEqual(wire,baseline);self.assertEqual(trace[-1],('validate',failed_step))

    def test_integrity_or_cancellation_failure_still_stops_without_later_steps_or_validation(self):
        for error in [ValueError('ciphertext verification failed'),InterruptedError('cancelled')]:
            trace,wire,run=self.attempt(execution_error=error)
            with self.assertRaises(type(error)):run()
            self.assertEqual(trace,[('execute',0),('execute',1),('execute',2),('close',)])
            self.assertNotIn(('finish',),wire)

    def test_finish_failure_closes_without_client_validation(self):
        trace,wire,run=self.attempt(finish_error=OSError('remote finish failed'))
        with self.assertRaises(OSError):run()
        self.assertEqual(trace[-2:],[('finish',),('close',)])
        self.assertFalse(any(row[0]=='validate' for row in trace))

    def test_cleanup_failure_never_reaches_validation(self):
        trace,wire,run=self.attempt(close_error=OSError('cleanup failed'))
        with self.assertRaises(OSError):run()
        self.assertFalse(any(row[0]=='validate' for row in trace))

    def test_invalid_phase_contract_closes_before_any_callback(self):
        for steps in [0,-1,True,None]:
            calls=[];bridge=SimpleNamespace(finish=lambda:calls.append('finish'),close=lambda:calls.append('close'))
            with self.assertRaises(ValueError):complete_verified_route(bridge,steps,lambda i:calls.append(i),lambda *args:calls.append('validate'))
            self.assertEqual(calls,['close'])


if __name__=='__main__':unittest.main()
