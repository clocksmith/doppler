import math
import threading
import unittest
from oblivious_experts import ExpertProgram, FlashMoEProgram, flash_expert_program


class ExpertTests(unittest.TestCase):
    def program(self):return ExpertProgram('a'*64,3,1,2,2)
    def projection(self,trace,fail=None,cancelled=None):
        def project(operation,private_value):
            trace.append(dict(operation))
            if operation['id']==fail:raise ValueError('verification rejected unselected expert')
            if cancelled is not None:cancelled.set()
            # Public fixture: expert e has diagonal gate=(e+1)*I, up=I, down=I.
            expert,name=operation['id'].split('/')[1:]
            return [value*(int(expert)+1 if name=='gate' else 1) for value in private_value]
        return project

    def test_dense_execution_matches_selected_reference_and_has_same_public_schedule(self):
        traces=[]
        for selected in [0,2]:
            trace=[];hidden=[1.,-2.]
            output=self.program().attempt().run(hidden,[selected],[.5],self.projection(trace))
            expected=[.5*value*(1/(1+math.exp(-(selected+1)*value)))*(selected+1)*value for value in hidden]
            for left,right in zip(output,expected):self.assertAlmostEqual(left,right)
            self.assertEqual(len(trace),9)
            self.assertNotIn('selected',str(trace));traces.append(trace)
        self.assertEqual(*traces)

    def test_corrupt_unselected_expert_rejects_for_every_private_selection(self):
        traces=[]
        for selected in [0,2]:
            trace=[];attempt=self.program().attempt()
            with self.assertRaises(ValueError):attempt.run([1,2],[selected],[1],self.projection(trace,fail='expert/1/up'))
            self.assertTrue(attempt.failed)
            with self.assertRaises(ValueError):attempt.run([1,2],[selected],[1],self.projection(trace))
            traces.append(trace)
        self.assertEqual(*traces)

    def test_revocation_and_bad_selection_stop_before_further_projection(self):
        trace=[];event=threading.Event();attempt=self.program().attempt()
        with self.assertRaises(InterruptedError):attempt.run([1,2],[0],[1],self.projection(trace,cancelled=event),cancelled=event)
        self.assertEqual(len(trace),1);event.clear()
        with self.assertRaises(ValueError):attempt.run([1,2],[0],[1],self.projection(trace))
        for selection in [[True],[3],[0,1]]:
            with self.assertRaises(ValueError):self.program().attempt().run([1,2],selection,[1],self.projection(trace))
        self.assertEqual(len(trace),1)

    def test_bad_shape_never_reaches_next_operator_and_success_is_not_reusable(self):
        attempt=self.program().attempt();trace=[]
        with self.assertRaises(ValueError):attempt.run([1,2],[0],[1],lambda operation,value:trace.append(operation) or [0])
        self.assertEqual(len(trace),1)
        attempt=self.program().attempt();attempt.run([1,2],[0],[1],self.projection([]))
        with self.assertRaises(ValueError):attempt.run([1,2],[0],[1],self.projection([]))

    def test_private_inputs_and_results_are_owned_snapshots(self):
        hidden=[1.,2.];selected=[0];weights=[1.];buffer=[];calls=[]
        def project(operation,value):
            calls.append(operation['id'])
            if len(calls)==1:
                hidden[0]=99.;selected[0]=2;weights[0]=0.
            expert,name=operation['id'].split('/')[1:]
            buffer[:]=[number*(int(expert)+1 if name=='gate' else 1) for number in value]
            return buffer
        result=self.program().attempt().run(hidden,selected,weights,project)
        for actual,number in zip(result,[1.,2.]):self.assertAlmostEqual(actual,number*number/(1+math.exp(-number)))
        self.assertEqual(len(calls),9)
        def mutate(operation,value):operation['id']='selected-only'
        with self.assertRaises(TypeError):self.program().attempt().run([1,2],[0],[1],mutate)

    def test_local_routing_semantics_are_bound_even_when_remote_schedule_is_identical(self):
        first=FlashMoEProgram(self.program(),2,True,'b'*64)
        second=FlashMoEProgram(ExpertProgram('a'*64,3,2,2,2),2,True,'b'*64)
        self.assertEqual(first.contract()['operations'],second.contract()['operations'])
        self.assertNotEqual(first.identity,second.identity)
        self.assertNotEqual(first.identity,FlashMoEProgram(self.program(),2,False,'b'*64).identity)

    def test_pinned_flash_geometry_builds_all_experts_without_claiming_execution(self):
        program=flash_expert_program();contract=program.contract()
        self.assertEqual((program.expert_count,program.top_k,program.hidden_width,program.intermediate_width),(512,10,2560,640))
        self.assertEqual(len(contract['operations']),1536)
        self.assertEqual(contract['operations'][-1]['id'],'expert/511/down')
        self.assertNotIn('qualified',contract);self.assertNotIn('selectedIds',contract)
        self.assertNotEqual(program.identity,self.program().identity)


if __name__=='__main__':unittest.main()
