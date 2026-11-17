"""Exact policy/program binding without NumPy, native keys or model payloads."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from check_policy import POLICY,POLICY_ID,freeze_policy,policy_identity,policy_binding,installed_binding,require_bound_policy

PROFILE='glwe-quant12-message35-storage48/v1'


class CheckPolicyTests(unittest.TestCase):
    def test_snapshot_is_deeply_immutable_and_isolated_from_caller_changes(self):
        original=copy.deepcopy(POLICY);snapshot,identity=freeze_policy(original)
        original['replaySample']['denominator']=64;original['noise'][PROFILE]['margin']=4
        self.assertEqual(snapshot['replaySample']['denominator'],16);self.assertEqual(snapshot['noise'][PROFILE]['margin'],2)
        self.assertEqual(identity,POLICY_ID);self.assertEqual(policy_identity(snapshot),identity)
        for mutate in (lambda:snapshot.__setitem__('checkVectors',32),lambda:snapshot['replaySample'].__setitem__('denominator',64)):
            with self.assertRaises((TypeError,AttributeError)):mutate()

    def test_every_policy_change_changes_digest_and_bound_program_identity(self):
        from flash_text_ntt12_check_plan import PROGRAM,PROGRAM_ID,CONFIG
        from flash_text_plan import PROGRAM_ID as REPLAY_PROGRAM_ID
        from research_parameters import component_parameters
        from full_plan import canonical
        from protocol import digest
        self.assertEqual(PROGRAM['integrityPolicyId'],POLICY_ID)
        self.assertEqual(component_parameters(CONFIG)['integrityPolicyId'],POLICY_ID)
        self.assertNotEqual(PROGRAM_ID,REPLAY_PROGRAM_ID)
        for field in ('sample','vectors','noise','quantile','evidence'):
            policy=copy.deepcopy(POLICY)
            if field=='sample':policy['replaySample']['denominator']=64
            elif field=='vectors':policy['checkVectors']=32
            elif field=='noise':policy['noise'][PROFILE]['margin']=3
            elif field=='quantile':policy['falseRejectLog2']=41
            else:policy['noise'][PROFILE]['evidence']='public-calibration-v2.json'
            with self.subTest(field=field):
                binding=policy_binding(policy);self.assertNotEqual(binding['integrityPolicyId'],POLICY_ID)
                self.assertNotEqual(digest(canonical(dict(PROGRAM,**binding))),PROGRAM_ID)
        self.assertNotIn('integrityPolicyId',component_parameters({}))

    def test_real_source_configuration_reload_changes_program_and_parameter_binding(self):
        import flash_text_ntt12_check_plan as original
        root=Path(__file__).resolve().parent;changed=copy.deepcopy(POLICY);changed['replaySample']['denominator']=64
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            (directory/'check_policy.py').write_bytes((root/'check_policy.py').read_bytes())
            (directory/'linear-check-policy.json').write_text(json.dumps(changed))
            code='import json; import flash_text_ntt12_check_plan as p; from check_policy import POLICY_ID; print(json.dumps(dict(policy=POLICY_ID,program=p.PROGRAM_ID,parameter=p.parameters()["integrityPolicyId"])))'
            result=json.loads(subprocess.check_output([sys.executable,'-c',code],cwd=directory,
                env=dict(PATH=os.defpath,PYTHONPATH=os.pathsep.join((str(directory),str(root))),PYTHONDONTWRITEBYTECODE='1'),text=True))
        self.assertEqual(result['policy'],policy_identity(changed));self.assertEqual(result['parameter'],result['policy'])
        self.assertNotEqual(result['program'],original.PROGRAM_ID)

    def test_module_mutation_rejects_new_admission_without_changing_existing_snapshot(self):
        snapshot,identity=freeze_policy(POLICY);before=copy.deepcopy(POLICY)
        try:
            POLICY['replaySample']['denominator']=64
            with self.assertRaisesRegex(ValueError,'new compiled policy'):installed_binding()
            self.assertEqual(snapshot['replaySample']['denominator'],16);self.assertEqual(policy_identity(snapshot),identity)
        finally:POLICY.clear();POLICY.update(before)
        self.assertEqual(installed_binding()['integrityPolicyId'],POLICY_ID)

    def test_missing_wrong_and_legacy_policy_bindings_reject(self):
        for parameters in ({},{'integrityPolicyId':'f'*64},{'integrityPolicyId':True}):
            with self.subTest(parameters=parameters),self.assertRaises(ValueError):
                require_bound_policy(SimpleNamespace(definition=lambda:dict(parameters=parameters)),PROFILE)
        plan=SimpleNamespace(definition=lambda:dict(parameters={'integrityPolicyId':POLICY_ID}))
        self.assertEqual(require_bound_policy(plan,PROFILE)['id'],POLICY['id'])
        with self.assertRaises(ValueError):require_bound_policy(plan,'unsupported-profile')

    def test_unknown_fields_and_qualification_flags_cannot_register_a_policy(self):
        for field in ('securityQualified','productionBackendAvailable','independentCryptographicReview','advertisedAttestation'):
            candidate=copy.deepcopy(POLICY);candidate[field]=True
            with self.subTest(field=field),self.assertRaises(ValueError):freeze_policy(candidate)

    def test_parameter_binding_cannot_attach_to_another_mechanism_or_unbound_profile(self):
        from research_parameters import component_parameters
        for policy in ({'integrityPolicyId':POLICY_ID},{'integrity':'advertised-check/v1'},
                {'integrity':POLICY['id'],'integrityPolicyId':'f'*64}):
            with self.subTest(policy=policy),self.assertRaises(ValueError):component_parameters(policy)


if __name__=='__main__':unittest.main()
