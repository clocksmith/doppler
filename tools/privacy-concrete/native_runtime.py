"""Installed native arithmetic selection; no advertisement can register an implementation."""
import hashlib
import importlib
import json
from pathlib import Path
from protocol import POLICY

ROOT=Path(__file__).resolve().parent

def build_policy():return json.loads((ROOT/'native-ntt/build-policy.json').read_text())


def ntt_source_id():
    return hashlib.sha256(json.dumps(build_policy(),sort_keys=True).encode()+(ROOT/'native-ntt/ntt_cpu.rs').read_bytes()+(ROOT/'native-ntt/python_api.rs').read_bytes()).hexdigest()


def ntt_parameters():
    dependencies=dict(POLICY['dependencies'],**{'concrete-ml-extensions':build_policy()['nativeVersion']})
    return dict(dependencies=dependencies,parameters=json.loads(json.dumps(POLICY['parameters'])),evaluationBackend='cpu-exact-ntt',nativeSourceId=ntt_source_id())


def ntt_versions():
    from full_plan import dependency_versions,FULL_POLICY
    actual=dependency_versions(ntt_parameters()['dependencies'],FULL_POLICY['dependencyVariants'])
    ext=importlib.import_module('concrete_ml_extensions')
    if (getattr(ext,'__doppler_ntt_build_id__',None)!=ntt_source_id() or ext.is_cuda_enabled()
            or not callable(getattr(ext,'cpu_matrix_multiplication_ntt',None))):
        raise ValueError('Installed CPU NTT implementation differs from the admitted source')
    binary=importlib.import_module('concrete_ml_extensions.concrete_ml_extensions')
    actual['native-binary-sha256']=hashlib.sha256(Path(binary.__file__).read_bytes()).hexdigest()
    actual['native-source-id']=ntt_source_id()
    return actual


def matrix_evaluator(plan):
    backend=plan.definition()['parameters']['evaluationBackend']
    if backend not in ['cpu','cpu-exact-ntt']:raise ValueError('Unsupported installed native evaluator')
    ext=importlib.import_module('concrete_ml_extensions')
    if backend=='cpu':return ext.cpu_matrix_multiplication
    ntt_versions()
    return ext.cpu_matrix_multiplication_ntt


def prepare_evaluator_key(plan,key):
    backend=plan.definition()['parameters']['evaluationBackend']
    if backend=='cpu':return
    if backend!='cpu-exact-ntt':raise ValueError('Unsupported native key preparation')
    importlib.import_module('concrete_ml_extensions').cpu_prepare_ntt_key(key)
