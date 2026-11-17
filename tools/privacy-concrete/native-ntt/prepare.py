"""Patch one exact public upstream checkout; never an advertisement-selected source or library."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
POLICY=json.loads((ROOT/'build-policy.json').read_text())


def replace_one(value,old,new):
    if value.count(old)!=1:raise ValueError('Exact native patch site changed')
    return value.replace(old,new)


def prepare(directory):
    directory=Path(directory)
    if directory.exists():raise ValueError('Fresh native checkout required')
    directory.mkdir(parents=True)
    def git(*arguments):subprocess.run(['git','-C',str(directory),*arguments],check=True,stdout=subprocess.DEVNULL)
    git('init');git('remote','add','origin',POLICY['repository']);git('fetch','--depth','1','origin',POLICY['revision']);git('checkout','--detach','FETCH_HEAD')
    revision=subprocess.check_output(['git','-C',str(directory),'rev-parse','HEAD'],text=True).strip()
    if revision!=POLICY['revision']:raise ValueError('Native revision mismatch')
    for name,identity in POLICY['files'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=identity:raise ValueError('Pinned native build input changed')
    p=directory/'rust/Cargo.toml';text=p.read_text().replace('version = "0.1.9"','version = '+json.dumps(POLICY['nativeVersion']))
    text=replace_one(text,'[dependencies]','[dependencies]\ntfhe-ntt = { version = "=0.7.1", default-features = false, features = ["std"] }');p.write_text(text)
    p=directory/'rust/src/lib_switcher.rs';p.write_text(replace_one(p.read_text(),'mod ml;','mod ml;\nmod ntt_cpu;'))
    p=directory/'rust/src/ml.rs';p.write_text(p.read_text()+"\nimpl EncryptedVector<u64> { pub(crate) fn ntt_parts(&self) -> (&[GlweCiphertextOwned<u64>], usize) { (&self.data,self.actual_len) } }\n")
    p=directory/'rust/src/fhext_classes.rs';p.write_text(replace_one(p.read_text(),
        'pub(crate) inner: compression::CompressionKey<Scalar>,',
        'pub(crate) inner: compression::CompressionKey<Scalar>,\n    #[serde(skip)]\n    pub(crate) ntt_cache: std::sync::OnceLock<std::sync::Arc<crate::ntt_cpu::PackingPlan>>,'))
    p=directory/'rust/src/lib_python.rs';text=replace_one(p.read_text(),
        'CpuCompressionKey {\n            inner: compression_key,',
        'CpuCompressionKey {\n            inner: compression_key,\n            ntt_cache: std::sync::OnceLock::new(),')
    text=replace_one(text,'m.add_function(wrap_pyfunction!(cpu_matrix_multiplication, m)?)?;',
        'm.add_function(wrap_pyfunction!(cpu_matrix_multiplication, m)?)?;\n    m.add_function(wrap_pyfunction!(cpu_prepare_ntt_key, m)?)?;\n    m.add_function(wrap_pyfunction!(cpu_matrix_multiplication_ntt, m)?)?;\n    m.add_function(wrap_pyfunction!(cpu_matrix_multiplication_ntt_serial, m)?)?;')
    text+='\n'+(ROOT/'python_api.rs').read_text();p.write_text(text)
    (directory/'rust/src/ntt_cpu.rs').write_bytes((ROOT/'ntt_cpu.rs').read_bytes())
    lock=(ROOT/'Cargo.lock').read_bytes()
    if hashlib.sha256(lock).hexdigest()!=POLICY['cargoLockSha256']:raise ValueError('Pinned native dependency lock changed')
    (directory/'rust/Cargo.lock').write_bytes(lock)
    build_id=hashlib.sha256(json.dumps(POLICY,sort_keys=True).encode()+(ROOT/'ntt_cpu.rs').read_bytes()+(ROOT/'python_api.rs').read_bytes()).hexdigest()
    for name in ['pyproject.toml','src/concrete_ml_extensions/__init__.py']:
        p=directory/name;p.write_text(p.read_text().replace('"0.1.9"',json.dumps(POLICY['nativeVersion'])))
    p=directory/'src/concrete_ml_extensions/__init__.py';p.write_text(p.read_text()+'\n__doppler_ntt_build_id__ = '+repr(build_id)+'\n')
    print(json.dumps(dict(buildId=build_id,revision=revision,nativeVersion=POLICY['nativeVersion'],productionBackendAvailable=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);args=parser.parse_args();prepare(args.destination)
