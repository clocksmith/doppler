import hashlib, json, os, pathlib, subprocess, sys, time
root, models, configs, out = map(pathlib.Path, sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
tests = [
 'tests/kernels/fused-matmul-q4-widetile-parity.test.js',
 'tests/inference/fused-matmul-q4-metal-simd16-gpu-regression.test.js',
 'tests/inference/linear-attention-core-gpu-regression.test.js',
 'tests/inference/logits-output-ownership.test.js',
 'tests/inference/logits-paired-physical.test.js',
 'tests/gpu/command-recorder-cleanup.test.js',
 'tests/gpu/dispatch-device-loss-context.test.js',
 'tests/gpu/device-buffer-pool-epoch.test.js',
 'tests/inference/decode-output-ownership.test.js',
]
rows = []
def run(name, argv, env_extra={}, timeout=240):
 dest=out/name;dest.mkdir(exist_ok=True)
 cmd=dict(cwd=str(root),argv=argv,environment=env_extra,sourceSha256=hashlib.sha256((root/argv[1]).read_bytes()).hexdigest())
 (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
 start=time.monotonic()
 with (dest/'run.log').open('w') as log:
  try:
   p=subprocess.run(argv,cwd=root,env={**os.environ,**env_extra},stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
   status=dict(exitCode=p.returncode)
  except subprocess.TimeoutExpired:status=dict(timeout=True)
 output=(dest/'run.log').read_text()
 status.update(name=name,elapsedSeconds=time.monotonic()-start,skipLines=[l for l in output.splitlines() if 'skipped' in l.lower()])
 (dest/'status.json').write_text(json.dumps(status,indent=2)+'\n');rows.append(status)
 (out/'status.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(status),flush=True)
for test in tests:run(pathlib.Path(test).stem,['node',test,'--require-gpu'])
for size in ['0-8b','2b']:
 model=models/f'qwen-3-5-{size}-q4k-ehaf16'
 run(f'{size}-prepared',['node','tests/integration/linear-prepared-sequence-physical.test.js','--require-gpu'],{'DOPPLER_PREPARED_MODEL_DIR':str(model),'DOPPLER_PREPARED_CONFIG':str(configs/f'{size}-lifecycle/config.json')})
 run(f'{size}-linear',['node','tests/integration/qwen-linear-attention-regression.test.js'],{'DOPPLER_QWEN35_LINEAR_MODEL_DIR':str(model)})
