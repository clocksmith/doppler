import subprocess,os,signal,time,json
from pathlib import Path
r=Path(__file__).parent
out=r/'run-11';out.mkdir(exist_ok=True)
def available():return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
assert available()>65*1024**3, 'Insufficient memory headroom'
with (out/'runtime.log').open('w') as log:
 p=subprocess.Popen(['node','artifacts/qwen38-27b-webgpu-20261015/run.js',str(Path.home()/'deco/doppler/models/local/qwen3-8-27b-text-f16-af32'),str(out)],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 (out/'pid').write_text(str(p.pid))
 start=time.monotonic();reason=None
 with (out/'memory.jsonl').open('w') as samples:
  while p.poll() is None:
   mem=available();gtt=int(Path('/sys/class/drm/card1/device/mem_info_gtt_used').read_text());samples.write(json.dumps({'time':time.time(),'availableBytes':mem,'gttUsedBytes':gtt})+chr(10));samples.flush()
   if gtt>int(Path('/sys/class/drm/card1/device/mem_info_gtt_total').read_text())-2*1024**3:reason='gpu-reserve';break
   if mem<12*1024**3:reason='memory-reserve';break
   if time.monotonic()-start>600:reason='600s-deadline';break
   time.sleep(.5)
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
  p.wait()
 (out/'exit.json').write_text(json.dumps({'code':p.returncode,'guardReason':reason,'elapsed':time.monotonic()-start}))
