import pathlib,json,hashlib,collections,tarfile,shutil
root=pathlib.Path.cwd();out=root/'reports/qwen-profile-correction-20261017';rows=[];diagnostics=[];lifecycle=[]
for p in sorted(out.glob('*/*/compare_latest.json')):
 d=json.loads(p.read_text());s=d['sections']['warm'];rows.append({'lane':p.parent.parent.name,'order':d['methodology']['warmEngineOrder'],'model':d['dopplerModelId'],'receipt':str(p.relative_to(out)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'fairness':d['fairness'],'outputParity':s['outputParity'],'doppler':s['doppler']['result']['timing'],'transformersjs':s['transformersjs']['timing'],'memoryAccounting':s['memoryAccounting'],'efficiencyMetrics':s['efficiencyMetrics'],'dopplerBottleneck':s['dopplerBottleneck'],'artifact':d['dopplerArtifactIdentity'],'environment':d['environment'],'deviceInfo':s['doppler']['result']['deviceInfo'],'tjsDeviceInfo':s['transformersjs']['deviceInfo']})
for p in sorted(out.glob('*/*diagnostic-debug/run.log')):
 t=p.read_text();r=json.loads(t[t.index('{'):])['result'];m=r['metrics'];g=collections.Counter()
 for step in m['prefillProfileSteps']:
  for name,ms in step['timings'].items():g[':'.join(name.split(':')[:2]) if name.startswith('matmul:') else name]+=ms
 diagnostics.append({'lane':p.parent.parent.name,'model':r['modelId'],'scope':'One fresh instrumented generation; debug does not run the requested benchmark warmup loop. Not warm or comparative timing.','prefillWallMs':m['prefillMs'],'gpuMs':m['gpu']['prefillMs'],'recordWallMs':m['gpu']['prefillRecordMs'],'submitWaitWallMs':m['gpu']['prefillSubmitWaitMs'],'gpuGroupsMs':dict(g.most_common()),'source':str(p.relative_to(out))})
for p in sorted(out.glob('*/*lifecycle/run.log')):
 t=p.read_text();start=t.index('{\n  "modelId"');d=json.JSONDecoder().raw_decode(t[start:])[0];assert d['passed'];lifecycle.append({'lane':p.parent.parent.name,'model':d['modelId'],'passed':d['passed'],'rows':len(d['rows']),'cancellation':d['cancellation'],'scope':d['scope'],'source':str(p.relative_to(out))})
assert len(rows)==8 and len(diagnostics)==4 and len(lifecycle)==4
summary={'decision':'retain explicit Linux parity profile selection; no runtime code or kernel correction','scope':'Local Qwen3.5 Q4_K/f32-compute/f16-KV versus Transformers.js 4.2.0 ONNX q4f16; product formats differ. Browser performance and Node lifecycle are separate.','rows':rows,'prefillDiagnostics':diagnostics,'lifecycle':lifecycle,'openFailures':['0.8B warm TTFT loses on both GPUs','Metal 2B TTFT changes winner between engine orders','Vulkan 2B loading loses in both orders','Every measured Doppler token-delivery p95 loses','No network-cold startup qualification','No universal numerical or useful-answer qualification; movie quality rejection unchanged','Existing architecture limit and skipped release checks remain open']}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
source=out/'executed-config';source.mkdir(exist_ok=True)
for p in ['benchmarks/vendors/compare-engines.config.json','src/config/runtime/profiles/qwen-3-5-0-8b-vulkan-parity.json','src/config/runtime/profiles/qwen-3-5-2b-vulkan-parity.json']:
 target=source/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/'.tmp/qwen-profile-correction-20261017'/p,target)
with tarfile.open(out/'raw-receipts.tar.gz','w:gz') as tar:
 for lane in ['metal','vulkan']:
  for p in sorted((out/lane).rglob('*')):
   if p.is_file() and (p.name in ['compare_latest.json','run.log','command.json','config.json','status.json']):tar.add(p,arcname=str(p.relative_to(out)))
print('summary',len(rows),'comparisons',len(diagnostics),'diagnostics',len(lifecycle),'lifecycle proofs')
