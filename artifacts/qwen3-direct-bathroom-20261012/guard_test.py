from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('guard',Path(__file__).with_name('run.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
seen=[];write=m.guard_writer(lambda path, report:seen.append(report))
for row in [{},{'stopReason':'eos'},{'completionPassed':True,'passed':None},{'completionPassed':True,'passed':False}]:
    write(None,{'results':[row]})
assert len(seen)==4
bad={'results':[{'id':'synthetic','completionPassed':False}]}
try:write(None,bad);raise AssertionError('Structural rejection was not stopped')
except m.StructuralRejection:pass
assert seen[-1] is bad,'Retain full original report before stopping'
print('five guard controls pass; factual/pending rows do not select early results')
