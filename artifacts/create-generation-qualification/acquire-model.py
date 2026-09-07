import concurrent.futures, datetime, hashlib, json, pathlib, urllib.request
base='https://huggingface.co/clocksmith/rdrr/resolve/7c3d30e300bcb02cbd68fb0db3eee64fbf738f99/models/gemma-3-1b-it-q4k-ehf16-af32/'
out=pathlib.Path('models/local/create-gemma-3-1b-qualification')
out.mkdir(parents=True,exist_ok=True)
evidence=pathlib.Path('artifacts/create-generation-qualification')
def fetch(name):
    target=out/name
    if not target.exists():
        temporary=target.with_suffix(target.suffix+'.partial')
        with urllib.request.urlopen(base+name,timeout=120) as response, temporary.open('wb') as dest:
            while chunk:=response.read(1024*1024): dest.write(chunk)
        temporary.rename(target)
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    row={'filename':name,'bytes':target.stat().st_size,'sha256':digest,'url':base+name}
    print(json.dumps(row),flush=True)
    return row
manifest=fetch('manifest.json')
raw=(out/'manifest.json').read_bytes()
(evidence/'published-manifest.json').write_bytes(raw)
m=json.loads(raw)
names=[s['filename'] for s in m['shards']]+['tokenizer.json','origin.json']
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool: rows=list(pool.map(fetch,names))
(evidence/'acquisition.json').write_text(json.dumps({'capturedAt':datetime.datetime.now(datetime.timezone.utc).isoformat(),'manifest':manifest,'files':rows},indent=2)+'\n')
