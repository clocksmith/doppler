import fcntl
import json
import os
import pathlib
import shutil
import signal
import subprocess
import threading
import time

old = pathlib.Path('/tmp/doppler-shared-platform-mars-20261102')
root = pathlib.Path('/tmp/doppler-mars-20261103-diagnostic-pooled')
root.mkdir(exist_ok=False)
shutil.copytree(old / 'native-resource-screen-harness-04', root / 'harness', symlinks=True)
runner = root / 'harness/tools/qwen-economics/run.js'
browser = root / 'harness/tools/qwen-economics/browser.js'
s = browser.read_text()
needle = 'onToken(id) { tokenIds.push(id); deliveryMs.push(performance.now() - begin); if (config.countGpuApi) apiSnapshots.push(snapApi());'
assert s.count(needle) == 1
s = s.replace(needle, needle + " console.log('TOKEN_PROGRESS', JSON.stringify({index:tokenIds.length-1,id,elapsedMs:performance.now()-begin,allocations:{...allocationStats}}));")
browser.write_text(s)
s = runner.read_text()
needle = "  receipt.error = String(error.stack || error); receipt.complete = false; process.exitCode = 1;"
assert s.count(needle) == 1
s = s.replace(needle, needle + "\n  await save('receipt.json', receipt); console.log('RECEIPT_WRITTEN', receipt.error);")
needle = '    await item.context?.close(); await closeProbeServer(item.server);'
assert s.count(needle) == 1
s = s.replace(needle, "    console.log('CLEANUP_CONTEXT_START');\n" + needle + "\n    console.log('CLEANUP_SERVER_DONE');")
runner.write_text(s)
contract = json.loads((old / '27b-mars-native-resource-contract-04.json').read_text())
contract['scope'] = 'Diagnostic four-token reproduction, not performance evidence.'
contract['shared']['maxNewTokens'] = 4
(root / 'contract.json').write_text(json.dumps(contract, indent=2) + '\n')
stop = threading.Event()
def sample():
    with (root / 'memory-samples.jsonl').open('w') as out:
        while not stop.is_set():
            row = {'time': time.time()}
            for cmd in (['sysctl', 'vm.swapusage'], ['memory_pressure']):
                p = subprocess.run(cmd, capture_output=True, text=True)
                row[cmd[0]] = p.stdout[-1000:] + p.stderr[-500:]
            out.write(json.dumps(row) + '\n'); out.flush()
            stop.wait(5)

with open('/tmp/doe-gpu-timing.lock', 'r+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    thread = threading.Thread(target=sample, daemon=True); thread.start()
    try:
        for name, source, commit in [
            ('pooled', old / 'runtime-04', '727d36aa0f0e8705d4f3121ae727e463de4a9d6f'),
]:
            c = json.loads((old / '27b-mars-native-resource-pooled-04.json').read_text())
            c.update(contractPath=str(root / 'contract.json'), dopplerRoot=str(source),
                     runtimeRoots={'doppler':str(source)}, runtimeSourceCommit=commit,
                     outputDir=str(root / name), profileRoot=str(root / 'profiles'),
                     requestTimeoutMs=180000, repetitionsOverride=1, taskIds=['short'])
            p = root / (name + '-config.json'); p.write_text(json.dumps(c, indent=2) + '\n')
            print('START', name, flush=True)
            child = subprocess.Popen(['node', str(runner), str(p)], start_new_session=True)
            try:
                code = child.wait(timeout=270)
            except subprocess.TimeoutExpired:
                print('CLEANUP_TIMEOUT', name, flush=True)
                os.killpg(child.pid, signal.SIGTERM)
                try: code = child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); code = child.wait()
            print('EXIT', name, code, flush=True)
    finally:
        stop.set(); thread.join(timeout=10)
