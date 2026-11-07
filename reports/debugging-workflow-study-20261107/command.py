#!/usr/bin/python3
"""Investigator command wrapper. OS restrictions apply to its subprocess only."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
if len(sys.argv) != 2:
    raise SystemExit('Usage: /usr/bin/python3 case/command.py "shell command"')
observations = root / 'observations'
observations.mkdir(exist_ok=True)
started = time.time()
identifier = str(time.time_ns())
environment = {key: os.environ[key] for key in ['HOME', 'TMPDIR', 'USER', 'LANG'] if key in os.environ}
environment['PATH'] = str(root / 'bin') + ':/usr/bin:/bin:/opt/homebrew/bin'
args = ['/usr/bin/sandbox-exec', '-f', str(root / 'case/offline.sb'), '/bin/sh', '-c', sys.argv[1]]
with (observations / (identifier + '.stdout')).open('wb') as stdout, (observations / (identifier + '.stderr')).open('wb') as stderr:
    result = subprocess.run(args, cwd=root, env=environment, stdout=stdout, stderr=stderr)
completed = time.time()
row = {'id': identifier, 'command': sys.argv[1], 'startedAt': datetime.datetime.fromtimestamp(started, datetime.timezone.utc).isoformat(),
       'completedAt': datetime.datetime.fromtimestamp(completed, datetime.timezone.utc).isoformat(),
       'elapsedSeconds': completed - started, 'exitCode': result.returncode}
with (observations / 'commands.jsonl').open('a') as journal:
    journal.write(json.dumps(row) + '\n')
sys.stdout.buffer.write((observations / (identifier + '.stdout')).read_bytes())
sys.stderr.buffer.write((observations / (identifier + '.stderr')).read_bytes())
raise SystemExit(result.returncode)
