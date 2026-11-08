#!/usr/bin/env python3
"""Admission observation, not a guarantee of an idle host throughout a run."""
from pathlib import Path
import datetime
import json
import subprocess
import sys
import urllib.request

contract = json.loads(Path('experiment.json').read_text())
gtt = Path('/sys/class/drm/card1/device')
total = int((gtt / 'mem_info_gtt_total').read_text())
used = int((gtt / 'mem_info_gtt_used').read_text())
memory = {line.split(':')[0]: int(line.split()[1]) * 1024
          for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')}
unit = 'reploid-qwen27b-0633-20261104.service'
provider = subprocess.check_output(['systemctl', '--user', 'show', unit,
    '--property=ActiveState,MainPID', '--no-pager'], text=True).strip()
health = json.load(urllib.request.urlopen('http://127.0.0.1:8989/health'))
budget = contract['invariants']['gpuBudgetBytes']
row = {'checkedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'gttTotalBytes': total, 'gttUsedBytes': used, 'availableSystemBytes': memory['MemAvailable'],
       'budgetBytes': budget, 'provider': provider, 'experimentalEndpointHealth': health,
       'ttmPagesLimit': Path('/sys/module/ttm/parameters/pages_limit').read_text().strip()}
Path(sys.argv[1]).write_text(json.dumps(row, indent=2) + '\n')
assert total - used > budget + 2 * 1024 ** 3, 'Insufficient free GTT for budget plus admission headroom'
assert memory['MemAvailable'] > budget + 8 * 1024 ** 3, 'Insufficient system memory'
assert 'ActiveState=active' in provider and health['ready'] and not health['active']
