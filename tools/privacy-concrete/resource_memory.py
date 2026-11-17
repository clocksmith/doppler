"""Client-local resource observations inside an already enforced private cgroup.

RSS sums double-count shared mappings; sampled peaks may miss short allocations.
Kernel cgroup counters include file cache. Neither is hardware attestation or a
prediction that a different model or deployment fits the same budget.
"""
import os
from pathlib import Path
import threading


def private_cgroup_path():
    rows=[line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    if len(rows)!=1:raise ValueError('One explicit cgroup-v2 process scope required')
    path=Path(rows[0])
    if not path.is_absolute() or len(path.parts)<2 or '..' in path.parts:raise ValueError('Private cgroup path required')
    return Path('/sys/fs/cgroup').joinpath(*path.parts[1:])


def process_rss(pid):
    if type(pid) is not int or pid<=0:raise ValueError('Positive owned process id required')
    try:
        rows=Path(f'/proc/{pid}/status').read_text().splitlines()
    except (FileNotFoundError,ProcessLookupError):return None
    values={line.split(':',1)[0]:line.split(':',1)[1].split() for line in rows if ':' in line}
    row=values.get('VmRSS')
    if row is None:return None
    if len(row)!=2 or row[1]!='kB' or not row[0].isdigit():raise ValueError('Explicit kernel RSS units required')
    return int(row[0])*1024


class ResourceMeter:
    """Count only processes in the caller's initially exclusive kernel scope."""
    def __init__(self,*,interval_seconds=.02):
        if type(interval_seconds) not in (float,int) or not .01<=interval_seconds<=1:
            raise ValueError('Bounded resource observation interval required')
        self.scope=private_cgroup_path()
        if (self.scope/'cgroup.procs').read_text().split()!=[str(os.getpid())]:
            raise ValueError('Resource observations require an initially exclusive scope')
        self.interval=interval_seconds;self.stop_event=threading.Event();self.thread=None
        self.lock=threading.Lock();self.samples=0;self.rss_peak=self.scope_peak=self.process_peak=0
        self.incomplete=0;self.failure=False;self.phases=[]

    def _sample(self):
        members=(self.scope/'cgroup.procs').read_text().split()
        if not members or any(not item.isdigit() or int(item)<=0 for item in members):raise ValueError('Owned kernel process scope changed')
        rss=[process_rss(int(item)) for item in members]
        current=int((self.scope/'memory.current').read_text());peak=int((self.scope/'memory.peak').read_text())
        if current<0 or peak<0:raise ValueError('Kernel memory counters differ')
        record=dict(processCount=len(members),summedProcessRssBytes=None if any(value is None for value in rss) else sum(rss),
            cgroupMemoryCurrentBytes=current,cgroupMemoryPeakBytes=peak)
        with self.lock:
            self.samples+=1;self.process_peak=max(self.process_peak,len(members));self.scope_peak=max(self.scope_peak,peak)
            if record['summedProcessRssBytes'] is None:self.incomplete+=1
            else:self.rss_peak=max(self.rss_peak,record['summedProcessRssBytes'])
        return record

    def mark(self,phase):
        if phase not in ('after-public-preparation','after-key-generation','after-provider-readiness','after-provider-close','after-generation-validation','probe-start','probe-complete'):
            raise ValueError('Installed public resource phase required')
        record=self._sample()
        with self.lock:self.phases.append(dict(phase=phase,**record))

    def start(self):
        if self.thread is not None or self.stop_event.is_set():raise ValueError('Resource observer cannot restart')
        def observe():
            while not self.stop_event.is_set():
                try:self._sample()
                except (OSError,ValueError):
                    with self.lock:self.failure=True
                    return
                self.stop_event.wait(self.interval)
        self.thread=threading.Thread(target=observe,name='client-resource-observation',daemon=True);self.thread.start()

    def close(self):
        self.stop_event.set()
        if self.thread is not None:self.thread.join()

    def report(self):
        with self.lock:
            return dict(schema='doppler.client-resource-observation/v1',audience='client-local',sampleIntervalSeconds=self.interval,
                samples=self.samples,incompleteProcessSamples=self.incomplete,observationFailed=self.failure,
                sampledPeakSummedProcessRssBytes=self.rss_peak if self.samples>self.incomplete else None,
                peakObservedProcessCount=self.process_peak,cgroupMemoryPeakBytes=self.scope_peak if self.samples else None,
                phases=[dict(row) for row in self.phases],rssSumDoubleCountsSharedMappings=True,sampledRssPeakMayMissAllocations=True,
                cgroupMemoryIncludesFileCache=True,productionBackendAvailable=False,securityQualified=False)
