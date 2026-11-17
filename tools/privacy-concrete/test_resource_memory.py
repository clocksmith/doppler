from pathlib import Path
import os
import unittest
from unittest.mock import patch
from resource_memory import ResourceMeter,private_cgroup_path,process_rss


class ResourceMemoryTests(unittest.TestCase):
    def fixture(self):
        scope='/sys/fs/cgroup/test.scope'
        values={'/proc/self/cgroup':'0::/test.scope\n',scope+'/cgroup.procs':str(os.getpid()),
            scope+'/memory.current':'6000',scope+'/memory.peak':'8000',
            f'/proc/{os.getpid()}/status':'VmRSS:\t2 kB\n'}
        return values,scope

    def test_initial_exclusive_scope_tracks_parent_and_new_child_without_private_fields(self):
        values,scope=self.fixture()
        with patch.object(Path,'read_text',lambda path:values[str(path)]):
            meter=ResourceMeter();meter.mark('probe-start')
            values[scope+'/cgroup.procs']+=' 999999';values['/proc/999999/status']='VmRSS:\t3 kB\nPRIVATE: never copied'
            meter.mark('after-provider-readiness');meter.close();report=meter.report()
        self.assertEqual(report['sampledPeakSummedProcessRssBytes'],5120)
        self.assertEqual(report['peakObservedProcessCount'],2);self.assertEqual(report['cgroupMemoryPeakBytes'],8000)
        self.assertTrue(report['rssSumDoubleCountsSharedMappings']);self.assertFalse(report['securityQualified'])
        self.assertNotIn('PRIVATE',str(report));self.assertNotIn('999999',str(report))

    def test_missing_process_is_unavailable_not_zero_and_background_failure_is_explicit(self):
        values,scope=self.fixture();values[f'/proc/{os.getpid()}/status']='Name:\tfixture\n'
        with patch.object(Path,'read_text',lambda path:values[str(path)]):
            meter=ResourceMeter();meter.mark('probe-start');self.assertIsNone(meter.report()['sampledPeakSummedProcessRssBytes'])
            with patch.object(meter,'_sample',side_effect=OSError('PRIVATE path')):
                meter.start();meter.thread.join(timeout=1);meter.close()
            self.assertTrue(meter.report()['observationFailed']);self.assertNotIn('PRIVATE',str(meter.report()))

    def test_provider_close_samples_remaining_client_and_retains_combined_peak(self):
        values,scope=self.fixture()
        values[scope+'/cgroup.procs']+=' 999999'
        values['/proc/999999/status']='VmRSS:\t3 kB\nPRIVATE: never copied'
        with patch.object(Path,'read_text',lambda path:values[str(path)]):
            # The real meter starts before the child, then observes its lifetime.
            values[scope+'/cgroup.procs']=str(os.getpid());meter=ResourceMeter()
            values[scope+'/cgroup.procs']+=' 999999';meter.mark('after-provider-readiness')
            values[scope+'/cgroup.procs']=str(os.getpid());values[scope+'/memory.current']='3000'
            del values['/proc/999999/status']
            meter.mark('after-provider-close');meter.close();report=meter.report()
            with self.assertRaises(ValueError):meter.mark('PRIVATE arbitrary phase')
        self.assertEqual([row['phase'] for row in report['phases']],['after-provider-readiness','after-provider-close'])
        self.assertEqual(report['phases'][-1]['processCount'],1)
        self.assertEqual(report['phases'][-1]['summedProcessRssBytes'],2048)
        self.assertEqual(report['phases'][-1]['cgroupMemoryCurrentBytes'],3000)
        self.assertEqual(report['peakObservedProcessCount'],2)
        self.assertEqual(report['sampledPeakSummedProcessRssBytes'],5120)
        self.assertEqual(report['cgroupMemoryPeakBytes'],8000)
        self.assertFalse(report['observationFailed']);self.assertFalse(report['securityQualified'])
        self.assertNotIn('PRIVATE',str(report));self.assertNotIn('999999',str(report))

    def test_unsafe_scope_shared_initial_processes_units_and_phase_reject(self):
        values,scope=self.fixture()
        for path in ('0::/\n','0::/../outside\n','0::relative\n'):
            with self.subTest(path=path),patch.object(Path,'read_text',return_value=path):
                with self.assertRaises(ValueError):private_cgroup_path()
        with patch.object(Path,'read_text',lambda path:values[str(path)]):
            values[scope+'/cgroup.procs']+=' 999999'
            with self.assertRaises(ValueError):ResourceMeter()
            values[scope+'/cgroup.procs']=str(os.getpid());meter=ResourceMeter()
            with self.assertRaises(ValueError):meter.mark('PRIVATE arbitrary phase')
            values[f'/proc/{os.getpid()}/status']='VmRSS: 5 bytes\n'
            with self.assertRaises(ValueError):process_rss(os.getpid())


if __name__=='__main__':unittest.main()
