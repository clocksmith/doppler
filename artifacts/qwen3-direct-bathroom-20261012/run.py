"""Frozen source control: retain existing evaluator, stop completed structural failures."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys

class StructuralRejection(Exception):
    pass

def audit_identities(root, source):
    here = Path(__file__).resolve().parent
    freeze = json.loads((here / 'freeze.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for name, expected in freeze['files'].items():
        assert digest(here / name) == expected, name
    for name, expected in freeze['reusedSources'].items():
        assert digest(root / name) == expected, name
    for name, expected in freeze['sourceIdentity'].items():
        assert digest(source / name) == expected, name
    return freeze

def guard_writer(write):
    def retain_then_check(output, report):
        write(output, report)
        rows = report.get('results', [])
        if rows and rows[-1].get('completionPassed') is False:
            raise StructuralRejection(rows[-1]['id'])
    return retain_then_check

def main():
    parser = argparse.ArgumentParser()
    for name in ['doppler', 'source', 'out']: parser.add_argument('--' + name, required=True)
    args = parser.parse_args(); root = Path(args.doppler).resolve(); source = Path(args.source).resolve()
    freeze = audit_identities(root, source)
    sys.path.insert(0, str(root / 'tools'))
    import lib.source_task_quality as quality
    original_writer = quality.write_quality_report
    quality.write_quality_report = guard_writer(original_writer)
    spec = importlib.util.spec_from_file_location('frozen_source_loader', root / 'tools/source-model-task-quality.py')
    loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)
    settings = argparse.Namespace(source=str(source), suite=str(Path(__file__).with_name('suite.json')),
        out=args.out, threads=4, deployed=None, head_probe_contract=None,
        probe_report=None, probe_case=None, probe_step=None)
    disposition = {'freezeSha256': hashlib.sha256(Path(__file__).with_name('freeze.json').read_bytes()).hexdigest(),
        'sourceLoaderSha256': freeze['reusedSources']['tools/source-model-task-quality.py'],
        'sourceEvaluatorSha256': freeze['reusedSources']['tools/lib/source_task_quality.py'],
        'status': 'completed', 'qualified': False}
    try:
        loader.run(settings)
    except StructuralRejection as error:
        disposition.update(status='structural-rejection', stoppedAfterCase=str(error), remaining='unrun')
    finally:
        quality.write_quality_report = original_writer
    Path(args.out + '.disposition.json').write_text(json.dumps(disposition, indent=2) + '\n')

if __name__ == '__main__': main()
