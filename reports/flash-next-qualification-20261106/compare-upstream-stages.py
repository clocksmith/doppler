"""Compare observed stages with independent CPU traces for layers 0..N.

Usage: python3 -P compare-upstream-stages.py <gpu.json> <out.json> [<first-layer>] <cpu-layer.json> ...
Add --manifest <manifest.json> for older traces without explicit stream geometry.
No model acceptance thresholds are applied or changed by this diagnostic.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

def error_metrics(actual, reference):
    """Descriptive scale-aware errors; nonzero error / zero RMS is undefined."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    assert actual.shape == reference.shape and actual.size
    assert np.isfinite(actual).all() and np.isfinite(reference).all()
    actual64, reference64 = actual.astype(np.float64), reference.astype(np.float64)
    difference = actual64 - reference64
    maximum = float(np.abs(difference).max())
    error_rms = float(np.sqrt(np.mean(difference ** 2)))
    reference_rms = float(np.sqrt(np.mean(reference64 ** 2)))
    def relative(error):
        return error / reference_rms if reference_rms else (0.0 if error == 0 else None)
    return dict(referenceRms=reference_rms, referenceMaxAbsolute=float(np.abs(reference64).max()),
                observedRms=float(np.sqrt(np.mean(actual64 ** 2))),
                maxAbsoluteDifference=maximum, rmsDifference=error_rms,
                maxErrorOverReferenceRms=relative(maximum), rmsErrorOverReferenceRms=relative(error_rms),
                relativeErrorDefined=reference_rms != 0 or maximum == 0)


def stage_metrics(actual, reference, positions, stream_count=None, stream_width=None):
    assert actual.dtype == reference.dtype == np.dtype('<f4')
    assert actual.ndim == reference.ndim == 2 and len(positions) == actual.shape[0]
    result = error_metrics(actual, reference)
    result['differingF32Values'] = int(np.count_nonzero(actual.view(np.uint32) != reference.view(np.uint32)))
    result['positionMetrics'] = []
    for row, position in enumerate(positions):
        metrics = dict(position=position, **error_metrics(actual[row], reference[row]), streams=[])
        if stream_count is not None:
            assert stream_width is not None and actual.shape[1] == stream_count * stream_width
            for stream in range(stream_count):
                start, end = stream * stream_width, (stream + 1) * stream_width
                metrics['streams'].append(dict(stream=stream, **error_metrics(actual[row, start:end], reference[row, start:end])))
        result['positionMetrics'].append(metrics)
    return result


def main():
    args = sys.argv[1:]
    manifest_path = None
    if '--manifest' in args:
        index = args.index('--manifest')
        manifest_path = Path(args[index + 1])
        del args[index:index + 2]
    gpu_path, output = map(Path, args[:2])
    first = int(args[2]) if args[2].isdigit() else 0
    source_paths = list(map(Path, args[3:] if args[2].isdigit() else args[2:]))
    sha = lambda b: hashlib.sha256(b).hexdigest()
    gpu_bytes = gpu_path.read_bytes()
    gpu = json.loads(gpu_bytes)
    assert gpu['diagnosticCompleted'] and not gpu['gpuErrors'] and not gpu['cleanupFailures']
    geometry = gpu.get('hyperConnectionGeometry')
    if manifest_path:
        manifest_bytes = manifest_path.read_bytes()
        assert sha(manifest_bytes) == gpu['manifestSha256']
        declared = json.loads(manifest_bytes)['inference']['hyperConnections']
        assert geometry is None or geometry == declared
        geometry = declared
    assert geometry, 'Trace geometry or a pinned --manifest is required for stream metrics'
    stream_count, stream_width = geometry['streamCount'], geometry['streamWidth']
    report = dict(schema='doppler.upstream-stage-comparison/v2', hyperConnectionGeometry=geometry,
                  scope='Observed stage slices and reference-RMS-relative errors; not model qualification',
                  normalModelQualified=False, gpuTraceSha256=sha(gpu_bytes),
                  scriptSha256=sha(Path(__file__).read_bytes()), sourceTraces=[], rows=[])
    for layer, source_path in enumerate(source_paths, start=first):
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        assert source['diagnosticCompleted'] and source['manifestSha256'] == gpu['manifestSha256']
        assert source['inputsSha256'] == gpu['inputsSha256']
        assert source.get('layer', layer) == layer
        report['sourceTraces'].append(dict(layer=layer, file=str(source_path), sha256=sha(source_bytes)))
        for observed in [r for r in gpu['tensors'] if r['layer'] == layer]:
            role, stage = observed['role'], observed['stage']
            if role == 'linear' and stage == 'attn.post_input_norm':
                role, stage = 'attn_hc', 'branchInput'
            expected = next((r for r in source['tensors'] if (r['role'], r['stage']) == (role, stage)), None)
            assert expected, f'No independent source observation for layer {layer}: {role}/{stage}'
            raw = Path(observed['file']).read_bytes()
            assert sha(raw) == observed['sha256']
            actual = np.frombuffer(raw, dtype='<f4').reshape(observed['shape'])
            reference_path = source_path.parent / expected['file']
            reference_bytes = reference_path.read_bytes()
            assert sha(reference_bytes) == expected['sha256']
            reference = np.load(reference_path, allow_pickle=False)
            positions = expected['positions']
            assert positions
            if observed.get('positions') is not None:
                assert observed['positions'] == positions
            else:
                actual = actual[positions]
            assert actual.dtype == reference.dtype == np.dtype('<f4')
            grouped = role in ('attn_hc', 'mlp_hc') and stage in ('input', 'norm', 'up', 'mix')
            metrics = stage_metrics(actual, reference, positions,
                                    stream_count if grouped else None, stream_width if grouped else None)
            report['rows'].append(dict(layer=layer, role=observed['role'], stage=observed['stage'], positions=positions,
                gpuSha256=observed['sha256'], sourceSha256=expected['sha256'], **metrics))
    assert report['rows']
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(rows=len(report['rows']), output=str(output))))


if __name__ == '__main__':
    main()
