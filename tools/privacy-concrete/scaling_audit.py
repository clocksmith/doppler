"""Inspect pinned source tensors and conservative arithmetic/memory limits; no performance extrapolation."""
import argparse
import json
import math
from pathlib import Path
from safetensors import safe_open
from profiles import load_profile, validate_arithmetic_capacity
from protocol import POLICY
from real_model import validate_source


def audit(model_dir):
    specification = load_profile('qwen05b-k-projection').specification()
    directory = Path(model_dir)
    validate_source(directory, specification)
    rows, total_parameters = [], 0
    with safe_open(directory / 'model.safetensors', framework='pt', device='cpu') as artifact:
        for name in artifact.keys():
            shape = artifact.get_slice(name).get_shape()
            total_parameters += math.prod(shape)
            if name.startswith('model.layers.') and name.endswith('_proj.weight') and len(shape) == 2:
                output_width, input_width = shape
                try:
                    worst = validate_arithmetic_capacity(input_width, 8)
                    eligible, reason = True, None
                except ValueError as error:
                    worst = input_width * 255 * 128
                    eligible, reason = False, str(error)
                rows.append(dict(name=name, shape=shape, inputWidth=input_width, outputWidth=output_width,
                                 conservativeRawDotBound=worst, withinPinnedArithmeticRange=eligible,
                                 arithmeticBlocker=reason, publicUint64WeightBytes=input_width * output_width * 8,
                                 selectedProjection=name == specification['projection']['name'] + '.weight'))
    flash = json.loads(Path(__file__).with_name('flash-readiness.json').read_text())
    config = flash['config']
    try:
        validate_arithmetic_capacity(config['hidden_size'], 8)
        arithmetic_blocker = None
    except ValueError as error:
        arithmetic_blocker = str(error)
    parameters = sum(flash['publisherRoundedParameterCounts'].values())
    flash.update(conservativeRawDotBound=config['hidden_size'] * 255 * 128, arithmeticBlocker=arithmetic_blocker,
                 weightMemoryEstimate=dict(scope='Rounded publisher counts, excluding vision/overhead; not measured residency',
                                           twoBytesPerParameterGB=parameters * 2 / 1e9,
                                           fourBytesPerParameterGB=parameters * 4 / 1e9),
                 blockers=['Pinned Transformers 4.44.2 does not implement this declared qwen4_exp architecture',
                           'Hidden-width raw dots exceed the pinned conservative 27-bit signed computation budget',
                           'Expert, sparse-indexer and n-gram access patterns need an explicit privacy protocol',
                           'Recurrent/attention state needs validated policy, key and cache isolation',
                           'Full encrypted replay gives no client compute saving or reviewed malicious security',
                           'One selected small-Qwen projection is not a full encrypted model'])
    return dict(schema='doppler.qwen-encrypted-scaling-audit/v1',
                scope='Pinned tensor/config inspection and conservative integer bounds; no unmeasured latency or full-model privacy qualification',
                smallQwenSource=specification['source'], sourceTensorParameterCount=total_parameters,
                decoderProjectionTensorCount=len(rows), decoderUint64PublicWeightBytes=sum(row['publicUint64WeightBytes'] for row in rows),
                decoderProjectionsOutsidePinnedArithmetic=sum(not row['withinPinnedArithmeticRange'] for row in rows),
                pinnedSignedRawDotLimit=2 ** (POLICY['parameters']['bits_reserved_for_computation'] - 1),
                projections=rows, flash=flash)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    print(json.dumps(audit(parser.parse_args().model_dir), indent=2))
