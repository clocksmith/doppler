"""Independent, non-mutating source boundary observation for a frozen prefix."""
import hashlib
import json
from pathlib import Path
import torch
from lib.source_task_quality import StoredHalfCache
from lib.w8a8_model_reference import quantize_rows


def replay_w8a8_boundary(model, config, manifest, args, report):
    contract_path = Path(args.boundary_probe)
    contract = json.loads(contract_path.read_text())
    assert contract['schema'] == 'doppler.w8a8-source-boundary/v1'
    assert contract['modelIdentity'] == report['modelIdentity']
    assert contract['execution'] == 'sequential-decode'
    assert contract['layer'] == 0 and len(contract['inputIds']) == 2
    captured, hooks = {}, []
    active = False

    def hook(name):
        def observe(module, operands, result):
            if active:
                values = operands[0].detach().reshape(-1, operands[0].shape[-1]).numpy()
                codes, scales = quantize_rows(values)
                captured[name] = {'shape': list(operands[0].shape), 'input': values.flatten().tolist(),
                                  'codes': codes.flatten().tolist(), 'scales': scales.tolist(),
                                  'output': result.detach().flatten().tolist()}
        return observe

    try:
        for name in ['q_proj', 'k_proj', 'v_proj']:
            module = model.get_submodule('model.layers.0.self_attn.' + name)
            hooks.append(module.register_forward_hook(hook(name)))
        cache = StoredHalfCache(config=config)
        with torch.inference_mode():
            for index, token in enumerate(contract['inputIds']):
                active = index == len(contract['inputIds']) - 1
                result = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                               use_cache=True, logits_to_keep=1)
        target = float(result.logits[0, -1, contract['targetTokenId']])
        assert target == contract['independentExpectedTargetLogit'], (target, contract['independentExpectedTargetLogit'])
        report.update({'schema': 'doppler.w8a8-source-boundary-result/v1',
                       'scope': 'Independently evolved source operands, not a GPU-derived reference',
                       'boundaryToolSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'contractSha256': hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                       'contract': contract, 'targetLogit': target, 'captured': captured})
        return report
    finally:
        for handle in hooks:
            handle.remove()
