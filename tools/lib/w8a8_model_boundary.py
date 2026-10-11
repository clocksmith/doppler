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
    layer = contract['layer']
    assert type(layer) is int and 0 <= layer < len(model.model.layers)
    assert len(contract['inputIds']) == 2
    captured, residuals, hooks = {}, {}, []
    residual_layers = contract.get('residualLayers', [])
    assert len(set(residual_layers)) == len(residual_layers)
    assert all(isinstance(index, int) and 0 <= index < len(model.model.layers) for index in residual_layers)
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

    def residual_hook(index):
        def observe(module, operands, result):
            if active:
                assert isinstance(result, torch.Tensor) and list(result.shape) == [1, 1, config.hidden_size]
                residuals[str(index)] = result.detach().flatten().tolist()
        return observe

    try:
        for index in residual_layers:
            hooks.append(model.model.layers[index].register_forward_hook(residual_hook(index)))
        names = contract.get('projections', ['q_proj', 'k_proj', 'v_proj'])
        assert len(set(names)) == len(names)
        for name in names:
            assert name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
            owner = 'self_attn' if name in ['q_proj', 'k_proj', 'v_proj', 'o_proj'] else 'mlp'
            module = model.get_submodule(f'model.layers.{layer}.{owner}.{name}')
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
                       'contract': contract, 'targetLogit': target, 'captured': captured, 'residuals': residuals})
        return report
    finally:
        for handle in hooks:
            handle.remove()
