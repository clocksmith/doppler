"""Test-only public geometry substitution; actual native encryption is untouched.

Normal backend loaders do not admit this synthetic scope. Both executable test
processes construct the same fixed public fixture from installed geometry.
"""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from flash_compute_budget import workload
from protocol import digest


@contextmanager
def fixture(source_directory):
    import numpy as np
    import torch
    import flash_text_rebuild5_plan as stream
    from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
    from test_checkpoint_fixture import checkpoint_fixture
    inventory=workload();generator=np.random.default_rng(73);matrices={};weights={};rows=[];geometry={}
    if inventory['steps']!=4 or inventory['prefillRows']!=5:raise ValueError('Explicit installed four-step, five-row resource fixture required')
    executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
    context=SimpleNamespace(input_quantizers=[SimpleNamespace(quant_options=SimpleNamespace(n_bits=12))])
    for shape in inventory['shapes']:
        width,height=shape['inputWidth'],shape['outputWidth'];name=f'public-shape-{width}-{height}'
        weight=torch.from_numpy(generator.integers(-11,12,size=(height,width),dtype=np.int64)).float().to(torch.bfloat16).float()
        quantized,*metadata=executor._per_channel_weight_quantization(weight.numpy().T,context,torch.device('cpu'))
        if any(value.dtype!=torch.float32 for value in metadata):raise ValueError('Installed float32 metadata footprint changed')
        matrix=quantized.numpy().astype(np.int64);weights[name]=weight;matrices[name]=matrix;geometry[name]=(width,height)
        rows.append(dict(name=name,inputWidth=width,outputWidth=height,weightId=digest(matrix.astype('<i2').tobytes())))
    source_directory=Path(source_directory);source_directory.mkdir(mode=0o700)
    specification,bindings=checkpoint_fixture(source_directory,weights)
    with ExitStack() as owned:
        owned.enter_context(patch('flash_provider_source.linear_source_spec',return_value=specification))
        owned.enter_context(patch('flash_provider_source.projection_bindings',return_value=bindings))
        owned.enter_context(patch.object(stream,'linear_source_spec',return_value=specification))
        owned.enter_context(patch.object(stream,'GEOMETRY',geometry))
        owned.enter_context(patch.object(stream,'TEXT',{'schema':'synthetic-installed-flash-geometries/v1'}))
        previous_threads=torch.get_num_threads();torch.set_num_threads(stream.CONFIG['cpuThreads'])
        try:
            yield SimpleNamespace(stream=stream,matrices=matrices,rows=rows,inventory=inventory,generator=generator,
                source_directory=source_directory,geometry=geometry)
        finally:torch.set_num_threads(previous_threads)
