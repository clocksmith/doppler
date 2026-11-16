"""Load explicitly inspected pure-Torch declarations from one hash-pinned official source."""
import ast
import __future__
import hashlib
import json
from pathlib import Path

SPEC=json.loads(Path(__file__).with_name('flash-reference-programs.json').read_text())
SOURCE=json.loads(Path(__file__).with_name('flash-readiness.json').read_text())['operatorReference']


class EagerReferenceOnly:
    def get_interface(self,implementation,reference):
        if implementation!='eager':raise ValueError('Only the inspected eager attention reference is installed')
        return reference


def inspected_transforms(nodes,group):
    transforms=SPEC.get('transforms',{}).get(group,[])
    for transform in transforms:
        if transform!='qsa-scatter-index64/v1':raise ValueError('Unsupported inspected source transform')
        changed=0
        for root in nodes:
            if not isinstance(root,ast.ClassDef) or root.name!='Qwen4ExpTextQSAIndexer':continue
            for node in ast.walk(root):
                if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='scatter'
                        and isinstance(node.func.value,ast.Name) and node.func.value.id=='selected_token_mask'
                        and len(node.args)==3 and not node.keywords and isinstance(node.args[1],ast.Name)
                        and node.args[1].id=='scatter_indices'):
                    node.args[1]=ast.Call(func=ast.Attribute(value=node.args[1],attr='to',ctx=ast.Load()),
                                          args=[ast.Attribute(value=ast.Name(id='torch',ctx=ast.Load()),attr='int64',ctx=ast.Load())],keywords=[])
                    changed+=1
        if changed!=1:raise ValueError('Pinned QSA index promotion site changed')
    return tuple(transforms)


def load_reference(path,group):
    if group not in SPEC['groups']:raise ValueError('Unsupported inspected Flash reference program')
    with Path(path).open('rb') as handle:data=handle.read(1024*1024+1)
    if hashlib.sha256(data).hexdigest()!=SOURCE['files']['modeling_qwen4_exp.py']['sha256']:
        raise ValueError('Pinned official Flash reference source required')
    allowed=set(SPEC['groups'][group])
    nodes=[node for node in ast.parse(data).body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in allowed]
    if {node.name for node in nodes}!=allowed:raise ValueError('Complete inspected Flash declarations required')
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node,(ast.ClassDef,ast.FunctionDef)):continue
            for decorator in node.decorator_list:
                target=decorator.func if isinstance(decorator,ast.Call) else decorator
                if not isinstance(target,ast.Name) or target.id not in SPEC['removedDecorators']:
                    raise ValueError('Unreviewed reference decorator cannot execute')
            node.decorator_list=[]
    transforms=inspected_transforms(nodes,group)
    # No downloaded imports, model-loading code, fused kernels or network dispatch execute.
    # The non-export eager branch uses the source's ordinary Torch triangular solver.
    import math
    import torch
    from torch import nn
    from torch.nn import functional as F
    scope=dict(__name__='doppler_pinned_flash_reference',__referenceTransforms__=transforms,torch=torch,nn=nn,F=F,math=math,
               ACT2FN={'silu':F.silu,'sigmoid':torch.sigmoid},is_torchdynamo_exporting=lambda:False,
               ALL_ATTENTION_FUNCTIONS=EagerReferenceOnly())
    code=compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),
                 'pinned-flash-reference','exec',flags=__future__.annotations.compiler_flag)
    exec(code,scope)
    return scope


def provision_reference(directory):
    """Provision only the two inspected public source files; corrupted cached code rejects."""
    import urllib.request
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    for name,item in SOURCE['files'].items():
        path=directory/name
        if path.exists():data=path.read_bytes()
        else:
            with urllib.request.urlopen(item['url'],timeout=30) as response:data=response.read(1024*1024+1)
        if hashlib.sha256(data).hexdigest()!=item['sha256']:raise ValueError('Pinned reference source mismatch')
        if not path.exists():path.write_bytes(data)
