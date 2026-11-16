"""Inspected checkpoint-to-client configuration, without downloaded imports/loaders.

Executes only the pinned text class's literal field defaults and its two local
normalization/architecture methods. The PreTrainedConfig base is deliberately
not executed; this is the bounded eager client-reference configuration contract.
"""
import ast
import __future__
import hashlib
import json
from types import SimpleNamespace
from flash_reference import SOURCE


class ClientConfigBase:
    def __post_init__(self,**kwargs):
        if kwargs:raise ValueError('Unsupported configuration inheritance arguments')


def configuration_declarations(data):
    if hashlib.sha256(data).hexdigest()!=SOURCE['files']['configuration_qwen4_exp.py']['sha256']:
        raise ValueError('Pinned official Flash text configuration required')
    classes=[node for node in ast.parse(data).body if isinstance(node,ast.ClassDef) and node.name=='Qwen4ExpTextConfig']
    if len(classes)!=1:raise ValueError('One inspected text configuration class required')
    fields=[node for node in classes[0].body if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name)]
    defaults={node.target.id:ast.literal_eval(node.value) for node in fields}
    methods=[node for node in classes[0].body if isinstance(node,ast.FunctionDef) and node.name in ['__post_init__','validate_architecture']]
    if len(methods)!=2 or any(method.decorator_list for method in methods):raise ValueError('Inspected configuration methods changed')
    return defaults,fields,methods


def resolve_text_config(raw,configuration_source,*,attention_implementation):
    if (not isinstance(raw,dict) or attention_implementation!='eager'
            or not isinstance(raw.get('layer_types'),list) or not raw['layer_types']
            or any(not isinstance(key,str) or key.startswith('__') or key=='validate_architecture' for key in raw)
            or raw.get('_attn_implementation',attention_implementation)!=attention_implementation):
        raise ValueError('Explicit inspected eager client configuration required')
    defaults,fields,methods=configuration_declarations(configuration_source)
    # Literal defaults plus exactly the inspected local methods; no downloaded
    # imports, class decorators or PreTrainedConfig implementation are executed.
    node=ast.ClassDef(name='InspectedClientConfig',bases=[ast.Name(id='ClientConfigBase',ctx=ast.Load())],
                     keywords=[],body=[*fields,*methods],decorator_list=[])
    scope=dict(ClientConfigBase=ClientConfigBase)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                 'pinned-flash-client-configuration','exec',flags=__future__.annotations.compiler_flag),scope)
    candidate=scope['InspectedClientConfig']()
    candidate.__dict__.update(json.loads(json.dumps(defaults)))
    candidate.__dict__.update(json.loads(json.dumps(raw,allow_nan=False)))
    candidate.__post_init__();candidate.validate_architecture()
    if type(candidate.num_hidden_layers) is not int or len(candidate.layer_types)!=candidate.num_hidden_layers:
        raise ValueError('Explicit complete checkpoint layer schedule required')
    result=dict(vars(candidate),_attn_implementation=attention_implementation)
    return SimpleNamespace(**result)
