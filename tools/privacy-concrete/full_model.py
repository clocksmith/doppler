"""Complete Qwen all-linear hybrid backend. Native GLWE operations; client nonlinearities and replay."""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
import concrete_ml_extensions as ext
from full_bridge import FullBridge
from public_weights import mapped_public_matrix,store_public_matrix
from full_status import ExecutionRejected, failure_event
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY
from public_qwen_fixtures import PUBLIC_FIXTURES
from full_plan import canonical,ranges,tiles
from protocol import POLICY, digest, require_active
from plan_backends import DEFAULT_PROFILE,QWEN_PROFILES,qwen_backend
from source_snapshot import pinned_source
from integer_reference import integer_dot
from weight_encoding import encoding_for_bits
from profiles import validate_declared_arithmetic_capacity


class FullLinear(nn.Module):
    """Quantize the entire input/weight once; tile only integer encrypted products, never scales."""
    def __init__(self, name, source, executor, directory, quantization_bits=POLICY['quantizationBits']):
        super().__init__()
        self.name, self.remote, self.executor = name, source, executor
        self.encoding=encoding_for_bits(quantization_bits)
        self.mode, self.bridge, self.secret, self.parameters = 'float-reference', None, None, None
        self.context = SimpleNamespace(input_quantizers=[SimpleNamespace(quant_options=SimpleNamespace(n_bits=quantization_bits))])
        weight = source.weight.detach().numpy().T
        quantized, self.scale, self.zero_point, self.sum_weights = executor._per_channel_weight_quantization(weight,self.context,torch.device('cpu'))
        data = np.ascontiguousarray(quantized.numpy(),dtype=self.encoding['dtype'])
        if not np.array_equal(data,quantized.numpy()):raise ValueError('Quantized public coefficient truncation')
        self.weight_path,self.definition=store_public_matrix(directory,name,source.in_features,source.out_features,
            dict(weightEncoding=self.encoding['id'],quantizationBits=quantization_bits),data.tobytes())

    def release_float_reference(self):
        """Drop owned FP32 weights after preparation; no implicit source reload."""
        if self.bridge is not None or self.secret is not None or self.parameters is not None:
            raise ValueError('Cannot release a bound encrypted projection')
        if self.remote is not None:
            self.prepared_bias=None if self.remote.bias is None else self.remote.bias.detach().clone()
            self.remote=None

    def prepared_copy(self):
        """Fresh attempt bindings over trusted client preparation, never wire metadata."""
        if self.remote is not None or self.bridge is not None or self.secret is not None or self.parameters is not None:
            raise ValueError('Unbound released public projection required')
        result=FullLinear.__new__(FullLinear);nn.Module.__init__(result)
        result.name=self.name;result.executor=self.executor;result.weight_path=self.weight_path
        result.encoding=copy.deepcopy(self.encoding);result.definition=copy.deepcopy(self.definition)
        result.context=copy.deepcopy(self.context)
        result.scale=self.scale.clone();result.zero_point=self.zero_point.clone();result.sum_weights=self.sum_weights.clone()
        result.prepared_bias=None if self.prepared_bias is None else self.prepared_bias.clone()
        result.remote=None;result.mode='clear-reference';result.bridge=result.secret=result.parameters=None
        return result

    @contextmanager
    def coefficient_matrix(self):
        prepared=self.remote is None
        if prepared and self.weight_path.is_symlink():raise ValueError('Regular public coefficient cache required')
        with mapped_public_matrix(self.weight_path,(self.definition['inputWidth'],self.definition['outputWidth']),self.encoding['dtype']) as matrix:
            if prepared:
                # Reuse skips preparation's store_public_matrix verification. Bind
                # the exact borrowed bytes before either clear or encrypted use.
                hasher=hashlib.sha256()
                with memoryview(matrix).cast('B') as payload:
                    for begin in range(0,len(payload),1024*1024):
                        if self.bridge is not None:self.bridge.check_active()
                        hasher.update(payload[begin:begin+1024*1024])
                if hasher.hexdigest()!=self.definition['weightId']:raise ValueError('Prepared public coefficient identity changed')
            yield matrix

    def forward(self,value):
        if self.mode == 'float-reference':
            if self.remote is None:raise ValueError('Floating reference weights were explicitly released')
            return self.remote(value)
        if self.mode not in ['clear-reference','execute'] or not torch.isfinite(value).all() or value.shape[-1] != self.definition['inputWidth']:
            raise ValueError('Unsupported full-model projection execution')
        q_input, scale, zero, original_shape = self.executor._dynamic_input_quantization(value,self.context,False)
        with self.coefficient_matrix() as weights:
            if self.mode == 'clear-reference':
                raw = integer_dot(q_input,weights)
            else:
                if self.bridge is None or self.secret is None:
                    raise ValueError('No bound encrypted evaluator')
                with self.bridge.projection(self.name,self.definition['inputWidth'],self.definition['outputWidth'],q_input.shape[0]):
                    integer_input = np.ascontiguousarray(q_input.long().numpy(),dtype=np.uint64)
                    if integer_input.ndim != 2 or not np.all(integer_input<2**self.encoding['quantizationBits']):
                        raise ValueError('Only one private request per key context is admitted')
                    geometry=self.bridge.execution_geometry
                    encrypted = {}
                    for begin,end in ranges(self.definition['inputWidth'],geometry['inputTileWidth']):
                        validate_declared_arithmetic_capacity(end-begin,self.encoding['quantizationBits'],geometry['arithmeticBits'])
                        encrypted[begin] = ext.encrypt_matrix(pkey=self.secret,crypto_params=self.parameters,
                                                             data=np.ascontiguousarray(integer_input[:,begin:end]))
                    result = np.zeros((integer_input.shape[0],self.definition['outputWidth']),dtype=np.int64)
                    for tile in tiles(self.definition,geometry['inputTileWidth'],geometry['outputTileWidth']):
                        matrix = np.ascontiguousarray(weights[tile['inputBegin']:tile['inputEnd'],tile['outputBegin']:tile['outputEnd']],dtype=np.int64).astype(np.uint64)
                        answer = self.bridge.evaluate(encrypted[tile['inputBegin']],matrix,tile)
                        self.bridge.check_active()
                        decoded = ext.decrypt_matrix(answer,self.secret,self.parameters,tile['outputEnd']-tile['outputBegin']).astype(np.int64)
                        # Exact signed integer accumulation precedes one global correction/dequantization.
                        result[:,tile['outputBegin']:tile['outputEnd']] += decoded
                    raw = torch.from_numpy(result)
        output = self.executor._apply_correction_and_dequantize(raw,q_input,zero,self.zero_point,
                                                               self.sum_weights,self.definition['inputWidth'],scale,self.scale)
        output = output.view(*original_shape[:-1],self.definition['outputWidth'])
        bias=self.remote.bias if self.remote is not None else self.prepared_bias
        if bias is not None:
            output = output+bias
        return output


def prepare(model_dir,cache_dir,backend_profile=DEFAULT_PROFILE):
    settings,plan_loader,installed_versions,parameters=qwen_backend(backend_profile)
    specification = json.loads(Path(__file__).with_name('qwen05b.json').read_text())
    versions=installed_versions()
    if json.loads(ext.default_params()) != POLICY['parameters']:
        raise ValueError('Unsupported encrypted parameter change')
    with pinned_source(model_dir,specification['source']['files']):
        torch.set_num_threads(settings['cpuThreads'])
        model = AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,use_safetensors=True,
                                                  trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager').eval()
        tokenizer = AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
        executor = GLWELinearLayerExecutor(use_dynamic_quantization=True)
        if parameters['parameters']!=POLICY['parameters']:
            executor.glwe_crypto_params=ext.MatmulCryptoParameters.deserialize(json.dumps(parameters['parameters']))
        directory = Path(cache_dir);directory.mkdir(parents=True,exist_ok=True)
        projections = []
        for name,module in list(model.named_modules()):
            if isinstance(module,nn.Linear):
                replacement = FullLinear(name,module,executor,directory,settings['quantizationBits'])
                parent,separator,leaf = name.rpartition('.')
                setattr(model.get_submodule(parent) if separator else model,leaf,replacement)
                projections.append(replacement)
        definitions = [projection.definition for projection in projections]
        source = specification['source']
        plan = dict(schema=settings['planSchema'],protocol=settings['protocol'],source=source,
                    modelId=digest(canonical(dict(source=source,projections=definitions))),
                    productionBackendAvailable=False,implementationVersions=versions,
                    **{key:settings[key] for key in ['steps','contextBucket','inputTileWidth','outputTileWidth','operationTimeoutSeconds','cpuThreads']},
                    parameters=parameters,projections=definitions)
        (directory/'plan.json').write_bytes(canonical(plan))
        frozen = plan_loader(directory)
        # Trace every real linear operator under public input, before keys/private input.
        visited=[]
        handles=[projection.register_forward_pre_hook(lambda module,args:visited.append(module.name)) for projection in projections]
        try:
            with torch.no_grad():
                model(input_ids=torch.arange(8).reshape(1,8),use_cache=False)
        finally:
            for handle in handles:handle.remove()
        if visited != [row['name'] for row in definitions]:
            raise ValueError('Actual full-model route differs from its declared all-linear order')
        return model,tokenizer,executor,projections,frozen


def model_step(model,projections,cache,tokens,original_count,step,mode,context_bucket,pad_token_id):
    """One shared prefill/decode contract for encrypted execution and clear quality checks."""
    if (mode not in ['float-reference','clear-reference','execute'] or type(step) is not int or step<0
            or not 1<=original_count<=context_bucket or len(tokens)!=original_count+step):
        raise ValueError('Exact model step and token history required')
    for projection in projections:projection.mode=mode
    if step==0:
        if cache.get_seq_length()!=0:raise ValueError('Fresh model prefill cache required')
        values=tokens+[pad_token_id]*(context_bucket-len(tokens))
        args=dict(input_ids=torch.tensor([values]),attention_mask=torch.tensor([[1]*len(tokens)+[0]*(context_bucket-len(tokens))]))
        index=len(tokens)-1
    else:
        length=cache.get_seq_length()
        if length!=original_count+step-1:raise ValueError('Model cache does not match this token history')
        args=dict(input_ids=torch.tensor([[tokens[-1]]]),attention_mask=torch.ones((1,length+1),dtype=torch.long));index=0
    began=time.perf_counter();output=model(**args,past_key_values=cache,use_cache=True)
    logits=output.logits[0,index]
    if step==0:cache.crop(original_count)
    return logits,(time.perf_counter()-began)*1000


def run(model_dir,cache_dir,prompt,progress=lambda event:None,cancelled=None,deployment=None,backend_profile=DEFAULT_PROFILE):
    settings,_,installed_versions,_=qwen_backend(backend_profile)
    require_active(cancelled)
    if os.environ.get('RAYON_NUM_THREADS') != str(settings['cpuThreads']):
        raise ValueError('The full-model process must declare the installed CPU thread budget before native initialization')
    started=time.perf_counter()
    model,tokenizer,executor,projections,plan=prepare(model_dir,cache_dir,backend_profile)
    require_active(cancelled)
    if deployment is not None:deployment.bind(plan)
    tokens=tokenizer.encode(prompt)
    if not 1<=len(tokens)<=settings['contextBucket']:
        raise ValueError('Prompt exceeds full-model fixed prefill bucket')
    original_count=len(tokens)
    parameters=executor.glwe_crypto_params
    secret,key=ext.cpu_create_private_key(parameters)
    bridge=None
    try:
        bridge=FullBridge(cache_dir,plan,key,cancelled=cancelled,deployment=deployment,backend_profile=None if backend_profile==DEFAULT_PROFILE else backend_profile,observe=lambda kind,value:progress(dict(event=kind,**value)))
        for projection in projections:
            projection.bridge,projection.secret,projection.parameters=bridge,secret,parameters
        caches={mode:DynamicCache() for mode in ['float-reference','clear-reference','execute']}
        setup_ms=(time.perf_counter()-started)*1000
        step_ms=[];reference_ms=[];errors=[];float_matches=[]
        def execute(step):
            require_active(cancelled)
            logits,duration=model_step(model,projections,caches['execute'],tokens,original_count,step,'execute',settings['contextBucket'],151643)
            step_ms.append(duration);tokens.append(int(logits.argmax()))
            progress(dict(event='step-executed',step=step+1,encryptedTiles=bridge.sequence-1))
            return logits.detach().clone()
        def validate(step,actual):
            # The exact encrypted token history is teacher-forced into both local
            # references, preserving the old checks without remote acceptance feedback.
            history=tokens[:original_count+step];logits={}
            for mode in ['float-reference','clear-reference']:
                require_active(cancelled)
                logits[mode],duration=model_step(model,projections,caches[mode],history,original_count,step,mode,settings['contextBucket'],151643)
                if mode=='clear-reference':reference_ms.append(duration)
            errors.append(float((actual-logits['clear-reference']).abs().max()))
            next_token=int(actual.argmax())
            if next_token!=int(logits['clear-reference'].argmax()):raise ExecutionRejected('token-reference-mismatch')
            float_matches.append(next_token==int(logits['float-reference'].argmax()))
        with torch.no_grad():complete_verified_route(bridge,settings['steps'],execute,validate)
        return dict(schema='doppler.full-qwen-hybrid-execution/v1',qualified=False,productionBackendAvailable=False,validationBoundary=VALIDATION_BOUNDARY,
                    descriptor=dict(schema='doppler.execution-backend/v1',id=settings['backendId'],version=1,kind='cryptographic',
                                    modelIdentities=['sha256:'+plan.definition()['modelId']],operations=['generate'],protocol=bridge.plan.protocol_id,
                                    inputRepresentation='opaque',outputRepresentation='opaque'),
                    backendProfile=backend_profile,protocol=bridge.plan.protocol_id,executionRouteId=bridge.plan.identity,transport='local-process' if deployment is None else 'ssh-ed25519-pinned',
                    planId=plan.identity,modelId=plan.definition()['modelId'],linearProjectionCount=len(projections),
                    encryptedTileCount=bridge.sequence-1,encryptedSteps=settings['steps'],clientLocalOutputTokens=tokens[original_count:],
                    clientLocalOutputText=tokenizer.decode(tokens[original_count:]),matchesFullQuantizedGreedyReference=True,
                    floatGreedyMatches=float_matches,maxLogitErrors=errors,setupMs=setup_ms,stepMs=step_ms,quantizedReferenceMs=reference_ms,
                    elapsedMs=(time.perf_counter()-started)*1000,implementationVersions=installed_versions(),clientEncryptedReplayMs=bridge.replay_seconds*1000,
                    evaluatorRoundTripMs=bridge.remote_seconds*1000,timingSemantics='round-trip-includes-client-replay/v1',sentBytes=bridge.sent,receivedBytes=bridge.received,
                    evaluationKeyBytes=bridge.key_bytes,clientPeakRss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    clientLifetimePeakRssBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system()=='Darwin' else 1024),
                    clientComputation='All nonlinear/attention/embedding/cache/quantization/decoding plus full encrypted replay; not all operations homomorphically evaluated',
                    integrity='Full deterministic replay; no client compute saving or independently reviewed malicious security')
    finally:
        try:
            if bridge is not None:bridge.close()
        finally:
            for projection in projections:projection.secret=projection.bridge=projection.parameters=None
            secret=key=None


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True);parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--backend-profile',choices=QWEN_PROFILES,default=DEFAULT_PROFILE)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--public-fixture',choices=tuple(PUBLIC_FIXTURES))
    parser.add_argument('--ssh-deployment')
    parser.add_argument('--ssh-identity-file')
    parser.add_argument('--allowed-operator',action='append')
    args=parser.parse_args()
    deployment=None
    if any([args.ssh_deployment,args.ssh_identity_file,args.allowed_operator]):
        if not all([args.ssh_deployment,args.ssh_identity_file,args.allowed_operator]):
            raise ValueError('SSH requires an explicit deployment, client identity and requester operator allowlist')
        from ssh_route import SSHRoute, SSHDeployment
        deployment=SSHDeployment(SSHRoute.from_file(args.ssh_deployment),args.ssh_identity_file,tuple(args.allowed_operator))
    if args.prepare_only:
        _,_,_,_,plan=prepare(args.model_dir,args.cache_dir,args.backend_profile)
        print(json.dumps(dict(planId=plan.identity,linearProjections=len(plan.definition()['projections']),encryptedTilesPerStep=len(plan.schedule()),available=False)))
    else:
        prompt=PUBLIC_FIXTURES[args.public_fixture] if args.public_fixture else sys.stdin.buffer.read(4097).decode('utf-8')
        if len(prompt.encode())>4096:raise ValueError('Bounded client prompt required')
        began=time.perf_counter()
        try:
            report=run(args.model_dir,args.cache_dir,prompt,deployment=deployment,backend_profile=args.backend_profile,progress=lambda event:print(json.dumps(event),file=sys.stderr,flush=True))
        except BaseException as error:
            print(json.dumps(failure_event(error,(time.perf_counter()-began)*1000)),file=sys.stderr,flush=True)
            raise
        if args.public_fixture:report['publicFixtureId']=args.public_fixture
        print(json.dumps(report,indent=2))
