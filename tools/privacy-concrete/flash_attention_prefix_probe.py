"""Fixed public checkpoint prefix measurement, never full-route acceptance.

The complete existing backend is admitted before keys/input disclosure. Its first
attention prefix runs unchanged, then the incomplete route is terminally closed.
All numerical comparisons and output evidence follow provider closure. This is
neither a generation backend nor a reviewed integrity/security qualification.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import time
import uuid
from full_plan import canonical
from full_status import ExecutionRejected
from protocol import require_active
from flash_text_rebuild5_plan import CONFIG as BACKEND

_raw=json.loads(Path(__file__).with_name('flash-attention-prefix-probe.json').read_text())
POLICY_ID=hashlib.sha256(canonical(_raw)).hexdigest()
if any(isinstance(value,dict) or isinstance(value,list) and any(type(item) not in (str,int) for item in value) for value in _raw.values()):
    raise ValueError('Flat installed public prefix contract required')
POLICY=MappingProxyType({key:tuple(value) if isinstance(value,list) else value for key,value in _raw.items()})
del _raw


def require_prefix(plan,tokens,*,policy=POLICY,expected_policy_id=None):
    if expected_policy_id is not None and hashlib.sha256(canonical(dict(policy))).hexdigest()!=expected_policy_id:
        raise ValueError('Prefix policy differs from the attempt binding')
    definition=plan.definition();names=policy['projections']
    if (type(tokens) not in (list,tuple) or any(type(token) is not int for token in tokens)
            or tuple(tokens)!=policy['promptTokens'] or definition.get('schema')!=BACKEND['planSchema']
            or plan.protocol_id!=BACKEND['protocol']
            or definition.get('contextBucket')!=len(tokens)
            or tuple(row['name'] for row in definition.get('projections',[])[:len(names)])!=names):
        raise ValueError('Installed public fixture and exact first-attention prefix required')
    schedule=plan.schedule()
    prefix=tuple(row for row in schedule if row['name'] in names)
    if (len(prefix)!=policy['encryptedTiles'] or tuple(schedule[:len(prefix)])!=prefix
            or tuple(dict.fromkeys(row['name'] for row in prefix))!=tuple(names)):
        raise ValueError('Complete fixed first-attention tile prefix required')
    return len(prefix)


class _PrefixReached(BaseException):
    """Trusted local control boundary, carrying no tensor or exception detail."""


def _capture(graph,context,forward,cancelled,policy):
    import torch
    state=graph.new_state(context);captured=[];marker=_PrefixReached()
    def boundary(module,args):
        require_active(cancelled)
        if (captured or state.position!=len(policy['projections']) or len(args)!=1
                or not isinstance(args[0],torch.Tensor) or args[0].dtype!=torch.float32
                or args[0].device.type!='cpu' or tuple(args[0].shape)!=policy['outputShape']
                or not torch.isfinite(args[0]).all()):
            raise ValueError('Exact finite completed attention-prefix output required')
        captured.append(args[0].detach().clone())
        raise marker
    handle=None
    try:
        handle=graph.layers[0].mlp_hyper_connection.register_forward_pre_hook(boundary)
        try:
            with torch.no_grad():forward(state)
        except _PrefixReached as reached:
            if reached is not marker or len(captured)!=1:raise ValueError('Foreign prefix boundary') from None
            return captured[0]
        raise ValueError('Installed attention boundary was not reached')
    finally:
        if handle is not None:handle.remove()
        state.close()


def execute_attention_prefix(prepared,tokens,bridge,secret,parameters,context,*,floating_reference,expected_policy_id):
    import torch
    from checked_bridge import CheckedBridge
    from flash_prepared_graph import PreparedFlashGraph
    from flash_request_state import ClientContext
    from flash_float_reference import BoundedFloatingReference
    owned=type(bridge) is CheckedBridge;started=time.perf_counter()
    policy,policy_id=POLICY,POLICY_ID
    if (type(policy) is not MappingProxyType or hashlib.sha256(canonical(dict(policy))).hexdigest()!=policy_id
            or expected_policy_id!=policy_id):
        if owned:bridge.close()
        raise ValueError('Installed prefix policy changed without its identity')
    try:
        if (not owned or type(prepared) is not PreparedFlashGraph or type(context) is not ClientContext
                or type(floating_reference) is not BoundedFloatingReference
                or floating_reference.graph is not prepared.graph or floating_reference.cancelled is not bridge.cancelled):
            raise ValueError('Exact admitted graph, checked bridge and owned floating oracle required')
        count=require_prefix(bridge.plan,tokens,policy=policy)
        if (tuple(prepared.graph.order[:len(policy['projections'])])!=tuple(policy['projections'])
                or bridge.sequence!=1 or bridge.admission.position!=0 or bridge.admission.in_flight
                or bridge.admission.projection is not None or secret is None or parameters is None):
            raise ValueError('Fresh fixed prefix attempt required')
        bridge.check_active();value=torch.tensor([tokens],dtype=torch.int64)
        actual=_capture(prepared.graph,context,
            lambda state:prepared.execute(value,state,context,bridge,secret,parameters),bridge.cancelled,policy)
        if (bridge.sequence!=count+1 or bridge.admission.position!=count or bridge.admission.in_flight
                or bridge.admission.projection is not None):
            raise ValueError('Actual encrypted prefix differs from the fixed route')
    finally:
        if owned:bridge.close()
    # Deliberate prefix truncation does not complete or authorize the full route.
    if bridge.child.poll() is None or not bridge.admission.failed:
        raise ValueError('Incomplete prefix route must be closed and unusable')
    require_active(bridge.cancelled);bridge.require_integrity()
    encrypted_seconds=time.perf_counter()-started;reference_started=time.perf_counter()
    references={}
    for mode in ('quantized','floating'):
        owner=replace(context,key_id='client-prefix-'+mode,request_id=uuid.uuid4().hex)
        def forward(state):
            if mode=='quantized':return prepared.reference(value,state,owner,cancelled=bridge.cancelled)
            return prepared.graph(value,state,owner,execution='verified-projections',
                verified_client_projection=floating_reference,cancelled=bridge.cancelled)
        references[mode]=_capture(prepared.graph,owner,forward,bridge.cancelled,policy)
    require_active(bridge.cancelled)
    error=float((actual-references['quantized']).abs().max())
    if not torch.allclose(actual,references['quantized'],atol=policy['maxAbsoluteError'],rtol=policy['maxRelativeError']):
        raise ExecutionRejected('component-reference-mismatch')
    return dict(schema='doppler.actual-flash-attention-prefix-execution/v1',probePolicyId=policy_id,
        boundary=policy['boundary'],publicFixtureId=policy['publicFixtureId'],outputShape=list(actual.shape),
        observedOutputSha256=hashlib.sha256(actual.contiguous().numpy().tobytes()).hexdigest(),
        linearProjectionCount=len(policy['projections']),encryptedTileCount=count,
        matchesQuantizedPrefixReference=True,maxEncryptedVsQuantizedError=error,
        maxQuantizedVsFloatingError=float((references['quantized']-references['floating']).abs().max()),
        absoluteTolerance=policy['maxAbsoluteError'],relativeTolerance=policy['maxRelativeError'],
        encryptedPrefixAndCloseSeconds=encrypted_seconds,clientPrefixReferenceSeconds=time.perf_counter()-reference_started,
        validationBoundary='after-prefix-route-close/v1',providerClosedBeforeValidation=True,
        fullRouteCompleted=False,fullFlashModelExecuted=False,generatedTokens=0,
        sentBytes=bridge.sent,receivedBytes=bridge.received,publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,
        evaluatorRoundTripSeconds=bridge.remote_seconds,sampledReplaySeconds=bridge.replay_seconds,
        integrity=bridge.integrity_report(),productionBackendAvailable=False,securityQualified=False,
        clientComputation='Embeddings, attention/nonlinear/cache operations, quantization, checks and complete floating/quantized prefix references')
