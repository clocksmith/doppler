"""Fixed Flash research generation over an already admitted client graph/bridge.

No provisioning, model selection, recipient selection or production admission.
The caller retains source leases and supplies the exact compiled route. All
candidate output stays client-local until the route is closed and integrity and
teacher-forced numerical references pass. References are full client computation,
not useful outsourced integrity or independent security qualification.
"""
from dataclasses import replace
import time
import uuid
from full_plan import canonical
from full_status import ExecutionRejected
from protocol import require_active
from verified_execution import complete_verified_route,VALIDATION_BOUNDARY


def generate(prepared,prompt_tokens,bridge,secret,parameters,context):
    """Return client-local tokens only after a complete, validated fixed route.

    This graph has no padding-mask contract: the prompt length must equal the
    plan's prefill row count, not merely fit below it. EOS does not shorten the
    fixed route. Both references follow encrypted token history after closure;
    float matches are observations, while quantized greedy equality is required.
    The bridge is consumed on every exit; source/preparation ownership stays with
    the caller. No plaintext output or acceptance callback is invoked.
    """
    import torch
    from checked_bridge import CheckedBridge
    from full_bridge import FullBridge
    from flash_prepared_graph import PreparedFlashGraph
    from flash_request_state import ClientContext
    states=[];route_started=False
    owned_bridge=type(bridge) in (FullBridge,CheckedBridge)
    try:
        if not owned_bridge or type(prepared) is not PreparedFlashGraph or type(context) is not ClientContext:
            raise ValueError('Exact prepared Flash graph, admitted bridge and client context required')
        bridge.check_active();prepared.require_graph();definition=bridge.plan.definition()
        steps=definition.get('steps');rows=definition.get('contextBucket');graph=prepared.graph
        if (type(steps) is not int or not 0<steps<=16 or type(rows) is not int or rows<=0
                or bridge.sequence!=1 or bridge.admission.position!=0 or bridge.admission.in_flight
                or bridge.admission.projection is not None
                or type(prompt_tokens) not in (tuple,list) or len(prompt_tokens)!=rows
                or any(type(token) is not int or not 0<=token<graph.config.vocab_size for token in prompt_tokens)
                or rows+steps-1>graph.context_limit or secret is None or parameters is None
                or definition.get('clientGraphId')!=prepared.identity
                or canonical(definition.get('projections'))!=canonical(prepared.definitions())
                or (context.model_id,context.key_id,context.request_id)!=(prepared.identity,bridge.key_id,bridge.session)):
            raise ValueError('Exact fixed Flash prompt, route and attempt binding required')
        # Distinct request/key namespaces prevent reference caches from becoming
        # aliases of actual encrypted-attempt state, including on validation failure.
        references={mode:replace(context,key_id='client-'+mode,request_id=uuid.uuid4().hex)
                    for mode in ('floating','quantized')}
        execute_state=graph.new_state(context);states.append(execute_state)
        reference_states={}
        for mode,reference_context in references.items():
            reference_states[mode]=graph.new_state(reference_context);states.append(reference_states[mode])
        prompt=tuple(prompt_tokens);generated=[];execution_seconds=[];reference_seconds=[]
        errors=[];floating_matches=[]

        def inputs(step):
            return torch.tensor([prompt if step==0 else (generated[step-1],)],dtype=torch.int64)

        def execute(step):
            require_active(bridge.cancelled);began=time.perf_counter()
            logits=prepared.execute(inputs(step),execute_state,context,bridge,secret,parameters)[0,-1].detach().clone()
            execution_seconds.append(time.perf_counter()-began)
            generated.append(int(logits.argmax()))
            return logits

        def validate(step,actual):
            require_active(bridge.cancelled);began=time.perf_counter();value=inputs(step)
            floating=graph(value,reference_states['floating'],references['floating'],execution='public-reference',cancelled=bridge.cancelled)[0,-1]
            quantized=prepared.reference(value,reference_states['quantized'],references['quantized'],cancelled=bridge.cancelled)[0,-1]
            reference_seconds.append(time.perf_counter()-began)
            errors.append(float((actual-quantized).abs().max()))
            if int(actual.argmax())!=int(quantized.argmax()):raise ExecutionRejected('token-reference-mismatch')
            floating_matches.append(int(actual.argmax())==int(floating.argmax()))

        route_started=True
        with torch.no_grad():complete_verified_route(bridge,steps,execute,validate)
        return dict(schema='doppler.fixed-flash-research-generation/v1',clientGraphId=prepared.identity,
            planId=bridge.plan.identity,promptRows=rows,encryptedSteps=steps,clientLocalOutputTokens=list(generated),
            matchesTeacherForcedQuantizedGreedyReference=True,floatTeacherForcedGreedyMatches=floating_matches,
            maxLogitErrors=errors,stepSeconds=execution_seconds,clientReferenceSeconds=reference_seconds,
            validationBoundary=VALIDATION_BOUNDARY,publicCoefficientBytesSent=bridge.public_coefficient_bytes_sent,
            sentBytes=bridge.sent,receivedBytes=bridge.received,productionBackendAvailable=False,securityQualified=False,
            clientComputation='All nonlinear, attention, embedding, cache, quantization and decoding operations; full floating and quantized reference generation after route closure')
    finally:
        try:
            # Once started, complete_verified_route owns closure, including a
            # failing close. Do not mask that error with a second close attempt.
            if owned_bridge and not route_started:bridge.close()
        finally:
            for state in states:state.close()
