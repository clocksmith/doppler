"""Trusted-client Flash coefficient preparation and attempt binding.

This is not an installed backend or plan loader. A caller must supply an admitted
bridge whose plan explicitly binds this graph and these exact coefficients.
FullBridge replays every tile. CheckedBridge is accepted only for a plan whose
protocol declares the checked integrity mechanism, and never for a replay plan.
"""
from full_model import FullLinear
from full_bridge import FullBridge
from checked_bridge import CheckedBridge,CHECKED_PROTOCOL_SUFFIX
from full_plan import canonical
from flash_model_graph import ClientFlashGraph
from protocol import require_active


class PreparedFlashGraph:
    def __init__(self,graph,executor,directory,*,quantization_bits=8,cancelled=None):
        require_active(cancelled)
        if type(graph) is not ClientFlashGraph:raise ValueError('Complete client Flash graph required')
        self.graph=graph;self.identity=graph.identity;self.order=tuple(graph.order)
        self._prepared={}
        for name in self.order:
            require_active(cancelled)
            projection=FullLinear(name,graph.projections[name].source,executor,directory,quantization_bits,cancelled=cancelled)
            projection.release_float_reference();self._prepared[name]=projection
        self._definitions=canonical([self._prepared[name].definition for name in self.order])
        self.require_graph();require_active(cancelled)

    def definitions(self):
        import json
        return json.loads(self._definitions)

    def require_graph(self):
        if (self.graph.identity!=self.identity or tuple(self.graph.order)!=self.order
                or canonical([self._prepared[name].definition for name in self.order])!=self._definitions):
            raise ValueError('Prepared Flash graph or coefficient identity changed')
        for source in self.graph.source_checks:source.require_source()

    def reference(self,tokens,state,context,*,cancelled=None):
        self.require_graph();require_active(cancelled)
        projections={name:projection.prepared_copy() for name,projection in self._prepared.items()}
        return self.graph(tokens,state,context,execution='verified-projections',
            verified_client_projection=lambda operation,value:projections[operation['id']](value),cancelled=cancelled)

    def execute(self,tokens,state,context,bridge,secret,parameters):
        projections={}
        try:
            self.require_graph()
            if type(bridge) not in (FullBridge,CheckedBridge):raise ValueError('Admitted integrity-verifying Flash bridge required')
            if (type(bridge) is CheckedBridge)!=bridge.plan.protocol_id.endswith(CHECKED_PROTOCOL_SUFFIX):
                raise ValueError('Bridge integrity mechanism differs from the admitted plan protocol')
            bridge.check_active();definition=bridge.plan.definition()
            if (definition.get('clientGraphId')!=self.identity
                    or canonical(definition.get('projections'))!=self._definitions
                    or context.model_id!=self.identity or context.key_id!=bridge.key_id
                    or context.request_id!=bridge.session or secret is None or parameters is None):
                raise ValueError('Exact Flash graph, coefficients and attempt context required')
            for name,prepared in self._prepared.items():
                projection=prepared.prepared_copy();projection.mode='execute'
                projection.bridge=bridge;projection.secret=secret;projection.parameters=parameters
                projections[name]=projection
            def project(operation,value):
                projection=projections[operation['id']]
                if (operation['inputWidth'],operation['outputWidth'])!=(projection.definition['inputWidth'],projection.definition['outputWidth']):
                    raise ValueError('Prepared Flash projection geometry changed')
                return projection(value)
            return self.graph(tokens,state,context,execution='verified-projections',verified_client_projection=project,cancelled=bridge.cancelled)
        except BaseException:
            state.fail()
            if type(bridge) in (FullBridge,CheckedBridge):
                bridge.admission.fail();bridge._release_projection_inputs()
            raise
        finally:
            for projection in projections.values():projection.secret=projection.bridge=projection.parameters=None
