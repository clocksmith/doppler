"""Trusted-client Flash coefficient preparation and attempt binding.

This is not an installed backend or plan loader. A caller must supply an admitted
bridge whose plan explicitly binds this graph and these exact coefficients.
FullBridge replays every tile. CheckedBridge is accepted only for a plan whose
protocol declares the checked integrity mechanism, and never for a replay plan.
"""
from contextlib import contextmanager
from pathlib import Path
import tempfile
import threading
import time
import torch
from full_model import FullLinear
from full_bridge import FullBridge
from checked_bridge import CheckedBridge,CHECKED_PROTOCOL_SUFFIX
from full_plan import canonical
from flash_model_graph import ClientFlashGraph
from protocol import require_active
from public_weights import PreparedPublicProjection,PreparedPublicTiles
from flash_streamed_moe import SourceLinear
from projection_identity import ProjectionIdentity

EPHEMERAL_STORAGE='doppler.ephemeral-public-projection/v1'


class PreparedFlashGraph:
    def __init__(self,graph,executor,directory,*,quantization_bits=8,cancelled=None,output_chunk_rows=2048,storage='retained'):
        require_active(cancelled)
        if type(graph) is not ClientFlashGraph:raise ValueError('Complete client Flash graph required')
        if storage not in ('retained',EPHEMERAL_STORAGE):raise ValueError('Unsupported public coefficient storage contract')
        self.graph=graph;self.identity=graph.identity;self.order=tuple(graph.order)
        self.storage=storage;self.directory=Path(directory);self.executor=executor
        self._cache_lease=threading.Lock()
        self.quantization_bits=quantization_bits;self.output_chunk_rows=output_chunk_rows
        self.preparation_seconds=0.;self.rematerialization_seconds=0.;self.rematerializations=0
        self._prepared={};self._proofs=[];self._source_snapshots={}
        for name in self.order:
            require_active(cancelled)
            source=graph.projections[name].source
            if storage==EPHEMERAL_STORAGE and not isinstance(source,SourceLinear):self._source_snapshots[name]=self._source_digest(source)
            began=time.perf_counter()
            with self._cache_scope() as cache:
                projection=FullLinear(name,source,executor,cache,quantization_bits,
                    cancelled=cancelled,output_chunk_rows=output_chunk_rows)
                if storage==EPHEMERAL_STORAGE:
                    definition=dict(parameters=dict(weightEncoding=projection.encoding['id'],quantizationBits=quantization_bits),
                        inputTileWidth=2048,outputTileWidth=2048)
                    with projection.weight_path.open('rb') as handle:
                        self._proofs.append(PreparedPublicProjection(projection.definition,definition,handle,cancelled=cancelled))
            self.preparation_seconds+=time.perf_counter()-began
            projection.release_float_reference()
            self._prepared[name]=ProjectionIdentity.from_projection(projection) if storage==EPHEMERAL_STORAGE else projection
        self._definitions=canonical([self._prepared[name].definition for name in self.order])
        self.require_graph();require_active(cancelled)

    @staticmethod
    def _source_digest(source):
        import hashlib
        result=hashlib.sha256()
        for name,value in source.state_dict().items():
            result.update(canonical(dict(name=name,shape=list(value.shape),dtype=str(value.dtype))))
            with memoryview(value.detach().contiguous().reshape(-1).view(torch.uint8).numpy()) as payload:result.update(payload)
        return result.hexdigest()

    @contextmanager
    def _cache_scope(self):
        if self.storage=='retained':yield self.directory
        else:
            if not self._cache_lease.acquire(blocking=False):raise ValueError('Client public projection lease is already active')
            try:
                with tempfile.TemporaryDirectory(dir=self.directory,prefix='flash-public-projection-') as temporary:yield Path(temporary)
            finally:self._cache_lease.release()

    def public_tile_digests(self):
        if self.storage!=EPHEMERAL_STORAGE:raise ValueError('Compiled evictable public projection proofs required')
        return dict(entry for proof in self._proofs for entry in proof.entries)

    def bind_public_tiles(self,plan):
        self.require_graph()
        if self.storage!=EPHEMERAL_STORAGE:raise ValueError('Compiled evictable public projection proofs required')
        return PreparedPublicTiles.from_projections(plan,self._proofs)

    @contextmanager
    def projection_lease(self,name,*,cancelled=None):
        """One public matrix, released on return/failure; no request/key state retained."""
        require_active(cancelled);prepared=self._prepared[name]
        if self.storage=='retained':yield prepared.prepared_copy();return
        source=self.graph.projections[name].source
        with self._cache_scope() as cache:
            began=time.perf_counter()
            try:
                actual=FullLinear(name,source,self.executor,cache,self.quantization_bits,
                    cancelled=cancelled,output_chunk_rows=self.output_chunk_rows)
            finally:self.rematerialization_seconds+=time.perf_counter()-began
            self.rematerializations+=1
            actual.release_float_reference()
            prepared.require_projection(actual)
            actual.mode='clear-reference'
            try:yield actual
            finally:actual.secret=actual.bridge=actual.parameters=None

    def definitions(self):
        import json
        return json.loads(self._definitions)

    def require_graph(self):
        if (self.graph.identity!=self.identity or tuple(self.graph.order)!=self.order
                or canonical([self._prepared[name].definition for name in self.order])!=self._definitions):
            raise ValueError('Prepared Flash graph or coefficient identity changed')
        for source in self.graph.source_checks:source.require_source()
        for name,identity in self._source_snapshots.items():
            if self._source_digest(self.graph.projections[name].source)!=identity:raise ValueError('Prepared public source parameter changed')

    def reference(self,tokens,state,context,*,cancelled=None):
        self.require_graph();require_active(cancelled)
        def project(operation,value):
            require_active(cancelled)
            with self.projection_lease(operation['id'],cancelled=cancelled) as projection:return projection(value,cancelled=cancelled)
        return self.graph(tokens,state,context,execution='verified-projections',
            verified_client_projection=project,cancelled=cancelled)

    def execute(self,tokens,state,context,bridge,secret,parameters):
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
            if getattr(bridge,'streams_public_coefficients',False) or getattr(bridge,'rebuilds_public_coefficients',False):
                import json
                if (not callable(getattr(parameters,'serialize',None))
                        or canonical(json.loads(parameters.serialize()))!=canonical(definition['parameters']['parameters'])):
                    raise ValueError('Actual encryption parameters differ from the streamed plan')
            def project(operation,value):
                bridge.check_active()
                with self.projection_lease(operation['id'],cancelled=bridge.cancelled) as projection:
                    if (operation['inputWidth'],operation['outputWidth'])!=(projection.definition['inputWidth'],projection.definition['outputWidth']):
                        raise ValueError('Prepared Flash projection geometry changed')
                    projection.mode='execute';projection.bridge=bridge;projection.secret=secret;projection.parameters=parameters
                    try:return projection(value)
                    finally:projection.secret=projection.bridge=projection.parameters=None
            return self.graph(tokens,state,context,execution='verified-projections',verified_client_projection=project,cancelled=bridge.cancelled)
        except BaseException:
            state.fail()
            if type(bridge) in (FullBridge,CheckedBridge):
                bridge.admission.fail();bridge._release_projection_inputs()
            raise
