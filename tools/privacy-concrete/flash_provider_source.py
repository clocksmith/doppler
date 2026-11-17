"""Provider-owned verified public checkpoint, one evictable coefficient projection.

Only installed public tensor bindings are accepted. No downloads, serialized
verification flags, private request buffers or ownership of the raw source.
"""
import copy
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile
import time
from flash_source_scopes import TEXT
from flash_text_plan import projection_bindings
from full_plan import canonical
from protocol import digest,require_active
from public_tile_stream import REBUILD_CONTRACT
from weight_encoding import weight_encoding


def linear_source_spec():
    bindings=projection_bindings(TEXT);names={row['tensor'] for row in bindings.values()}
    result=copy.deepcopy(TEXT)
    result['tensors']={name:copy.deepcopy(TEXT['tensors'][name]) for name in sorted(names)}
    shards={row['shard'] for row in result['tensors'].values()}
    result['shards']={name:copy.deepcopy(TEXT['shards'][name]) for name in sorted(shards)}
    result.update(schema='doppler.flash-provider-linear-source/v1',parentSourceId=digest(canonical(TEXT)),
        tensorCount=len(names),wholeShardBytes=sum(row['bytes'] for row in result['shards'].values()))
    result['logicalTensorBytes']=sum(row['dataOffsets'][1]-row['dataOffsets'][0] for row in result['tensors'].values())
    result['wholeSourceBytes']=result['wholeShardBytes']+sum(row['bytes'] for row in result['files'].values())
    return result


class ProviderCheckpointSource:
    def __init__(self,source_directory,cache_directory,plan,*,cancelled=None):
        from flash_weights import FlashSource
        from flash_streamed_moe import ProjectionSource
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        require_active(cancelled);definition=plan.definition()
        expected=linear_source_spec()
        if (definition.get('publicCoefficientStorage')!=REBUILD_CONTRACT
                or definition.get('providerSourceId')!=digest(canonical(expected))):
            raise ValueError('Installed verified provider checkpoint contract required')
        self.budget=definition['providerCoefficientCacheBudgetBytes']
        self.encoding=weight_encoding(definition['parameters'])
        if type(self.budget) is not int or self.budget<=0 or self.encoding['quantizationBits']!=12:
            raise ValueError('Bounded twelve-bit provider coefficient cache required')
        self.rows={row['name']:dict(row) for row in definition['projections']}
        self.bindings=projection_bindings(TEXT)
        if (len(self.rows)!=len(definition['projections']) or list(self.rows)!=list(self.bindings) or any((row['inputWidth'],row['outputWidth'])!=
                (self.bindings[name]['inputWidth'],self.bindings[name]['outputWidth']) for name,row in self.rows.items())):
            raise ValueError('Exact installed provider projection bindings required')
        self.cache_directory=Path(cache_directory)
        if not self.cache_directory.is_dir():raise ValueError('Existing owned public coefficient scratch directory required')
        if self.cache_directory.resolve().is_relative_to(Path(source_directory).resolve()):
            raise ValueError('Provider scratch must be outside the immutable public source tree')
        self.cancelled=cancelled;self.closed=False;self.materializations=0;self.live_cache_bytes=0;self.peak_cache_bytes=0
        self.materialization_seconds=0.;self.coefficient_bytes_materialized=0
        self.plan_id=plan.identity
        self.source=FlashSource.__new__(FlashSource);self.source.closed=False
        began=time.perf_counter()
        self.source._open(source_directory,expected,cancelled=cancelled) # Whole pinned hashes before readiness/keys.
        self.source_verification_seconds=time.perf_counter()-began
        self.owner=ProjectionSource(self.source);self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        self.source_bytes_verified=expected['wholeSourceBytes']

    def close(self):
        self.closed=True;self.source.closed=True

    def require_tile_source(self,tile):
        require_active(self.cancelled)
        if self.closed:raise ValueError('Provider checkpoint lease is closed')
        self.owner.require_tensor(self.bindings[tile['name']]['tensor'])

    @contextmanager
    def open_projection(self,row,encoding):
        from full_model import FullLinear
        from flash_streamed_moe import SourceLinear
        require_active(self.cancelled)
        if (self.closed or dict(row)!=self.rows.get(row['name']) or dict(encoding)!=self.encoding
                or self.live_cache_bytes):raise ValueError('Exact single owned provider projection lease required')
        self.owner.require_source();size=row['inputWidth']*row['outputWidth']*self.encoding['bytes']
        if size>self.budget or shutil.disk_usage(self.cache_directory).free<size+64*1024*1024:
            raise ValueError('Insufficient provider coefficient scratch capacity')
        binding=self.bindings[row['name']]
        with tempfile.TemporaryDirectory(dir=self.cache_directory,prefix='provider-public-projection-') as temporary:
            source=SourceLinear(self.source,binding['tensor'],row['inputWidth'],row['outputWidth'],
                expert=binding['expert'],rows=binding['rows'],source_owner=self.owner)
            began=time.perf_counter()
            try:prepared=FullLinear(row['name'],source,self.executor,temporary,12,cancelled=self.cancelled,output_chunk_rows=2048)
            finally:self.materialization_seconds+=time.perf_counter()-began
            prepared.release_float_reference()
            if canonical(prepared.definition)!=canonical(dict(row)):raise ValueError('Provider reconstructed coefficient identity differs from client preparation')
            require_active(self.cancelled);self.live_cache_bytes=size
            self.materializations+=1;self.peak_cache_bytes=max(self.peak_cache_bytes,size);self.coefficient_bytes_materialized+=size
            try:
                with prepared.weight_path.open('rb',buffering=0) as handle:yield handle
                self.owner.require_tensor(binding['tensor']);require_active(self.cancelled)
            finally:self.live_cache_bytes=0
