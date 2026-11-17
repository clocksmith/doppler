"""Bounded clear integer oracle for a caller-owned complete Flash graph.

Uses the same global input quantizer, chunked per-output weight quantizer and
integer product as encrypted execution. It never holds a whole FP32 source
projection or a complete coefficient cache. This is full client computation,
not an encrypted backend or a security qualification.
"""
import hashlib
from pathlib import Path
import shutil
import tempfile
import threading
import time
from full_plan import canonical
from protocol import digest,require_active
from flash_text_rebuild_plan import CACHE_BUDGET_BYTES

OUTPUT_CHUNK_ROWS=2048


class BoundedQuantizedReference:
    def __init__(self,graph,directory,*,cancelled=None,output_chunk_rows=OUTPUT_CHUNK_ROWS):
        from flash_model_graph import ClientFlashGraph
        from concrete.ml.quantization.linear_op_glwe_backend import GLWELinearLayerExecutor
        require_active(cancelled)
        if type(graph) is not ClientFlashGraph or type(output_chunk_rows) is not int or not 0<output_chunk_rows<=OUTPUT_CHUNK_ROWS:
            raise ValueError('Complete client graph and bounded reference chunk required')
        self.directory=Path(directory).resolve()
        if not self.directory.is_dir() or any(self.directory.is_relative_to(source.directory.resolve()) for source in graph.source_checks):
            raise ValueError('Existing reference scratch outside immutable model sources required')
        self.graph=graph;self.identity=graph.identity;self.order=tuple(graph.order);self.cancelled=cancelled
        self.output_chunk_rows=output_chunk_rows;self.executor=GLWELinearLayerExecutor(use_dynamic_quantization=True)
        self.identities={};self.metadata_ids={};self.materializations=0;self.peak_coefficient_bytes=0;self.seconds=0.
        self.failed=self.closed=False;self._lock=threading.Lock()

    def _active(self):
        require_active(self.cancelled)
        if self.closed or self.failed or self.graph.identity!=self.identity or tuple(self.graph.order)!=self.order:
            raise ValueError('Reference graph or lifetime changed')

    @staticmethod
    def _metadata_id(projection):
        import torch
        result=hashlib.sha256(canonical(projection.definition))
        for key in ('scale','zero_point','sum_weights','prepared_bias'):
            value=getattr(projection,key)
            if value is None:result.update(canonical(dict(name=key,value=None)));continue
            result.update(canonical(dict(name=key,shape=list(value.shape),dtype=str(value.dtype))))
            with memoryview(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()) as raw:result.update(raw)
        return result.hexdigest()

    def __call__(self,operation,value):
        from full_model import FullLinear
        if not self._lock.acquire(blocking=False):raise ValueError('Reference projection lease is already active')
        began=time.perf_counter()
        try:
            self._active();name=operation['id'];module=self.graph.projections.get(name)
            if module is None or canonical(dict(operation))!=canonical(dict(module.operation)):
                raise ValueError('Exact source-bound reference operation required')
            size=operation['inputWidth']*operation['outputWidth']*2
            if size>CACHE_BUDGET_BYTES or shutil.disk_usage(self.directory).free<size+64*1024**2:
                raise ValueError('Insufficient bounded reference coefficient scratch')
            with tempfile.TemporaryDirectory(dir=self.directory,prefix='flash-reference-projection-') as temporary:
                projection=FullLinear(name,module.source,self.executor,temporary,12,cancelled=self.cancelled,
                    output_chunk_rows=self.output_chunk_rows)
                projection.release_float_reference();self._active();metadata_id=self._metadata_id(projection)
                if (self.identities.get(name,projection.definition['weightId'])!=projection.definition['weightId']
                        or self.metadata_ids.get(name,metadata_id)!=metadata_id):
                    raise ValueError('Reference coefficient or quantizer metadata changed')
                projection.mode='clear-reference';result=projection(value,cancelled=self.cancelled);self._active()
                self.identities[name]=projection.definition['weightId'];self.metadata_ids[name]=metadata_id
                self.materializations+=1;self.peak_coefficient_bytes=max(self.peak_coefficient_bytes,size)
                return result
        except BaseException:
            self.failed=True
            raise
        finally:
            self.seconds+=time.perf_counter()-began;self._lock.release()

    def inventory(self,order):
        self._active()
        if tuple(order)!=self.order or set(self.identities)!=set(self.order):raise ValueError('Complete ordered reference coefficient inventory required')
        ordered=[dict(name=name,weightId=self.identities[name]) for name in self.order]
        return ordered,digest(canonical(ordered))

    def close(self):self.closed=True
