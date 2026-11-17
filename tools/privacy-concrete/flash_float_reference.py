"""Bounded FP32 source oracle; no whole-projection materialization or networking.

Only output channels are chunked; every input channel remains in each product.
BLAS column blocking may change rounding. This is a declared floating quality
observation, not the quantized acceptance oracle or a security qualification.
"""
from contextlib import nullcontext
import threading
from full_plan import canonical,ranges
from protocol import require_active

REFERENCE_ID='bounded-output-columns-fp32/v1'
MAX_OUTPUT_CHUNK_ROWS=2048


class BoundedFloatingReference:
    def __init__(self,graph,*,cancelled=None,output_chunk_rows=MAX_OUTPUT_CHUNK_ROWS):
        from flash_model_graph import ClientFlashGraph
        if type(graph) is not ClientFlashGraph or type(output_chunk_rows) is not int or not 0<output_chunk_rows<=MAX_OUTPUT_CHUNK_ROWS:
            raise ValueError('Exact complete client graph and bounded floating oracle required')
        self.graph=graph;self.identity=graph.identity;self.order=tuple(graph.order)
        self.cancelled=cancelled;self.output_chunk_rows=output_chunk_rows;self.closed=self.failed=False
        self.peak_fp32_weight_bytes=0;self.operations=0;self._lock=threading.Lock()

    def _active(self):
        require_active(self.cancelled)
        if self.closed or self.failed or self.graph.identity!=self.identity or tuple(self.graph.order)!=self.order:
            raise ValueError('Floating oracle graph or lifetime changed')

    def __call__(self,operation,value):
        import torch
        from torch.nn.functional import linear
        from flash_streamed_moe import SourceLinear
        if not self._lock.acquire(blocking=False):raise ValueError('Floating source lease already active')
        try:
            self._active();module=self.graph.projections.get(operation['id'])
            if module is None or canonical(dict(operation))!=canonical(dict(module.operation)):
                raise ValueError('Exact source-bound floating operation required')
            source=module.source;outputs=[]
            for begin,end in ranges(operation['outputWidth'],self.output_chunk_rows):
                self._active()
                lease=(source.materialized_weight(output_rows=(begin,end),cancelled=self.cancelled)
                    if isinstance(source,SourceLinear) else nullcontext(source.weight[begin:end]))
                with lease as weight:
                    if weight.dtype!=torch.float32:raise ValueError('Inspected FP32 source reference required')
                    self.peak_fp32_weight_bytes=max(self.peak_fp32_weight_bytes,weight.numel()*weight.element_size())
                    bias=None if source.bias is None else source.bias[begin:end]
                    outputs.append(linear(value,weight,bias))
                del weight
            self._active();self.operations+=1
            return torch.cat(outputs,dim=-1)
        except BaseException:
            self.failed=True
            raise
        finally:self._lock.release()

    def close(self):self.closed=True
