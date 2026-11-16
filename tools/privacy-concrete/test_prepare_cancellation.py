"""Preparation cancellation control flow; no native cryptography or model loading."""
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

class PreparationCancellationTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
            import transformers
        except ImportError:self.skipTest('Torch/Transformers adapter dependencies unavailable')
        spec=importlib.util.spec_from_file_location('_prepare_cancel_model',Path(__file__).with_name('full_model.py'))
        self.module=importlib.util.module_from_spec(spec)
        native=SimpleNamespace()
        with patch.dict(sys.modules,{'full_bridge':SimpleNamespace(FullBridge=object),'concrete_ml_extensions':native,
            'concrete.ml.quantization.linear_op_glwe_backend':SimpleNamespace(GLWELinearLayerExecutor=object)}):spec.loader.exec_module(self.module)
        native.default_params=lambda:json.dumps(self.module.POLICY['parameters'])
        self.event=threading.Event()

    def test_pre_cancelled_prepare_cannot_resolve_backend(self):
        self.event.set()
        with patch.object(self.module,'qwen_backend') as backend:
            with self.assertRaises(InterruptedError):self.module.prepare('unused','unused',cancelled=self.event)
            backend.assert_not_called()

    def test_cancellation_during_load_stops_before_tokenizer_and_conversion(self):
        model=SimpleNamespace(eval=lambda:self.event.set() or object())
        with patch.object(self.module,'qwen_backend',return_value=({'cpuThreads':1},None,lambda:{},{})), \
             patch.object(self.module,'pinned_source',return_value=nullcontext()) as source, \
             patch.object(self.module.AutoModelForCausalLM,'from_pretrained',return_value=model), \
             patch.object(self.module.AutoTokenizer,'from_pretrained') as tokenizer:
            with self.assertRaises(InterruptedError):self.module.prepare('unused','unused',cancelled=self.event)
            self.assertIs(source.call_args.kwargs['cancelled'],self.event)
            tokenizer.assert_not_called()

if __name__=='__main__':unittest.main()
