"""Actual source mutation/cancellation and per-call cache ownership without native tokenization."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import flash_tokenizer


class ClientTokenizerTests(unittest.TestCase):
    def fixture(self,directory):
        source=dict(flash_tokenizer.SOURCE)
        payloads={'config.json':json.dumps(dict(text_config=dict(vocab_size=source['vocabSize']))).encode(),
            'tokenizer_config.json':json.dumps(dict(tokenizer_class=source['tokenizerClass'])).encode(),'tokenizer.json':b'public-tokenizer'}
        for name,data in payloads.items():(Path(directory)/name).write_bytes(data)
        source['files']={name:dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for name,data in payloads.items()}
        return source

    def test_missing_sources_invalid_inputs_and_cancellation_reject_before_materialization(self):
        event=threading.Event();event.set()
        with tempfile.TemporaryDirectory() as directory,patch.object(flash_tokenizer,'_load_tokenizer') as load:
            with self.assertRaises(FileNotFoundError):flash_tokenizer.encode(directory,'private-canary',8)
            with self.assertRaises(InterruptedError):flash_tokenizer.encode(directory,'private-canary',8,cancelled=event)
            for prompt,limit in [('',8),('private-canary',True),('\ud800',8)]:
                with self.assertRaises(ValueError):flash_tokenizer.encode(directory,prompt,limit)
            load.assert_not_called()

    def test_unknown_implementation_and_private_exception_text_cannot_cross_the_boundary(self):
        with patch.object(flash_tokenizer,'version',return_value='unqualified'):
            with self.assertRaises(ValueError):flash_tokenizer._load_tokenizer('/unused')
        with tempfile.TemporaryDirectory() as directory:
            source=self.fixture(directory)
            def reject(*args,**kwargs):raise RuntimeError('private-input-canary')
            with patch.object(flash_tokenizer,'SOURCE',source),patch.object(flash_tokenizer,'_load_tokenizer',return_value=SimpleNamespace(is_fast=True,encode=reject)):
                with self.assertRaises(ValueError) as rejected:flash_tokenizer.encode(directory,'private-input-canary',8)
                self.assertNotIn('private-input-canary',str(rejected.exception))

    def test_private_tokenizer_instances_are_never_shared_between_calls(self):
        created=[]
        def fresh(directory):
            result=SimpleNamespace(is_fast=True,encode=lambda *args,**kwargs:[1,2],decode=lambda *args,**kwargs:'public-result');created.append(result);return result
        with tempfile.TemporaryDirectory() as directory:
            source=self.fixture(directory)
            with patch.object(flash_tokenizer,'SOURCE',source),patch.object(flash_tokenizer,'_load_tokenizer',side_effect=fresh):
                self.assertEqual(flash_tokenizer.encode(directory,'private-one',8),(1,2));self.assertEqual(flash_tokenizer.encode(directory,'private-two',8),(1,2))
                self.assertEqual(flash_tokenizer.decode(directory,(1,2),8),'public-result')
            self.assertEqual(len(created),3);self.assertEqual(len({id(value) for value in created}),3)

    def test_source_mutation_and_cancel_during_encode_never_release_private_token_ids(self):
        for kind in ['mutation','cancel']:
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as directory:
                source=self.fixture(directory);event=threading.Event()
                def encode(*args,**kwargs):
                    if kind=='mutation':(Path(directory)/'tokenizer.json').write_bytes(b'changed-tokenizer')
                    else:event.set()
                    return [1,2]
                with patch.object(flash_tokenizer,'SOURCE',source),patch.object(flash_tokenizer,'_load_tokenizer',return_value=SimpleNamespace(is_fast=True,encode=encode)):
                    with self.assertRaises(ValueError if kind=='mutation' else InterruptedError):flash_tokenizer.encode(directory,'private-canary',8,cancelled=event)


if __name__=='__main__':unittest.main()
