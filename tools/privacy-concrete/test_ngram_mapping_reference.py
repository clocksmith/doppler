"""Inspected official conversion versus local rows; no full table/model qualification."""
import ast
import __future__
import hashlib
import json,ast
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import test_local_ngram_rows as fixtures


@unittest.skipUnless(os.environ.get('DOPPLER_FLASH_REFERENCE')=='1','opt-in pinned source/Torch reference')
class NgramSourceTests(unittest.TestCase):
    def fixture(self):
        import torch
        from flash_reference import load_reference
        root=Path(__file__).parent;directory=Path(os.environ['DOPPLER_FLASH_SOURCE_DIR'])
        spec=json.loads((root/'flash-ngram-storage.json').read_text())
        source=directory/'core_model_loading.py';raw=source.read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),spec['conversionReference']['files'][source.name]['sha256'])
        # Execute only these two inspected declarations, no downloaded imports or loaders.
        nodes=[n for n in ast.parse(raw).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in ['Concatenate','dot_natural_key']]
        self.assertEqual(len(nodes),2)
        scope=dict(torch=torch,ConversionOps=object)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned-ngram-conversion','exec',flags=__future__.annotations.compiler_flag),scope)
        return torch,spec,directory,load_reference,scope

    def test_actual_source_configuration_and_numeric_checkpoint_order_match_inventory(self):
        torch,spec,directory,load,scope=self.fixture()
        keys=[p['name'] for p in spec['parts']]
        self.assertEqual(sorted(reversed(keys),key=scope['dot_natural_key']),keys)
        self.assertEqual(len(keys),spec['ngramConfig']['split_ngram_parts'])
        config=SimpleNamespace(**spec['ngramConfig'])
        cls=load(directory/'modeling_qwen4_exp.py','ple')['Qwen4ExpTextNGramEmbedding']
        with torch.device('meta'):table=cls(config,spec['ngramConfig']['ple_embed_dim'],1,0)
        self.assertEqual(list(table.ngram_embedding.weight.shape),spec['logicalShape'])
        self.assertEqual(sum(p['rows'] for p in spec['parts']),spec['logicalShape'][0])

    def test_official_concatenation_matches_local_cross_shard_row_reads(self):
        from local_ngram_rows import LocalNgramRows
        torch,spec,directory,load,scope=self.fixture()
        with tempfile.TemporaryDirectory() as target:
            fixture,raw=fixtures.LocalRowsTests().fixture(target)
            parts=[]
            for row in fixture['parts']:
                offset=row['firstRow']*4;data=bytearray(raw[offset:offset+row['rows']*4])
                parts.append(torch.frombuffer(data,dtype=torch.bfloat16).reshape(row['rows'],2).clone())
            whole=scope['Concatenate'](dim=0,num_shards_attribute='split_ngram_parts').convert(
                {'ngram_embedding.shard_*.weight':parts},['ngram_embedding.shard_*.weight'],['ngram_embedding.weight'])['ngram_embedding.weight']
            with LocalNgramRows(target,fixture,local_storage_limit_bytes=4096) as storage:
                indices=[5,2,0,4,2]
                actual=torch.frombuffer(bytearray(storage.read_rows(indices,max_output_bytes=20)),dtype=torch.bfloat16).reshape(-1,2)
                self.assertTrue(torch.equal(actual.view(torch.int16),whole[indices].view(torch.int16)))


if __name__=='__main__':unittest.main()
