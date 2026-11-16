"""Installed public fixtures bind both prompt selection and evidence publication."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from public_qwen_fixtures import PUBLIC_FIXTURES,SPEC,DEFAULT_PUBLIC_FIXTURE,public_fixture_prompts
from collect_full_evidence import collect


class PublicQwenFixtureTests(unittest.TestCase):
    def test_one_immutable_registry_preserves_legacy_and_distinct_prompt_identity(self):
        self.assertEqual(PUBLIC_FIXTURES[DEFAULT_PUBLIC_FIXTURE],next(row['prompt'] for row in SPEC['fixtures'] if row['id']=='capital-france'))
        self.assertEqual(len(PUBLIC_FIXTURES),len(SPEC['fixtures']))
        self.assertNotEqual(PUBLIC_FIXTURES['qwen05b-arithmetic'],PUBLIC_FIXTURES[DEFAULT_PUBLIC_FIXTURE])
        with self.assertRaises(TypeError):PUBLIC_FIXTURES['new']='private-canary'

    def test_ambiguous_or_invalid_public_prompt_registries_reject(self):
        candidates=[]
        for key,value in [('prompt',{}),('prompt',''),('publicFixtureId',True),('publicFixtureId','unregistered'),('id','')]:
            candidate=copy.deepcopy(SPEC);candidate['fixtures'][0][key]=value;candidates.append(candidate)
        candidate=copy.deepcopy(SPEC);candidate['fixtures'].append(dict(candidate['fixtures'][0]));candidates.append(candidate)
        candidate=copy.deepcopy(SPEC);candidate['fixtures'][0]['private']='private-canary';candidates.append(candidate)
        for candidate in candidates:
            with self.assertRaises(ValueError):public_fixture_prompts(candidate)

    def test_unknown_fixture_rejects_before_any_artifact_file_read(self):
        for fixture in ['private-request',{},None]:
            with patch('collect_full_evidence.Path',side_effect=AssertionError('artifact read before fixture admission')):
                with self.assertRaises(ValueError):collect('unused','unused','revision',fixture)

    def test_selected_public_fixture_is_bound_to_its_report_and_legacy_default(self):
        report=dict(schema='doppler.full-qwen-hybrid-execution/v1',planId='a'*64,modelId='b'*64,publicFixtureId='qwen05b-arithmetic',
                    qualified=False,productionBackendAvailable=False,linearProjectionCount=169,encryptedSteps=4,encryptedTileCount=1548,
                    matchesFullQuantizedGreedyReference=True,clientLocalOutputText='public arithmetic result',clientLocalOutputTokens=[1,2,3,4])
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'report';source.write_text(json.dumps(report));progress=Path(directory)/'progress'
            self.assertFalse(collect(source,progress,'revision')['completed'])
            actual=collect(source,progress,'revision','qwen05b-arithmetic')
            self.assertTrue(actual['completed']);self.assertEqual(actual['fixtureId'],'qwen05b-arithmetic')
            report['publicFixtureId']='private-request';report['clientLocalOutputText']='private-canary';source.write_text(json.dumps(report))
            rejected=collect(source,progress,'revision','qwen05b-arithmetic')
            self.assertFalse(rejected['completed']);self.assertNotIn('private-canary',json.dumps(rejected))


if __name__=='__main__':unittest.main()
