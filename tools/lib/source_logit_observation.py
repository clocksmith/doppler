"""Optional frozen offline observations; never supply values to model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np


class SourceLogitObserver:
    def __init__(self, contract, suite_sha256, manifest_sha256, directory):
        assert contract['schema'] == 'doppler.source-logit-observation/v1'
        assert contract['suiteSha256'] == suite_sha256
        assert contract['manifestSha256'] == manifest_sha256
        assert type(contract['vocabSize']) is int and contract['vocabSize'] > 0
        self.contract = json.loads(json.dumps(contract))
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.selected = set()
        for position in contract['positions']:
            assert isinstance(position['caseId'], str) and position['caseId']
            assert type(position['seed']) is int
            for step in position['steps']:
                assert type(step) is int and step >= 0
                key = (position['caseId'], position['seed'], step)
                assert key not in self.selected
                self.selected.add(key)
        assert self.selected
        self.rows = {}

    def observe(self, case_id, seed, step, context, prompt_token_count, logits):
        key = (case_id, seed, step)
        if key not in self.selected:
            return
        assert key not in self.rows, 'A frozen source boundary was observed twice.'
        assert len(context) == prompt_token_count + step
        assert all(type(token) is int and token >= 0 for token in context)
        values = np.asarray(logits)
        assert values.dtype == np.float32 and values.ndim == 1 and values.size == self.contract['vocabSize']
        assert np.isfinite(values).all()
        data = values.astype('<f4', copy=False).tobytes()
        digest = hashlib.sha256(data).hexdigest()
        destination = self.directory / (digest + '.f32')
        if destination.exists():
            assert destination.read_bytes() == data
        else:
            destination.write_bytes(data)
        self.rows[key] = {'caseId': case_id, 'seed': seed, 'step': step,
                          'context': list(context), 'promptTokenCount': prompt_token_count,
                          'logits': {'file': destination.name, 'sha256': digest,
                                     'dtype': 'f32-le', 'elements': int(values.size), 'bytes': len(data)}}

    def write_index(self, path, report):
        results = {(row['id'], row.get('seed')): row for row in report['results']}
        observed = []
        for key, row in sorted(self.rows.items()):
            source = results[(key[0], key[1])]
            tokens = source['tokenIds']
            assert row['context'] == source['inputIds'] + tokens[:key[2]]
            observed.append({**row, 'sourceSelectedToken': tokens[key[2]] if key[2] < len(tokens) else None,
                             'sourceStopReason': source['stopReason'] if key[2] == len(tokens) - 1 else None})
        index = {'schema': 'doppler.source-logit-observation-index/v1', 'qualified': False,
                 'scope': 'Independent selected-boundary source evidence, not reference activation or GPU qualification.',
                 'contract': self.contract, 'vectorDirectory': str(self.directory.resolve()),
                 'producerSha256': report['producerSha256'], 'evaluatorSha256': report['evaluatorSha256'],
                 'observerSha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 'packages': report['packages'], 'arithmetic': report['arithmetic'],
                 'generation': report['settings'], 'source': report['source'], 'deployed': report['deployed'],
                 'requestedPositions': len(self.selected), 'observedPositions': len(observed),
                 'unobserved': [{'caseId': key[0], 'seed': key[1], 'step': key[2]}
                                for key in sorted(self.selected - self.rows.keys())], 'observations': observed}
        Path(path).write_text(json.dumps(index, indent=2) + '\n')
        return index
