"""Pinned Flash-Next tokenizer CPU check: tokenizers runtime against an independent byte-level BPE.

Reads only hash-validated retained files. No model weights or Doppler execution.
Run: python3 -I artifacts/qwen38-flash-next-20261013/check-tokenizer.py
"""
import hashlib
import json
import unicodedata
from pathlib import Path
import regex
import tokenizers
import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parent
binding = json.loads((ROOT / 'source-binding.json').read_text())
for entry in binding['files'] + binding['referenceImplementation']['files']:
    assert hashlib.sha256((ROOT / 'source' / entry['path']).read_bytes()).hexdigest() == entry['sha256']
pinned = {entry['path']: entry for entry in binding['files']}
raw = {name: (ROOT / 'source' / name).read_bytes() for name in ['tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']}
# Upstream-declared identities observed in the Hugging Face HEAD responses at acquisition.
UPSTREAM = {'tokenizer.json': {'xLinkedSize': 12809320,
                               'xLinkedEtagSha256': '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'},
            'tokenizer_config.json': {'etagGitBlobSha1': '5de744b3fca2129d7186979ae47c06be33903243'}}
identity = {
    'tokenizer.json': {**{k: pinned['tokenizer.json'][k] for k in ['url', 'bytes', 'sha256']},
                       'matchesUpstreamDeclared': len(raw['tokenizer.json']) == UPSTREAM['tokenizer.json']['xLinkedSize']
                       and pinned['tokenizer.json']['sha256'] == UPSTREAM['tokenizer.json']['xLinkedEtagSha256']},
    'tokenizer_config.json': {**{k: pinned['tokenizer_config.json'][k] for k in ['url', 'bytes', 'sha256']},
                              'matchesUpstreamDeclared': hashlib.sha1(b'blob %d\0' % len(raw['tokenizer_config.json'])
                                                                      + raw['tokenizer_config.json']).hexdigest()
                              == UPSTREAM['tokenizer_config.json']['etagGitBlobSha1']},
}
spec = json.loads(raw['tokenizer.json'])
tok_config = json.loads(raw['tokenizer_config.json'])
model_config = json.loads((ROOT / 'source/config.json').read_text())['text_config']
generation = json.loads((ROOT / 'source/generation_config.json').read_text())
template = raw['chat_template.jinja'].decode()

# ---- Structure ------------------------------------------------------------------------
model = spec['model']
assert model['type'] == 'BPE' and not model['byte_fallback'] and not model['ignore_merges'] and model['dropout'] is None
assert model['continuing_subword_prefix'] == '' and model['end_of_word_suffix'] == ''
vocab = model['vocab']
assert sorted(vocab.values()) == list(range(len(vocab)))
added = {a['id']: a for a in spec['added_tokens']}
assert sorted(added) == list(range(len(vocab), len(vocab) + len(added)))
assert all(not (a['normalized'] or a['lstrip'] or a['rstrip'] or a['single_word']) for a in added.values())
assert len(vocab) + len(added) <= model_config['vocab_size']
split, byte_level = spec['pre_tokenizer']['pretokenizers']
assert spec['normalizer'] == {'type': 'NFC'} and split['type'] == 'Split' and split['behavior'] == 'Isolated' and not split['invert']
assert byte_level == {'type': 'ByteLevel', 'add_prefix_space': False, 'trim_offsets': False, 'use_regex': False}
assert spec['decoder']['type'] == 'ByteLevel' and spec['post_processor']['type'] == 'ByteLevel'
assert tok_config['pretokenize_regex'] == split['pattern']['Regex'] and tok_config['chat_template'] == template
by_content = {a['content']: i for i, a in added.items()}
special = {
    'config.eos_token_id': model_config['eos_token_id'], 'config.bos_token_id': model_config['bos_token_id'],
    'generation_config.eos_token_id': generation['eos_token_id'], 'generation_config.pad_token_id': generation['pad_token_id'],
    'tokenizer_config.eos_token': [tok_config['eos_token'], by_content[tok_config['eos_token']]],
    'tokenizer_config.pad_token': [tok_config['pad_token'], by_content[tok_config['pad_token']]],
    'tokenizer_config.bos_token': tok_config['bos_token'], 'tokenizer_config.add_bos_token': tok_config['add_bos_token'],
}
names = {i: added[i]['content'] for i in [248044, 248045, 248046, 248068, 248069]}
assert names == {248044: '<|endoftext|>', 248045: '<|im_start|>', 248046: '<|im_end|>', 248068: '<think>', 248069: '</think>'}

# ---- Independent byte-level BPE (pure Python plus the regex module) ---------------------
def bytes_to_unicode():
    keep = list(range(ord('!'), ord('~') + 1)) + list(range(ord('\xa1'), ord('\xac') + 1)) + list(range(ord('\xae'), ord('\xff') + 1))
    codes, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            codes.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, codes)}

BYTE_CHAR = bytes_to_unicode()
CHAR_BYTE = {c: b for b, c in BYTE_CHAR.items()}
RANKS = {tuple(m.split(' ')): rank for rank, m in enumerate(model['merges'])}
PATTERN = regex.compile(split['pattern']['Regex'])
ADDED = sorted(by_content, key=len, reverse=True)

def bpe(word):
    parts = list(word)
    while len(parts) > 1:
        pairs = [(RANKS.get((a, b), float('inf')), i) for i, (a, b) in enumerate(zip(parts, parts[1:]))]
        rank, i = min(pairs)
        if rank == float('inf'):
            break
        parts[i:i + 2] = [parts[i] + parts[i + 1]]
    return [vocab[p] for p in parts]

def independent_encode(text):
    ids, i, plain = [], 0, ''
    def flush(segment):
        segment = unicodedata.normalize('NFC', segment)
        pieces, last = [], 0
        for match in PATTERN.finditer(segment):
            pieces += [segment[last:match.start()], match.group()] if match.start() > last else [match.group()]
            last = match.end()
        pieces += [segment[last:]] if last < len(segment) else []
        for piece in pieces:
            ids.extend(bpe(''.join(BYTE_CHAR[b] for b in piece.encode('utf-8'))))
    while i < len(text):
        token = next((t for t in ADDED if text.startswith(t, i)), None)
        if token is None:
            plain += text[i]
            i += 1
            continue
        flush(plain)
        plain = ''
        ids.append(by_content[token])
        i += len(token)
    flush(plain)
    return ids

INVERSE = {v: k for k, v in vocab.items()}
def independent_decode(ids):
    out, pending = [], []
    for i in ids:
        if i in added:
            out.append(bytes(pending).decode('utf-8', errors='replace'))
            out.append(added[i]['content'])
            pending = []
        else:
            pending.extend(CHAR_BYTE[c] for c in INVERSE[i])
    return ''.join(out) + bytes(pending).decode('utf-8', errors='replace')

# ---- Fixed strings and chat template --------------------------------------------------
runtime = tokenizers.Tokenizer.from_str(raw['tokenizer.json'].decode())
TEXTS = ['Hello, world!', "Don't stop: it'S fine, they'LL see.", 'Numbers 1234567890 and 3.14159',
         '  leading spaces\tand tabs\n\n\nnewlines  \n', '中文分词测试，标点符号。', 'Ünïcödé façade naïve',
         'é combining acute', 'emoji 🦊 👩🏽‍💻 ✓', 'def f(x):\n    return x**2  # comment',
         '<|endoftext|>', '<|im_start|>user\nhi<|im_end|>\n', '<think>\nreasoning\n</think>\n\nanswer',
         'text<|endoftext|>more<|im_end|>']

def raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)

environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols])
environment.globals['raise_exception'] = raise_exception
compiled = environment.from_string(template)
conversation = [{'role': 'system', 'content': 'You are concise.'}, {'role': 'user', 'content': 'What is 2+2?'},
                {'role': 'assistant', 'content': 'Four.'}, {'role': 'user', 'content': 'Thanks!'}]
CHATS = {'thinkingDefault': compiled.render(messages=conversation, add_generation_prompt=True),
         'thinkingDisabled': compiled.render(messages=conversation, add_generation_prompt=True, enable_thinking=False)}
assert CHATS['thinkingDefault'].endswith('<|im_start|>assistant\n<think>\n')
assert CHATS['thinkingDisabled'].endswith('<|im_start|>assistant\n<think>\n\n</think>\n\n')

def case(text):
    ids = runtime.encode(text, add_special_tokens=True).ids
    decoded = runtime.decode(ids, skip_special_tokens=False)
    return {'text': text, 'ids': ids, 'idsSha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'independentIdsEqual': ids == independent_encode(text),
            'decoded': decoded, 'roundTrip': decoded == unicodedata.normalize('NFC', text),
            'independentDecodeEqual': independent_decode(ids) == decoded,
            'nfcChangedInput': unicodedata.normalize('NFC', text) != text,
            'specialTokens': {names.get(i, added[i]['content']): ids.count(i) for i in sorted(set(ids)) if i in added}}

cases = [case(text) for text in TEXTS]
chats = {name: case(text) for name, text in CHATS.items()}
for name, result in chats.items():
    text = CHATS[name]
    # system, user, assistant, user and the generation prompt open turns; four turns close.
    assert result['specialTokens']['<|im_start|>'] == text.count('<|im_start|>') == 5
    assert result['specialTokens']['<|im_end|>'] == text.count('<|im_end|>') == 4
    assert result['specialTokens']['<think>'] == text.count('<think>') == 2
assert CHATS['thinkingDefault'].startswith('<|im_start|>system\nReasoning effort is set to xhigh.')
assert '<|im_start|>assistant\n<think>\n\n</think>\n\nFour.<|im_end|>' in CHATS['thinkingDefault']
eos = {'endoftext': runtime.encode('<|endoftext|>').ids, 'imEnd': runtime.encode('<|im_end|>').ids,
       'decodeSkipSpecial': runtime.decode(runtime.encode('a<|im_end|>b<|endoftext|>').ids, skip_special_tokens=True)}
assert eos['endoftext'] == [248044] and eos['imEnd'] == [248046] and eos['decodeSkipSpecial'] == 'ab'
checks = {
    'identityMatchesUpstreamDeclared': all(v['matchesUpstreamDeclared'] for v in identity.values()),
    'independentEncodeEqual': all(c['independentIdsEqual'] for c in cases + list(chats.values())),
    'independentDecodeEqual': all(c['independentDecodeEqual'] for c in cases + list(chats.values())),
    'roundTripAfterNfc': all(c['roundTrip'] for c in cases + list(chats.values())),
    'singleIdSpecialTokens': True,  # asserted above for EOS, im_start, im_end and think markers
}
report = {
    'schema': 'doppler.qwen38-tokenizer-check/v1', 'checkpoint': binding['revision'], 'surface': 'CPU',
    'scope': 'Pinned tokenizer files; tokenizers runtime against an independent byte-level BPE; no model weights or Doppler execution',
    'libraries': {'tokenizers': tokenizers.__version__, 'regex': regex.__version__, 'jinja2': jinja2.__version__},
    'files': identity,
    'structure': {'model': 'BPE', 'baseVocabulary': len(vocab), 'merges': len(model['merges']), 'addedTokens': len(added),
                  'addedIdRange': [min(added), max(added)], 'configVocabularySize': model_config['vocab_size'],
                  'unusedEmbeddingRows': model_config['vocab_size'] - len(vocab) - len(added),
                  'normalizer': 'NFC', 'preTokenizer': 'regex Split (Isolated) then ByteLevel without prefix space',
                  'embeddedChatTemplateEqualsJinjaFile': True},
    'specialTokens': special,
    'eosNote': 'config.json eos/bos is 248044 <|endoftext|> (the n-gram EOS boundary); generation stops on [248046 <|im_end|>, 248044]; '
               'tokenizer_config eos_token is <|im_end|>; no BOS is added.',
    'eos': eos,
    'cases': cases,
    'chatTemplate': {'renderer': f'jinja2 {jinja2.__version__} ImmutableSandboxedEnvironment with trim_blocks, lstrip_blocks, '
                                 'loopcontrols and raise_exception; parity with the transformers renderer is not established',
                     'messages': conversation, 'rendered': CHATS, 'results': chats,
                     'note': 'thinking defaults on: the template injects the xhigh reasoning-effort instruction into the system turn, '
                             'renders prior assistant turns with an empty think block, and opens the generation prompt with <think>'},
    'checks': checks,
    'notEstablished': ['Learned embedding rows or model quality', 'Doppler tokenizer implementation parity',
                       'Tool-call, vision and audio template branches', 'Inputs outside these fixed strings'],
}
report['passed'] = all(checks.values())
(ROOT / 'tokenizer-reference.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'passed': report['passed'], 'checks': checks}))
if not report['passed']:
    raise SystemExit(1)
