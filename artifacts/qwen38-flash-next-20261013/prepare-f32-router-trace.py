"""Reuse the retained F32 source observer for an explicitly named prompt.

Usage: prepare-f32-router-trace.py PRODUCER SOURCE OUT ARTIFACT DATA INPUTS REFERENCE SEQUENCE
Added observations only; the existing observer checks all retained source samples
and router metadata bit-exactly against the independently generated reference.
"""
from pathlib import Path
import hashlib
import json
import sys

if len(sys.argv) != 9:
    raise SystemExit(__doc__)
arguments = sys.argv[1:]
sequence = arguments[-1]
base = Path(__file__).resolve().parents[2] / 'reports/flash-next-qualification-20261107/prepare-router-prefix.py'
original = base.read_bytes()
text = original.decode()
for anchor, replacement in [
    ("'--inputs', str(inputs), '--sequences', 'long'", "'--inputs', str(inputs), '--sequences', " + repr(sequence)),
    ("records = environment['records']['long']", "records = environment['records'][" + repr(sequence) + "]"),
    ("expected = baseline['sequences']['long']", "expected = baseline['sequences'][" + repr(sequence) + "]"),
]:
    if text.count(anchor) != 1:
        raise ValueError('Ambiguous source observer boundary: ' + anchor)
    text = text.replace(anchor, replacement)
sys.argv = [str(base)] + arguments[:-1]
exec(compile(text, str(base) + ':explicit-sequence', 'exec'), {'__name__': '__main__', '__file__': str(base)})
receipt_path = Path(arguments[2]) / 'receipt.json'
receipt = json.loads(receipt_path.read_bytes())
sha = lambda raw: hashlib.sha256(raw).hexdigest()
receipt.update(sequenceId=sequence, sequenceAdapterSha256=sha(Path(__file__).read_bytes()),
               baseObserverSha256=sha(original), adaptedObserverSha256=sha(text.encode()))
receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
