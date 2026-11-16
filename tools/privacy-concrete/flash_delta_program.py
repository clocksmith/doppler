"""Pure declared projection order for the inspected linear-attention client graph."""
import json
from pathlib import Path

POLICY_DELTA=json.loads(Path(__file__).with_name('flash-delta-policy.json').read_text())
ORDER=tuple(POLICY_DELTA['projectionOrder'])
