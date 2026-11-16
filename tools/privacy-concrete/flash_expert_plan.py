"""Exact fixed-public-expert component admission. It cannot identify or admit the full Flash model."""
import json
from pathlib import Path
from full_plan import FullPlan
from flash_component_plan import SOURCE_ID,WEIGHTS,component_versions

POLICY_EXPERT=json.loads(Path(__file__).with_name('flash-expert-policy.json').read_text())
GEOMETRY={'expert/0/gate_up':(2560,1280),'expert/0/down':(640,2560)}


class FlashExpertPlan(FullPlan):
    @property
    def protocol_id(self):return POLICY_EXPERT['protocol']


def versions():return component_versions(POLICY_EXPERT)

def load_plan(directory):
    from flash_component_plan import validate_component
    raw=validate_component(directory,POLICY_EXPERT,GEOMETRY,dict(expertIndex=POLICY_EXPERT['expertIndex']),versions())
    return FlashExpertPlan(raw)
