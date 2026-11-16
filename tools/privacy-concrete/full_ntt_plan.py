"""Separate all-linear Qwen route for the exact-arithmetic native source variant."""
import json
from pathlib import Path
from full_plan import FullPlan,load_qwen_plan
from native_runtime import ntt_parameters,ntt_versions

POLICY_NTT=json.loads(Path(__file__).with_name('full-ntt-policy.json').read_text())


class FullNttPlan(FullPlan):
    @property
    def protocol_id(self):return POLICY_NTT['protocol']


def load_plan(directory):
    return load_qwen_plan(directory,POLICY_NTT,ntt_versions,ntt_parameters(),FullNttPlan)
