"""Separately bound Qwen NTT route with the declared research precision variant."""
import json
from pathlib import Path
from full_plan import FullPlan,load_qwen_plan
from native_runtime import ntt_versions
from research_parameters import component_parameters

POLICY_NTT12=json.loads(Path(__file__).with_name('full-ntt12-policy.json').read_text())


class FullNtt12Plan(FullPlan):
    @property
    def protocol_id(self):return POLICY_NTT12['protocol']


def parameters():return component_parameters(POLICY_NTT12)


def load_plan(directory):return load_qwen_plan(directory,POLICY_NTT12,ntt_versions,parameters(),FullNtt12Plan)
