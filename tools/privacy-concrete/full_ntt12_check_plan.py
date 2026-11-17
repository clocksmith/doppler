"""Twelve-bit Qwen NTT route whose client integrity is the installed linear check, not full replay.

The evaluator computation and parameters equal the replay route's. The protocol
and plan identities differ so neither side can substitute one route for the other.
"""
import json
from pathlib import Path
from full_plan import FullPlan,load_qwen_plan
from native_runtime import ntt_versions
from research_parameters import component_parameters

POLICY_NTT12_CHECK=json.loads(Path(__file__).with_name('full-ntt12-check-policy.json').read_text())


class FullNtt12CheckPlan(FullPlan):
    @property
    def protocol_id(self):return POLICY_NTT12_CHECK['protocol']


def parameters():return component_parameters(POLICY_NTT12_CHECK)


def load_plan(directory):return load_qwen_plan(directory,POLICY_NTT12_CHECK,ntt_versions,parameters(),FullNtt12CheckPlan)
