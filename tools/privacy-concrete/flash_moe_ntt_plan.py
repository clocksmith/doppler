"""Separately bound all-expert exact-NTT route; stock crypto parameters, no production qualification."""
import json
from pathlib import Path
from full_plan import FullPlan
from flash_component_plan import component_versions
from flash_moe_plan import load_moe_plan

CONFIG_NTT=json.loads(Path(__file__).with_name('flash-moe-ntt-policy.json').read_text())


class FlashMoENttPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG_NTT['protocol']


def versions():return component_versions(CONFIG_NTT)


def load_plan(directory):return load_moe_plan(directory,CONFIG_NTT,versions,FlashMoENttPlan)
