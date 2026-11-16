"""Installed all-expert Flash MoE research route; client routing, no selected-only calls."""
import json
from pathlib import Path
from full_plan import FullPlan
from flash_component_plan import SOURCE_ID,WEIGHTS,validate_component,component_versions
from oblivious_experts import flash_moe_program

CONFIG=json.loads(Path(__file__).with_name('flash-moe-policy.json').read_text())
PROGRAM=flash_moe_program()


def versions():return component_versions(CONFIG)

def lowered_operations(program=PROGRAM):
    """Public lowering combines adjacent gate/up calls, never branches on selected experts."""
    result=[]
    for operation in program.contract()['operations']:
        name=operation['id']
        if name.startswith('expert/') and name.endswith('/up'):continue
        if name.startswith('expert/') and name.endswith('/gate'):
            result.append(dict(name=name[:-4]+'gate_up',inputWidth=operation['inputWidth'],outputWidth=2*operation['outputWidth']))
        else:result.append(dict(name=name,inputWidth=operation['inputWidth'],outputWidth=operation['outputWidth']))
    return tuple(result)


class FlashMoEPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_moe_plan(directory,policy,installed,plan_type):
    geometry={row['name']:(row['inputWidth'],row['outputWidth']) for row in lowered_operations()}
    raw=validate_component(directory,policy,geometry,dict(programId=PROGRAM.identity),installed())
    return plan_type(raw)


def load_plan(directory):return load_moe_plan(directory,CONFIG,versions,FlashMoEPlan)
