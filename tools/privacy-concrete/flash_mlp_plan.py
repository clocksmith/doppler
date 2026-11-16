"""Complete first-layer MLP branch: hyperconnection, all-expert MoE and residual."""
import json
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_component_plan import validate_component,component_versions
from flash_source_scopes import source_spec
from flash_hyper_program import flash_hyper_program
from flash_moe_plan import PROGRAM as MOE_PROGRAM,lowered_operations
from protocol import digest

CONFIG=json.loads(Path(__file__).with_name('flash-mlp-policy.json').read_text())
WEIGHTS=source_spec('mlp-block');SOURCE_ID=digest(canonical(WEIGHTS))
HYPER_PROGRAM=flash_hyper_program()
SEMANTICS=dict(hyperProgramId=HYPER_PROGRAM.identity,moeProgramId=MOE_PROGRAM.identity)
OPERATIONS=tuple(dict(name=row['id'],inputWidth=row['inputWidth'],outputWidth=row['outputWidth'])
                 for row in HYPER_PROGRAM.contract()['operations'])+lowered_operations()


def versions():return component_versions(CONFIG)


class FlashMLPPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    geometry={row['name']:(row['inputWidth'],row['outputWidth']) for row in OPERATIONS}
    return FlashMLPPlan(validate_component(directory,CONFIG,geometry,SEMANTICS,versions(),source=WEIGHTS))
