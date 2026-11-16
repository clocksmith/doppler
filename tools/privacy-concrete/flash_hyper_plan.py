"""Exact four-step actual-weight hyperconnection research admission."""
import json
from pathlib import Path
from full_plan import FullPlan,canonical
from flash_component_plan import validate_component,component_versions
from flash_source_scopes import source_spec
from flash_hyper_program import flash_hyper_program
from protocol import digest

CONFIG=json.loads(Path(__file__).with_name('flash-hyper-policy.json').read_text())
WEIGHTS=source_spec('mlp-hyper')
SOURCE_ID=digest(canonical(WEIGHTS))
PROGRAM=flash_hyper_program()
GEOMETRY={row['id']:(row['inputWidth'],row['outputWidth']) for row in PROGRAM.contract()['operations']}


def versions():return component_versions(CONFIG)


class FlashHyperPlan(FullPlan):
    @property
    def protocol_id(self):return CONFIG['protocol']


def load_plan(directory):
    return FlashHyperPlan(validate_component(directory,CONFIG,GEOMETRY,dict(programId=PROGRAM.identity),versions(),source=WEIGHTS))
