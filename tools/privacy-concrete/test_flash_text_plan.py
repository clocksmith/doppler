"""Installed route metadata and tiny patched cache admission; no model run."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import flash_text_plan as plan
from flash_component_plan import write_component
from full_plan import canonical
from protocol import digest

class FlashTextPlanTests(unittest.TestCase):
    def test_installed_route_has_every_expert_and_distinct_protocol(self):
        self.assertEqual(len(plan.GEOMETRY),74501)
        for layer in range(48):
            for expert in range(512):
                for part in ['gate','up','down']:self.assertIn(f'layers.{layer}.mlp.ops.expert_{expert}_{part}',plan.GEOMETRY)
        self.assertEqual(list(plan.GEOMETRY)[-1],'head')
        self.assertFalse(plan.PROGRAM['productionBackendAvailable']);self.assertFalse(plan.PROGRAM['securityQualified'])
        self.assertEqual(plan.PROGRAM['integrity'],'full-encrypted-tile-replay')

    def fixture(self,root):
        data=b'\x01\x02\x03\x04';identity=digest(data);(root/(identity+'.i8')).write_bytes(data)
        definitions=[dict(name='projection',weightId=identity,inputWidth=2,outputWidth=2)]
        source={'schema':'tiny-test-only'};geometry={'projection':(2,2)};versions={'test':'simulated'}
        write_component(root,plan.CONFIG,definitions,plan.semantics('a'*64),versions,source=source)
        return source,geometry,versions,root/(identity+'.i8')

    def test_tiny_fixture_loads_via_installed_backend_dispatch(self):
        from plan_backends import load_backend_plan,PROFILES
        self.assertIn(plan.CONFIG['profile'],PROFILES)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,geometry,versions,_=self.fixture(root)
            with patch.object(plan,'TEXT',source),patch.object(plan,'GEOMETRY',geometry),patch.object(plan,'versions',return_value=versions):
                admitted=load_backend_plan(root,plan.CONFIG['profile'])
            self.assertIs(type(admitted),plan.FlashTextPlan);self.assertEqual(admitted.definition()['clientGraphId'],'a'*64)
            self.assertEqual(admitted.protocol_id,plan.CONFIG['protocol'])
            self.assertFalse(admitted.definition()['fullFlashModelQualified'])

    def test_changed_source_route_qualification_and_bytes_reject(self):
        for changed in ['source','order','geometry','graph','security','production','cache','symlink','extra']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source,geometry,versions,cache=self.fixture(root)
                value=json.loads((root/'plan.json').read_bytes())
                if changed=='source':value['source']['schema']='untrusted'
                elif changed=='order':value['projections'][0]['name']='other'
                elif changed=='geometry':value['projections'][0]['inputWidth']=1
                elif changed=='graph':value['clientGraphId']='a'*63
                elif changed=='security':value['securityQualified']=True
                elif changed=='production':value['productionBackendAvailable']=True
                elif changed=='extra':value['projections'][0]['extra']=True
                elif changed=='cache':cache.write_bytes(b'bad!')
                elif changed=='symlink':
                    alternate=root/'other';cache.rename(alternate);cache.symlink_to(alternate)
                (root/'plan.json').write_bytes(canonical(value))
                with patch.object(plan,'TEXT',source),patch.object(plan,'GEOMETRY',geometry),patch.object(plan,'versions',return_value=versions):
                    with self.assertRaises(ValueError):plan.load_plan(root)

if __name__=='__main__':unittest.main()
