"""Installed twelve-bit checked Flash route metadata and tiny patched cache admission; no model run."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import flash_text_plan as replay
import flash_text_ntt12_check_plan as plan
from flash_component_plan import write_component
from full_plan import canonical
from protocol import digest

PARAMETERS=dict(parameters=dict(bits_reserved_for_computation=35),quantizationBits=12,weightEncoding='signed-int12-in-int16')


class FlashTextNtt12CheckPlanTests(unittest.TestCase):
    def test_route_is_complete_and_cannot_be_confused_with_the_replay_profile(self):
        self.assertIs(plan.GEOMETRY,replay.GEOMETRY);self.assertEqual(len(plan.GEOMETRY),74501)
        self.assertNotEqual(plan.CONFIG['protocol'],replay.CONFIG['protocol']);self.assertNotEqual(plan.CONFIG['planSchema'],replay.CONFIG['planSchema'])
        self.assertNotEqual(plan.CONFIG['profile'],replay.CONFIG['profile']);self.assertNotEqual(plan.PROGRAM_ID,replay.PROGRAM_ID)
        self.assertEqual(plan.PROGRAM['integrity'],'secret-linear-check-sampled-replay/v1');self.assertEqual(replay.PROGRAM['integrity'],'full-encrypted-tile-replay')
        self.assertFalse(plan.PROGRAM['productionBackendAvailable']);self.assertFalse(plan.PROGRAM['securityQualified'])
        self.assertEqual(plan.CONFIG['quantizationBits'],12);self.assertEqual(plan.CONFIG['nativeVariant'],'cpu-exact-ntt/v1')

    def fixture(self,root,values=(1,-2,2047,-2048)):
        data=np.array(values,dtype='<i2').tobytes();identity=digest(data);(root/(identity+'.i16')).write_bytes(data)
        definitions=[dict(name='projection',weightId=identity,inputWidth=2,outputWidth=2)]
        source={'schema':'tiny-test-only'};geometry={'projection':(2,2)};versions={'test':'simulated'}
        with patch('flash_component_plan.component_parameters',return_value=PARAMETERS):
            write_component(root,plan.CONFIG,definitions,plan.semantics('a'*64),versions,source=source)
        return source,geometry,versions,root/(identity+'.i16')

    def patched(self,source,geometry,versions):
        return (patch.object(plan,'TEXT',source),patch.object(plan,'GEOMETRY',geometry),patch.object(plan,'versions',return_value=versions),
                patch.object(plan,'parameters',return_value=PARAMETERS),patch('flash_component_plan.component_parameters',return_value=PARAMETERS))

    def load(self,root,source,geometry,versions,loader):
        a,b,c,d,e=self.patched(source,geometry,versions)
        with a,b,c,d,e:return loader(root)

    def test_tiny_int16_fixture_loads_via_installed_backend_dispatch(self):
        from plan_backends import load_backend_plan,PROFILES
        self.assertIn(plan.CONFIG['profile'],PROFILES)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,geometry,versions,_=self.fixture(root)
            admitted=self.load(root,source,geometry,versions,lambda path:load_backend_plan(path,plan.CONFIG['profile']))
            self.assertIs(type(admitted),plan.FlashTextNtt12CheckPlan);self.assertEqual(admitted.protocol_id,plan.CONFIG['protocol'])
            self.assertEqual(admitted.definition()['parameters']['quantizationBits'],12)

    def test_replay_profile_plan_and_changed_content_reject(self):
        for changed in ['protocol','schema','security','production','cache','symlink','graph']:
            with self.subTest(changed=changed),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);source,geometry,versions,cache=self.fixture(root)
                value=json.loads((root/'plan.json').read_bytes())
                if changed=='protocol':value['protocol']=replay.CONFIG['protocol']
                elif changed=='schema':value['schema']=replay.CONFIG['planSchema']
                elif changed=='security':value['securityQualified']=True
                elif changed=='production':value['productionBackendAvailable']=True
                elif changed=='graph':value['clientGraphId']='a'*63
                elif changed=='cache':cache.write_bytes(np.array([1,2,3,4],dtype='<i2').tobytes())
                elif changed=='symlink':
                    alternate=root/'other';cache.rename(alternate);cache.symlink_to(alternate)
                (root/'plan.json').write_bytes(canonical(value))
                with self.assertRaises(ValueError):self.load(root,source,geometry,versions,plan.load_plan)

    def test_coefficient_outside_twelve_bits_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source,geometry,versions,_=self.fixture(root,values=(1,2,2048,4))
            with self.assertRaises(ValueError):self.load(root,source,geometry,versions,plan.load_plan)


if __name__=='__main__':unittest.main()
