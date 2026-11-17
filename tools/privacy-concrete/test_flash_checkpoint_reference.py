"""Reference identity admission only; no model execution or encryption."""
from contextlib import ExitStack
import copy
from types import SimpleNamespace
import sys
import threading
import unittest
from unittest.mock import patch
import flash_checkpoint_reference as reference
from full_plan import canonical
from protocol import digest


class CheckpointReferenceTests(unittest.TestCase):
    def test_installed_reference_matches_the_complete_public_source_scope(self):
        from flash_source_scopes import TEXT
        from flash_text_plan import GEOMETRY
        self.assertEqual(reference.REFERENCE['sourceId'],digest(canonical(TEXT)))
        self.assertEqual(reference.REFERENCE['sourceRevision'],TEXT['revision'])
        self.assertEqual(reference.REFERENCE['projections'],len(GEOMETRY))
        self.assertEqual(reference.REFERENCE['quantizationBits'],12)
        for name in ('encryptedExecution','productionBackendAvailable','securityQualified'):
            self.assertIs(reference.REFERENCE[name],False)

    def fixture(self,stack):
        geometry={'first':(4,3),'second':(3,5)};source={'revision':'public-fixture'}
        rows=[dict(name=name,inputWidth=shape[0],outputWidth=shape[1],weightId=str(i+1)*64)
              for i,(name,shape) in enumerate(geometry.items())]
        class Prepared:
            identity='a'*64;quantization_bits=12;order=tuple(geometry)
            graph=SimpleNamespace(identity='a'*64)
            def require_graph(self):pass
            def definitions(self):return rows
        prepared=Prepared();pin=dict(reference.REFERENCE,sourceRevision=source['revision'],sourceId=digest(canonical(source)),
            projections=len(rows),orderedCoefficientDigest=digest(canonical([
                dict(name=row['name'],weightId=row['weightId']) for row in rows])))
        stack.enter_context(patch.dict(sys.modules,{
            'flash_prepared_graph':SimpleNamespace(PreparedFlashGraph=Prepared),
            'flash_source_scopes':SimpleNamespace(TEXT=source),
            'flash_text_plan':SimpleNamespace(GEOMETRY=geometry)}))
        stack.enter_context(patch.object(reference,'REFERENCE',pin))
        return prepared,rows,pin

    def test_ordered_complete_coefficient_identity_matches_and_changes_reject(self):
        for change in (None,'weight','order','missing','duplicate','geometry','graph','bits','source','production','unowned'):
            with self.subTest(change=change),ExitStack() as stack:
                prepared,rows,pin=self.fixture(stack)
                if change=='weight':rows[0]['weightId']='f'*64
                if change=='order':rows.reverse()
                if change=='missing':rows.pop()
                if change=='duplicate':rows[1]=copy.deepcopy(rows[0])
                if change=='geometry':rows[0]['inputWidth']+=1
                if change=='graph':prepared.identity='b'*64
                if change=='bits':prepared.quantization_bits=8
                if change=='source':pin['sourceId']='c'*64
                if change=='production':pin['securityQualified']=True
                if change=='unowned':prepared=SimpleNamespace(**prepared.__dict__)
                if change is None:
                    result=reference.require_checkpoint_reference(prepared)
                    self.assertEqual(result['orderedCoefficientDigest'],pin['orderedCoefficientDigest'])
                    self.assertEqual(result['pinId'],digest(canonical(pin)))
                    self.assertEqual(result['admittedClientGraphId'],prepared.identity)
                    self.assertEqual(result['referenceClientGraphId'],pin['referenceClientGraphId'])
                else:
                    with self.assertRaises(ValueError):reference.require_checkpoint_reference(prepared)

    def test_cancelled_or_revoked_graph_rejects_before_inventory(self):
        for change in ('cancel','graph'):
            with self.subTest(change=change),ExitStack() as stack:
                prepared,_,_=self.fixture(stack);event=threading.Event()
                if change=='cancel':event.set()
                else:prepared.require_graph=lambda:(_ for _ in ()).throw(ValueError('source revoked'))
                prepared.definitions=lambda:self.fail('inventory read after cancellation or graph revocation')
                with self.assertRaises(InterruptedError if change=='cancel' else ValueError):
                    reference.require_checkpoint_reference(prepared,cancelled=event)


if __name__=='__main__':unittest.main()
