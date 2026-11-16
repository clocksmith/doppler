"""Actual installed input layout rejects missing files before private rows or network access."""
import tempfile
import unittest
from unittest.mock import patch
from local_token_rows import LocalTokenRows,installed_inventory,POLICY
from flash_tokenizer import SOURCE
from protocol import digest
from full_plan import canonical


class InstalledTokenRowsTests(unittest.TestCase):
    def test_actual_embedding_geometry_and_tokenizer_are_one_source_contract(self):
        value=installed_inventory();self.assertEqual(value['logicalShape'],[248320,2560])
        self.assertEqual(value['wholeShardBytes'],2974788136);self.assertEqual(len(value['parts']),1)
        self.assertEqual(value['tokenizerSourceId'],digest(canonical(SOURCE)))

    def test_missing_or_underbudget_complete_file_never_fetches_private_rows(self):
        with tempfile.TemporaryDirectory() as directory,patch('flash_weights.urllib.request.urlopen',side_effect=AssertionError('remote row fallback')) as network:
            with self.assertRaises(ValueError):LocalTokenRows.flash(directory,local_storage_limit_bytes=1)
            with self.assertRaises(FileNotFoundError):LocalTokenRows.flash(directory,local_storage_limit_bytes=POLICY['ciLocalStorageLimitBytes'])
            network.assert_not_called()


if __name__=='__main__':unittest.main()
