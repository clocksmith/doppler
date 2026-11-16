"""Metadata bounds reject full-weight fallback before body reads; no model/network workload."""
import io
import json
import struct
import unittest
from unittest.mock import patch
from flash_inventory_metadata import shard_headers,public_bytes


class Response(io.BytesIO):
    status=200
    headers={}


class InventoryBoundsTests(unittest.TestCase):
    def setup_responses(self,header=b'{}',status=206,content_range=None):
        name='model-00001-of-00001.safetensors';size=4096
        base=dict(repository='public/repo',revision='a'*40)
        tree=Response(json.dumps([dict(path=name,size=size,lfs=dict(oid='b'*64))]).encode())
        prefix=Response(struct.pack('<Q',len(header)));prefix.status=status;prefix.headers={'Content-Range':content_range or 'bytes 0-7/4096'}
        body=Response(header);body.status=206;body.headers={'Content-Range':f'bytes 8-{7+len(header)}/4096'}
        return name,base,[tree,prefix,body]

    def test_exact_header_range_is_recorded_without_weight_payload(self):
        name,base,responses=self.setup_responses()
        with patch('flash_inventory_metadata.urllib.request.urlopen',side_effect=responses) as request:headers=shard_headers(base,[name])
        self.assertEqual(request.call_count,3);self.assertEqual(headers[name][0]['headerBytes'],2)
        self.assertEqual(headers[name][1],{})

    def test_full_objects_inexact_ranges_and_unbounded_headers_fail_closed(self):
        for change in [dict(status=200),dict(content_range='bytes 0-7/9999'),dict(header=b'x'*1048577)]:
            name,base,responses=self.setup_responses(**change)
            with patch('flash_inventory_metadata.urllib.request.urlopen',side_effect=responses) as request:
                with self.assertRaises(ValueError):shard_headers(base,[name])
                self.assertEqual(request.call_count,2)
        with patch('flash_inventory_metadata.urllib.request.urlopen') as request:
            with self.assertRaises(ValueError):shard_headers(dict(repository='public/repo',revision='a'*40),['../secret'])
            request.assert_not_called()

    def test_bounded_public_metadata_rejects_digest_size_and_partial_objects(self):
        for response,limit,expected in [(Response(b'large'),2,None),(Response(b'bad'),9,'0'*64)]:
            with patch('flash_inventory_metadata.urllib.request.urlopen',return_value=response):
                with self.assertRaises(ValueError):public_bytes('https://public.invalid/metadata',limit,expected)
        response=Response(b'partial');response.status=206
        with patch('flash_inventory_metadata.urllib.request.urlopen',return_value=response):
            with self.assertRaises(ValueError):public_bytes('https://public.invalid/metadata',99)


if __name__=='__main__':unittest.main()
