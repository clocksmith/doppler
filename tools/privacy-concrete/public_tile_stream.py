"""Versioned bounded public coefficients beside ciphertext, never plaintext activations.

One admitted tile per existing route sequence. The provider owns only this tile;
the client must already hold a complete byte-derived preparation binding.
"""
import struct
from protocol import digest

CONTRACT='doppler.client-compiled-public-tile-stream/v1'
REBUILD_CONTRACT='doppler.provider-verified-public-checkpoint/v1'
MAGIC=b'DPCT\x01'


def tile_bytes(tile,encoding):
    return (tile['inputEnd']-tile['inputBegin'])*(tile['outputEnd']-tile['outputBegin'])*encoding['bytes']


def encode(public,ciphertext,tile,encoding,expected_digest):
    if (type(public) is not bytes or type(ciphertext) is not bytes or not ciphertext
            or len(public)!=tile_bytes(tile,encoding) or digest(public)!=expected_digest):
        raise ValueError('Exact admitted public coefficient tile and ciphertext required')
    return MAGIC+struct.pack('!I',len(public))+public+ciphertext


def decode(payload,tile,encoding,expected_digest):
    count=tile_bytes(tile,encoding);prefix=len(MAGIC)+4
    if (type(payload) is not bytes or len(payload)<=prefix+count or payload[:len(MAGIC)]!=MAGIC
            or struct.unpack('!I',payload[len(MAGIC):prefix])[0]!=count):
        raise ValueError('Versioned complete public tile stream required')
    public=payload[prefix:prefix+count]
    if digest(public)!=expected_digest:raise ValueError('Streamed public coefficient tile changed')
    return public,payload[prefix+count:]
