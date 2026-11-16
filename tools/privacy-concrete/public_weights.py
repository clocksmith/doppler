"""Bind public tile contents to one verified snapshot, without retaining model-sized copies."""
from pathlib import Path
from full_plan import tiles
from protocol import digest


def bind_public_tiles(directory,plan):
    """Retain only public tile digests; later cache mutation cannot redefine replay weights.

    Each projection is read into owned bytes and verified against its admitted weightId
    before deriving strided tile digests. A filesystem path or read-only mmap alone is
    not immutable when another process can rewrite its backing file.
    """
    result={}
    definition=plan.definition()
    for row in definition['projections']:
        identity=row['weightId']
        if not isinstance(identity,str) or len(identity)!=64 or any(char not in '0123456789abcdef' for char in identity):
            raise ValueError('Public weight identity required')
        data=(Path(directory)/(identity+'.i8')).read_bytes()
        if len(data)!=row['inputWidth']*row['outputWidth'] or digest(data)!=identity:
            raise ValueError('Public model cache changed before tile binding')
        stride=row['outputWidth']
        for tile in tiles(row,definition['inputTileWidth'],definition['outputTileWidth']):
            payload=b''.join(data[index*stride+tile['outputBegin']:index*stride+tile['outputEnd']]
                             for index in range(tile['inputBegin'],tile['inputEnd']))
            result[tile['id']]=digest(payload)
    return result
