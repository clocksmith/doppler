"""Bind public tile contents to the same whole-verified byte stream, with bounded reads.

Only public projection metadata reaches the source callback. A completed binding
can outlive evicted coefficient files; it is not provider evidence, a serialized
import format or permission to admit a new execution backend.
"""
import copy
from contextlib import contextmanager,ExitStack
from dataclasses import dataclass
import hashlib
import os
import tempfile
from pathlib import Path
from types import MappingProxyType
from full_plan import canonical,tiles
from protocol import digest,require_active
from weight_encoding import weight_encoding
from source_snapshot import file_signature

PUBLIC_TILE_BINDING_INTERFACE='doppler.public-tile-binding/v1'
PUBLIC_PROJECTION_BINDING_INTERFACE='doppler.public-projection-binding/v1'
MAX_PROJECTION_READ_BYTES=1024*1024


def validate_public_projection(row):
    if (not isinstance(row,dict) or set(row)!={'name','weightId','inputWidth','outputWidth'}
            or not isinstance(row['name'],str) or not row['name']
            or not isinstance(row['weightId'],str) or len(row['weightId'])!=64
            or any(char not in '0123456789abcdef' for char in row['weightId'])):
        raise ValueError('Exact public projection identity required')
    if any(type(row[key]) is not int or row[key]<=0 for key in ['inputWidth','outputWidth']):
        raise ValueError('Explicit positive public projection geometry required')


def projection_schedule(row,definition):
    validate_public_projection(row)
    return tiles(row,definition['inputTileWidth'],definition['outputTileWidth'])


def projection_tile_digests(handle,row,schedule,encoding,*,cancelled=None):
    identity=row['weightId']
    by_input={}
    for tile in schedule:by_input.setdefault(tile['inputBegin'],[]).append(tile)
    whole=hashlib.sha256();result={};size=encoding['bytes']
    for begin,group in sorted(by_input.items()):
        hashers={tile['id']:hashlib.sha256() for tile in group}
        if len(group)==1 and group[0]['outputBegin']==0 and group[0]['outputEnd']==row['outputWidth']:
            # One output tile covers consecutive complete rows. Consume the
            # same owned stream in bounded blocks instead of one syscall and
            # cancellation-path stat per row. No coefficient bytes are skipped.
            spans=((group[0],(group[0]['inputEnd']-begin)*row['outputWidth']*size),)
        else:
            spans=((tile,(tile['outputEnd']-tile['outputBegin'])*size)
                   for _index in range(begin,group[0]['inputEnd']) for tile in group)
        for tile,remaining in spans:
            while remaining:
                require_active(cancelled);requested=min(remaining,MAX_PROJECTION_READ_BYTES)
                chunk=handle.read(requested)
                if not isinstance(chunk,bytes) or not chunk or len(chunk)>requested:
                    raise ValueError('Complete bounded public coefficient stream required')
                whole.update(chunk);hashers[tile['id']].update(chunk);remaining-=len(chunk)
        result.update({tile_id:hasher.hexdigest() for tile_id,hasher in hashers.items()})
    # Tile and whole-file digests consume exactly the same owned chunks, in row
    # order. A later rewrite cannot change this binding, and a mixed read fails
    # the admitted whole-file digest. No reread or mutable mmap defines truth.
    require_active(cancelled);tail=handle.read(1)
    if not isinstance(tail,bytes) or tail or whole.hexdigest()!=identity:
        raise ValueError('Public model cache changed before tile binding')
    return result


@dataclass(frozen=True,init=False)
class PreparedPublicProjection:
    """Client-created proof from one complete coefficient byte stream, not JSON."""
    encoded_scope:bytes
    entries:tuple

    def __init__(self,row,definition,handle,*,cancelled=None):
        encoding=weight_encoding(definition.get('parameters',{}))
        schedule=projection_schedule(row,definition)
        entries=projection_tile_digests(handle,row,schedule,encoding,cancelled=cancelled)
        object.__setattr__(self,'encoded_scope',canonical(dict(projection=dict(row),encoding=encoding,
            inputTileWidth=definition['inputTileWidth'],outputTileWidth=definition['outputTileWidth'])))
        object.__setattr__(self,'entries',tuple(entries.items()))

    def digests_for(self,row,definition):
        expected=canonical(dict(projection=dict(row),encoding=weight_encoding(definition.get('parameters',{})),
            inputTileWidth=definition['inputTileWidth'],outputTileWidth=definition['outputTileWidth']))
        if self.encoded_scope!=expected:raise ValueError('Compiled public projection scope changed')
        return dict(self.entries)


@dataclass(frozen=True,init=False)
class PreparedPublicTiles:
    encoded_definition:bytes
    entries:tuple

    def __init__(self,plan,open_projection):
        if not callable(open_projection):raise ValueError('Explicit client-owned public coefficient source required')
        definition=copy.deepcopy(plan.definition());encoding=weight_encoding(definition.get('parameters',{}));result={}
        schedules=[projection_schedule(row,definition) for row in definition['projections']]
        identifiers=[tile['id'] for schedule in schedules for tile in schedule]
        if not identifiers or len(set(identifiers))!=len(identifiers):raise ValueError('Unique complete public tile route required')
        for row,schedule in zip(definition['projections'],schedules):
            with open_projection(MappingProxyType(dict(row)),MappingProxyType(dict(encoding))) as handle:
                current=projection_tile_digests(handle,row,schedule,encoding)
            if result.keys()&current.keys():raise ValueError('Duplicate public tile identity in complete route')
            result.update(current)
        if not result:raise ValueError('Complete public tile binding required')
        object.__setattr__(self,'encoded_definition',canonical(definition))
        object.__setattr__(self,'entries',tuple(result.items()))

    @classmethod
    def from_projections(cls,plan,projections):
        """Combine byte-derived records after their owned coefficient files evict.

        Only actual compiler objects are accepted. Wire dictionaries, advertised
        hashes and serialized manifests cannot instantiate this client admission.
        """
        definition=plan.definition();rows=definition['projections'];proofs=tuple(projections)
        if len(rows)!=len(proofs) or not rows:raise ValueError('Complete compiled public projection route required')
        result={}
        for row,proof in zip(rows,proofs):
            if type(proof) is not PreparedPublicProjection:raise ValueError('Byte-derived client projection proof required')
            current=proof.digests_for(row,definition)
            if result.keys()&current.keys():raise ValueError('Duplicate compiled public tile')
            result.update(current)
        binding=object.__new__(cls)
        object.__setattr__(binding,'encoded_definition',canonical(definition))
        object.__setattr__(binding,'entries',tuple(result.items()))
        return binding

    @property
    def interface(self):return PUBLIC_TILE_BINDING_INTERFACE

    @property
    def plan_id(self):return digest(self.encoded_definition)

    def digests_for(self,plan):
        if canonical(plan.definition())!=self.encoded_definition:
            raise ValueError('Prepared public tiles belong to a different plan')
        return MappingProxyType(dict(self.entries))


def bind_public_tiles(directory,plan):
    """Legacy cache interface, now bounded by one output-tile row of coefficients."""
    @contextmanager
    def source(row,encoding):
        path=Path(directory)/(row['weightId']+encoding['suffix'])
        with path.open('rb') as handle:yield handle
    return PreparedPublicTiles(plan,source).digests_for(plan)


class PublicTileReader:
    """One active public coefficient file, with owned tile bytes and terminal close.

    The installed plan and whole-file cache are validated before this owner is
    constructed. This reader limits descriptors/mappings; it neither downloads
    missing files nor evicts files from disk, and is not an integrity verifier.
    """
    def __init__(self,directory,plan,*,open_projection=None,check_active=None):
        definition=plan.definition()
        self.directory=Path(directory)
        if open_projection is not None and not callable(open_projection):raise ValueError('Trusted public projection lease callback required')
        if check_active is not None and not callable(check_active):raise ValueError('Explicit public source lifetime check required')
        self.open_projection=open_projection;self.handle_scope=ExitStack()
        self.check_active=check_active
        self.encoding=weight_encoding(definition.get('parameters',{}))
        self.input_limit,self.output_limit=definition['inputTileWidth'],definition['outputTileWidth']
        self.rows={}
        for row in definition['projections']:
            projection_schedule(row,definition)
            if row['name'] in self.rows:raise ValueError('Unique public projection required')
            self.rows[row['name']]=MappingProxyType(dict(row))
        if not self.rows:raise ValueError('Complete public projection route required')
        self.handle,self.active_name,self.closed=None,None,False

    def __enter__(self):
        if self.closed:raise ValueError('Public tile reader is closed')
        return self

    def __exit__(self,*_args):self.close()

    def close(self):
        self.closed=True
        try:self.handle_scope.close()
        finally:self.handle,self.active_name=None,None

    def read(self,tile):
        try:
            if self.closed:raise ValueError('Public tile reader is closed')
            row=self.rows.get(tile.get('name'))
            if row is None:raise ValueError('Unknown public projection')
            begin_in,begin_out=tile.get('inputBegin'),tile.get('outputBegin')
            if (type(begin_in) is not int or type(begin_out) is not int
                    or not 0<=begin_in<row['inputWidth'] or begin_in%self.input_limit
                    or not 0<=begin_out<row['outputWidth'] or begin_out%self.output_limit):
                raise ValueError('Unadmitted public tile geometry')
            expected=dict(name=row['name'],weightId=row['weightId'],inputBegin=begin_in,
                          inputEnd=min(begin_in+self.input_limit,row['inputWidth']),
                          outputBegin=begin_out,outputEnd=min(begin_out+self.output_limit,row['outputWidth']))
            expected['id']=digest(canonical(expected))
            if dict(tile)!=expected:raise ValueError('Public tile identity changed')
            if self.check_active is not None:self.check_active(tile)
            if self.active_name!=row['name']:
                self.handle_scope.close();self.handle_scope=ExitStack()
                self.handle=None;self.active_name=None
                lease=((self.directory/(row['weightId']+self.encoding['suffix'])).open('rb',buffering=0)
                    if self.open_projection is None else self.open_projection(row,MappingProxyType(dict(self.encoding))))
                self.handle=self.handle_scope.enter_context(lease)
                if os.fstat(self.handle.fileno()).st_size!=row['inputWidth']*row['outputWidth']*self.encoding['bytes']:
                    raise ValueError('Public coefficient file geometry changed')
                self.active_name=row['name']
            size=self.encoding['bytes'];width=(expected['outputEnd']-begin_out)*size
            result=bytearray()
            for index in range(begin_in,expected['inputEnd']):
                self.handle.seek((index*row['outputWidth']+begin_out)*size)
                remaining=width
                while remaining:
                    chunk=self.handle.read(remaining)
                    if not chunk:raise ValueError('Public coefficient file was truncated')
                    result.extend(chunk);remaining-=len(chunk)
            return bytes(result)
        except BaseException:
            self.close()
            raise


def coefficient_signature(path):
    # Atomic cache publication uses a temporary hard link. Removing that owned
    # link changes ctime without changing content; bind inode/size/mtime instead.
    # Whole-file admission and before-disclosure tile digests remain required.
    return file_signature(path)[:4]


@contextmanager
def mapped_public_matrix(path,shape,dtype):
    """Borrow one read-only matrix for this scope; copies may outlive the mapping.

    Whole-file identity admission and per-tile integrity are owned by the plan
    loader/bridge. This scope detects ordinary file changes and closes explicitly;
    it is not a replacement for those checks or malicious-host protection.
    """
    import numpy as np
    if (not isinstance(shape,tuple) or len(shape)!=2
            or any(type(width) is not int or width<=0 for width in shape)
            or np.dtype(dtype) not in [np.dtype('int8'),np.dtype('<i2')]):
        raise ValueError('Explicit signed public matrix geometry/encoding required')
    path=Path(path);before=coefficient_signature(path)
    if before[2]!=shape[0]*shape[1]*np.dtype(dtype).itemsize:
        raise ValueError('Public matrix cache geometry changed')
    matrix=np.memmap(path,dtype=dtype,mode='r',shape=shape)
    try:
        if coefficient_signature(path)!=before:raise ValueError('Public matrix changed while opening')
        yield matrix
    finally:
        matrix._mmap.close()
        if coefficient_signature(path)!=before:raise ValueError('Public matrix changed while in use')


def store_public_matrix(directory,name,input_width,output_width,parameters,payload):
    """Install complete trusted public coefficients without replacing existing data.

    Only public weight preparation calls this owner. An existing cache must match;
    concurrent identical writers share it. Partial writes stay under an owned
    temporary name and cannot become a content-addressed cache entry.
    """
    encoding=weight_encoding(parameters)
    if not isinstance(payload,bytes):raise ValueError('Owned immutable public coefficient bytes required')
    row=dict(name=name,inputWidth=input_width,outputWidth=output_width,weightId=digest(payload))
    validate_public_projection(row)
    if len(payload)!=input_width*output_width*encoding['bytes']:
        raise ValueError('Complete public coefficient matrix required')
    path=Path(directory)/(row['weightId']+encoding['suffix'])
    def verify_existing():
        if path.is_symlink() or not path.is_file():raise ValueError('Regular public coefficient cache required')
        before=coefficient_signature(path)
        with path.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
        if before[2]!=len(payload) or actual!=row['weightId'] or coefficient_signature(path)!=before:
            raise ValueError('Public coefficient cache corruption')
    if path.exists():
        verify_existing();return path,row
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+row['weightId']+'.',suffix='.partial',delete=False) as handle:
            temporary=Path(handle.name)
            view=memoryview(payload)
            for begin in range(0,len(view),1024*1024):
                chunk=view[begin:begin+1024*1024]
                if handle.write(chunk)!=len(chunk):raise OSError('Incomplete public coefficient write')
        try:os.link(temporary,path) # Atomic create-only publication on the same filesystem.
        except FileExistsError:verify_existing()
        return path,row
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def store_public_matrix_columns(directory,name,input_width,output_width,parameters,columns,*,cancelled=None):
    """Publish consecutive output-channel chunks without a whole-matrix copy.

    ``columns`` is trusted client quantization output, never provider input. Its
    signed integer arrays have shape (input_width, channel_count). The owned
    temporary mapping is closed before create-only publication; failures remove
    only that temporary file. Existing content-addressed entries are verified.
    """
    import numpy as np
    encoding=weight_encoding(parameters)
    validate_public_projection(dict(name=name,inputWidth=input_width,outputWidth=output_width,weightId='0'*64))
    require_active(cancelled);temporary=None;matrix=None
    expected_bytes=input_width*output_width*encoding['bytes']
    try:
        with tempfile.NamedTemporaryFile(dir=directory,prefix='.public-columns-',suffix='.partial',delete=False) as handle:
            temporary=Path(handle.name);handle.truncate(expected_bytes)
        matrix=np.memmap(temporary,dtype=encoding['dtype'],mode='r+',shape=(input_width,output_width))
        end=0
        for chunk in columns:
            require_active(cancelled)
            if (not isinstance(chunk,np.ndarray) or chunk.ndim!=2 or chunk.shape[0]!=input_width
                    or not 0<chunk.shape[1]<=output_width-end or chunk.dtype!=np.dtype(encoding['dtype'])):
                raise ValueError('Complete consecutive signed public coefficient columns required')
            following=end+chunk.shape[1];matrix[:,end:following]=chunk;end=following
        if end!=output_width:raise ValueError('Incomplete public coefficient columns')
        require_active(cancelled);matrix.flush();matrix._mmap.close();matrix=None
        def file_digest(path):
            hasher=hashlib.sha256()
            with path.open('rb') as handle:
                while True:
                    require_active(cancelled);chunk=handle.read(1024*1024)
                    if not chunk:break
                    hasher.update(chunk)
            return hasher.hexdigest()
        identity=file_digest(temporary)
        row=dict(name=name,inputWidth=input_width,outputWidth=output_width,weightId=identity)
        path=Path(directory)/(identity+encoding['suffix'])
        require_active(cancelled)
        try:os.link(temporary,path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():raise ValueError('Regular public coefficient cache required')
            before=coefficient_signature(path)
            if before[2]!=expected_bytes or file_digest(path)!=identity or coefficient_signature(path)!=before:
                raise ValueError('Public coefficient cache corruption')
        # Publication contains only public bytes. Cancellation here rejects the
        # preparation but leaves a complete shared cache entry intact.
        require_active(cancelled)
        return path,row
    finally:
        if matrix is not None:matrix._mmap.close()
        if temporary is not None:temporary.unlink(missing_ok=True)
