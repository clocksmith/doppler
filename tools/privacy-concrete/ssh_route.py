"""Explicit, pinned SSH endpoint; no source installation, ambient SSH config or plaintext fallback."""
import argparse
import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import subprocess
import tempfile
from full_plan import FULL_POLICY, canonical
from plan_backends import DEFAULT_PROFILE,QWEN_PROFILES,qwen_backend
from protocol import digest, parse_unique_json

POLICY=json.loads(Path(__file__).with_name('ssh-policy.json').read_text())
POLICY_V2=json.loads(Path(__file__).with_name('ssh-v2-policy.json').read_text())
# Checked-integrity backends get their own route schema and frame protocol, so
# an SSH frame never names replay for a route that does not replay every tile.
POLICY_CHECK=json.loads(Path(__file__).with_name('ssh-check-policy.json').read_text())
POLICY_STREAM_CHECK=json.loads(Path(__file__).with_name('ssh-stream-check-policy.json').read_text())
POLICY_REBUILD_CHECK=json.loads(Path(__file__).with_name('ssh-rebuild-check-policy.json').read_text())
ENDPOINT_FIELDS=frozenset({'operatorId','host','port','user','hostKey','remotePython','remoteProvider','remoteCache','remoteManifest'})
REBUILD_ENDPOINT_FIELDS=ENDPOINT_FIELDS|{'remotePublicSource'}
FIELDS=ENDPOINT_FIELDS|{'schema','basePlanId','modelId','backendProtocol'}
SHA=re.compile(r'[0-9a-f]{64}\Z')


def public_key(value):
    if not isinstance(value,dict) or set(value)!={'algorithm','base64'} or value['algorithm']!=POLICY['hostKeyAlgorithm']:
        raise ValueError('Explicit pinned Ed25519 SSH host key required')
    try:
        raw=base64.b64decode(value['base64'],validate=True)
        algorithm=POLICY['hostKeyAlgorithm'].encode()
        if value['base64']!=base64.b64encode(raw).decode():raise ValueError()
        if raw!=struct.pack('!I',len(algorithm))+algorithm+struct.pack('!I',32)+raw[-32:]:
            raise ValueError()
    except (ValueError,TypeError):
        raise ValueError('Invalid pinned SSH host key') from None
    return 'SHA256:'+base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip('=')


def route_policy(value):
    if not isinstance(value,dict):raise ValueError('Exact versioned encrypted SSH route required')
    schema=value.get('schema')
    if schema==POLICY['routeSchema']:return POLICY
    if schema==POLICY_V2['routeSchema']:return POLICY_V2
    if schema==POLICY_CHECK['routeSchema']:return POLICY_CHECK
    if schema==POLICY_STREAM_CHECK['routeSchema']:return POLICY_STREAM_CHECK
    if schema==POLICY_REBUILD_CHECK['routeSchema']:return POLICY_REBUILD_CHECK
    raise ValueError('Unsupported versioned encrypted SSH route')


def route_backend(value):
    policy=route_policy(value)
    if policy is POLICY:return DEFAULT_PROFILE,FULL_POLICY
    profile=value.get('backendProfile')
    if policy is POLICY_STREAM_CHECK:
        from flash_text_stream_plan import CONFIG
        if profile!=CONFIG['profile']:raise ValueError('Streamed SSH route requires its explicit installed storage backend')
        return profile,CONFIG
    if policy is POLICY_REBUILD_CHECK:
        from flash_text_rebuild_plan import CONFIG
        if profile!=CONFIG['profile']:raise ValueError('Checkpoint SSH route requires its explicit installed reconstruction backend')
        return profile,CONFIG
    settings,_,_,_=qwen_backend(profile)
    if settings['productionBackendAvailable'] is not False:raise ValueError('Research-only SSH backend required')
    if (policy is POLICY_CHECK)!=(settings.get('integrity') is not None):
        raise ValueError('SSH route schema differs from the backend integrity mechanism')
    return profile,settings


def valid_operator(value):return isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:@/-]{1,128}',value) is not None


def require_operator(operator,allowed_operators):
    if (not isinstance(allowed_operators,tuple) or not all(valid_operator(value) for value in allowed_operators)
            or operator not in allowed_operators):
        raise ValueError('SSH operator is not explicitly allowed by this requester')


def validate_endpoint(value):
    if (not isinstance(value['host'],str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?',value['host'])
            or '..' in value['host'] or type(value['port']) is not int or not 1<=value['port']<=65535
            or not isinstance(value['user'],str) or not re.fullmatch(r'[a-z_][a-z0-9_-]{0,63}',value['user'])
            or not valid_operator(value['operatorId'])):
        raise ValueError('Bounded explicit SSH endpoint and operator required')
    public_key(value['hostKey'])
    for key in ['remotePython','remoteProvider','remoteCache','remoteManifest']:
        validate_remote_path(value[key])


def validate_remote_path(path):
    if (not isinstance(path,str) or len(path)>1024 or not re.fullmatch(r'/[A-Za-z0-9_./-]+',path)
                or path=='/' or str(PurePosixPath(path))!=path or '..' in PurePosixPath(path).parts):
        raise ValueError('Canonical pre-provisioned remote paths required')


def validate(value):
    policy=route_policy(value)
    fields=FIELDS if policy is POLICY else FIELDS|{'backendProfile'}
    if policy is POLICY_REBUILD_CHECK:fields=fields|{'remotePublicSource'}
    if set(value)!=fields:raise ValueError('Exact versioned encrypted SSH route required')
    validate_endpoint(value)
    if policy is POLICY_REBUILD_CHECK:validate_remote_path(value['remotePublicSource'])
    for key in ['basePlanId','modelId']:
        if not isinstance(value[key],str) or not SHA.fullmatch(value[key]):
            raise ValueError('Exact SSH model and plan identities required')
    if value['backendProtocol']!=route_backend(value)[1]['protocol']:
        raise ValueError('Unsupported SSH evaluator protocol')
    return canonical(value)


@dataclass(frozen=True)
class SSHRoute:
    encoded: bytes

    def __post_init__(self):
        if type(self.encoded) is not bytes:raise ValueError('Immutable encoded SSH route required')
        if validate(self.definition())!=self.encoded:
            raise ValueError('Canonical immutable SSH route required')

    @classmethod
    def from_file(cls,path):
        value=parse_unique_json(Path(path).read_bytes())
        return cls(validate(value))

    @classmethod
    def for_cache(cls,directory,profile,endpoint,allowed_operators):
        """Derive identities from an installed validated cache, without connecting."""
        if not isinstance(endpoint,dict) or set(endpoint)!=ENDPOINT_FIELDS:
            raise ValueError('Exact explicit SSH endpoint fields required')
        validate_endpoint(endpoint)
        require_operator(endpoint['operatorId'],allowed_operators)
        settings,load,_,_=qwen_backend(profile)
        plan=load(directory)
        value=dict(endpoint,schema=(POLICY_V2 if settings.get('integrity') is None else POLICY_CHECK)['routeSchema'],backendProfile=profile,backendProtocol=settings['protocol'],
                   basePlanId=plan.identity,modelId=plan.definition()['modelId'])
        route=cls(validate(value));route.bind(plan,allowed_operators)
        return route

    @classmethod
    def for_compiled_stream(cls,directory,prepared_public_tiles,endpoint,allowed_operators):
        """Bind a streamed route only after complete byte-derived client admission."""
        if not isinstance(endpoint,dict) or set(endpoint)!=ENDPOINT_FIELDS:raise ValueError('Exact explicit SSH endpoint fields required')
        validate_endpoint(endpoint);require_operator(endpoint['operatorId'],allowed_operators)
        from flash_text_stream_plan import load_client_plan,CONFIG
        plan=load_client_plan(directory,prepared_public_tiles)
        value=dict(endpoint,schema=POLICY_STREAM_CHECK['routeSchema'],backendProfile=CONFIG['profile'],
            backendProtocol=CONFIG['protocol'],basePlanId=plan.identity,modelId=plan.definition()['modelId'])
        route=cls(validate(value));route.bind(plan,allowed_operators)
        return route

    @classmethod
    def for_compiled_checkpoint(cls,directory,prepared_public_tiles,endpoint,allowed_operators):
        if not isinstance(endpoint,dict) or set(endpoint)!=REBUILD_ENDPOINT_FIELDS:raise ValueError('Exact explicit checkpoint SSH endpoint required')
        validate_endpoint(endpoint);validate_remote_path(endpoint['remotePublicSource']);require_operator(endpoint['operatorId'],allowed_operators)
        from flash_text_rebuild_plan import load_client_plan,CONFIG
        plan=load_client_plan(directory,prepared_public_tiles)
        value=dict(endpoint,schema=POLICY_REBUILD_CHECK['routeSchema'],backendProfile=CONFIG['profile'],
            backendProtocol=CONFIG['protocol'],basePlanId=plan.identity,modelId=plan.definition()['modelId'])
        route=cls(validate(value));route.bind(plan,allowed_operators)
        return route

    def write_file(self,path):
        """Private create-only publication; never overwrite an existing route."""
        target=Path(path);temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.encrypted-route-',delete=False) as handle:
                temporary=Path(handle.name)
                os.fchmod(handle.fileno(),0o600);handle.write(self.encoded);handle.flush();os.fsync(handle.fileno())
            os.link(temporary,target)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)

    def definition(self):return json.loads(self.encoded)

    @property
    def identity(self):return digest(canonical(dict(policy=route_policy(self.definition()),route=self.definition())))

    def require_requester(self,allowed_operators):
        require_operator(self.definition()['operatorId'],allowed_operators)

    def bind(self,plan,allowed_operators):
        value=self.definition()
        self.require_requester(allowed_operators)
        if (plan.protocol_id!=value['backendProtocol'] or plan.identity!=value['basePlanId'] or plan.definition()['modelId']!=value['modelId']):
            raise ValueError('SSH endpoint is not bound to the requested complete model plan')
        return BoundSSHPlan(plan,self)


@dataclass(frozen=True)
class BoundSSHPlan:
    base: object
    route: SSHRoute

    @property
    def protocol_id(self):return route_policy(self.route.definition())['protocol']
    @property
    def identity(self):return self.route.identity
    def definition(self):return self.base.definition()
    def schedule(self):return self.base.schedule()
    def frame(self,*args):
        value=self.base.frame(*args)
        value.update(schema=self.protocol_id,policyId=self.identity)
        return value


class SSHChild:
    """Own the temporary public pin until the SSH process has actually exited."""
    def __init__(self,child,directory):
        self.child,self.directory=child,directory
        self.stdin,self.stdout=child.stdin,child.stdout
    def poll(self):return self.child.poll()
    def kill(self):return self.child.kill()
    def wait(self,timeout=None):
        result=self.child.wait(timeout=timeout)
        self.directory.cleanup()
        return result


@dataclass(frozen=True)
class SSHDeployment:
    route: SSHRoute
    identity_file: str
    allowed_operators: tuple

    def bind(self,plan):
        bound=self.route.bind(plan,self.allowed_operators)
        self.private_identity()
        return bound

    def private_identity(self):
        identity=Path(self.identity_file)
        status=identity.stat()
        if (not identity.is_absolute() or not identity.is_file() or '%' in str(identity)
                or status.st_uid!=os.getuid() or status.st_mode&0o077):
            raise ValueError('Owned, private, explicit SSH client identity file required')
        return identity

    def command(self,known_hosts):
        self.route.require_requester(self.allowed_operators)
        value=self.route.definition()
        identity=self.private_identity()
        profile,settings=route_backend(value)
        remote=['env','RAYON_NUM_THREADS='+str(settings['cpuThreads']),
                'OMP_NUM_THREADS='+str(settings['cpuThreads']),value['remotePython'],value['remoteProvider'],
                value['remoteCache'],value['basePlanId']]
        if route_policy(value) is not POLICY:remote+=['--profile',profile]
        if route_policy(value) is POLICY_REBUILD_CHECK:remote+=['--public-source-dir',value['remotePublicSource']]
        remote+=['--ssh-route',value['remoteManifest'],self.route.identity]
        options=['HostKeyAlgorithms=ssh-ed25519','PreferredAuthentications=publickey','GSSAPIAuthentication=no','HostbasedAuthentication=no','BatchMode=yes','StrictHostKeyChecking=yes','UserKnownHostsFile='+str(known_hosts),
                 'GlobalKnownHostsFile=/dev/null','IdentitiesOnly=yes','IdentityAgent=none','ForwardAgent=no',
                 'ClearAllForwardings=yes','PermitLocalCommand=no','RequestTTY=no','PasswordAuthentication=no',
                 'KbdInteractiveAuthentication=no','UpdateHostKeys=no','VerifyHostKeyDNS=no',
                 'ConnectTimeout='+str(route_policy(value)['connectTimeoutSeconds'])]
        return ['/usr/bin/ssh','-F','/dev/null',*[item for option in options for item in ['-o',option]],
                '-p',str(value['port']),'-i',str(identity),value['user']+'@'+value['host'],shlex.join(remote)]

    def spawn(self,plan):
        self.bind(plan)
        directory=tempfile.TemporaryDirectory(prefix='doppler-ssh-pin-')
        try:
            value=self.route.definition()
            host=value['host'] if value['port']==22 else '['+value['host']+']:'+str(value['port'])
            known=Path(directory.name)/'known_hosts'
            known.write_text(host+' '+value['hostKey']['algorithm']+' '+value['hostKey']['base64']+'\n')
            known.chmod(0o600)
            child=subprocess.Popen(self.command(known),stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.DEVNULL,bufsize=0,env={})
            return SSHChild(child,directory)
        except BaseException:
            directory.cleanup()
            raise


def prepare_route(arguments):
    parser=argparse.ArgumentParser(description='Prepare a private pinned research route; never connect or provision a host.')
    parser.add_argument('--cache-dir',required=True);parser.add_argument('--backend-profile',required=True,choices=QWEN_PROFILES)
    parser.add_argument('--endpoint',required=True);parser.add_argument('--allowed-operator',required=True,action='append')
    parser.add_argument('--output',required=True)
    args=parser.parse_args(arguments)
    endpoint=parse_unique_json(Path(args.endpoint).read_bytes())
    route=SSHRoute.for_cache(args.cache_dir,args.backend_profile,endpoint,tuple(args.allowed_operator))
    route.write_file(args.output)
    return route


if __name__=='__main__':
    import sys
    route=prepare_route(sys.argv[1:])
    print(json.dumps(dict(schema=route.definition()['schema'],routeId=route.identity,productionBackendAvailable=False)))
