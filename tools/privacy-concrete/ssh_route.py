"""Explicit, pinned SSH endpoint; no source installation, ambient SSH config or plaintext fallback."""
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
from full_plan import FULL_POLICY, PROTOCOL, canonical
from protocol import digest, parse_unique_json

POLICY=json.loads(Path(__file__).with_name('ssh-policy.json').read_text())
FIELDS={'schema','operatorId','host','port','user','hostKey','basePlanId','modelId','backendProtocol',
        'remotePython','remoteProvider','remoteCache','remoteManifest'}
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


def validate(value):
    if not isinstance(value,dict) or set(value)!=FIELDS or value['schema']!=POLICY['routeSchema']:
        raise ValueError('Exact versioned encrypted SSH route required')
    if (not isinstance(value['host'],str) or not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?',value['host'])
            or '..' in value['host'] or type(value['port']) is not int or not 1<=value['port']<=65535
            or not isinstance(value['user'],str) or not re.fullmatch(r'[a-z_][a-z0-9_-]{0,63}',value['user'])
            or not isinstance(value['operatorId'],str) or not re.fullmatch(r'[A-Za-z0-9_.:@/-]{1,128}',value['operatorId'])):
        raise ValueError('Bounded explicit SSH endpoint and operator required')
    for key in ['basePlanId','modelId']:
        if not isinstance(value[key],str) or not SHA.fullmatch(value[key]):
            raise ValueError('Exact SSH model and plan identities required')
    if value['backendProtocol']!=PROTOCOL:
        raise ValueError('Unsupported SSH evaluator protocol')
    public_key(value['hostKey'])
    for key in ['remotePython','remoteProvider','remoteCache','remoteManifest']:
        path=value[key]
        if (not isinstance(path,str) or len(path)>1024 or not re.fullmatch(r'/[A-Za-z0-9_./-]+',path)
                or path=='/' or str(PurePosixPath(path))!=path or '..' in PurePosixPath(path).parts):
            raise ValueError('Canonical pre-provisioned remote paths required')
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

    def definition(self):return json.loads(self.encoded)

    @property
    def identity(self):return digest(canonical(dict(policy=POLICY,route=self.definition())))

    def bind(self,plan,allowed_operators):
        value=self.definition()
        if not isinstance(allowed_operators,(list,tuple)) or value['operatorId'] not in allowed_operators:
            raise ValueError('SSH operator is not explicitly allowed by this requester')
        if (plan.identity!=value['basePlanId'] or plan.definition()['modelId']!=value['modelId']):
            raise ValueError('SSH endpoint is not bound to the requested complete model plan')
        return BoundSSHPlan(plan,self)


@dataclass(frozen=True)
class BoundSSHPlan:
    base: object
    route: SSHRoute

    @property
    def identity(self):return self.route.identity
    def definition(self):return self.base.definition()
    def schedule(self):return self.base.schedule()
    def frame(self,*args):
        value=self.base.frame(*args)
        value.update(schema=POLICY['protocol'],policyId=self.identity)
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
        value=self.route.definition()
        identity=self.private_identity()
        remote=['env','RAYON_NUM_THREADS='+str(FULL_POLICY['cpuThreads']),
                'OMP_NUM_THREADS='+str(FULL_POLICY['cpuThreads']),value['remotePython'],value['remoteProvider'],
                value['remoteCache'],value['basePlanId'],'--ssh-route',value['remoteManifest'],self.route.identity]
        options=['HostKeyAlgorithms=ssh-ed25519','PreferredAuthentications=publickey','GSSAPIAuthentication=no','HostbasedAuthentication=no','BatchMode=yes','StrictHostKeyChecking=yes','UserKnownHostsFile='+str(known_hosts),
                 'GlobalKnownHostsFile=/dev/null','IdentitiesOnly=yes','IdentityAgent=none','ForwardAgent=no',
                 'ClearAllForwardings=yes','PermitLocalCommand=no','RequestTTY=no','PasswordAuthentication=no',
                 'KbdInteractiveAuthentication=no','UpdateHostKeys=no','VerifyHostKeyDNS=no',
                 'ConnectTimeout='+str(POLICY['connectTimeoutSeconds'])]
        return ['/usr/bin/ssh','-F','/dev/null',*[item for option in options for item in ['-o',option]],
                '-p',str(value['port']),'-i',str(identity),value['user']+'@'+value['host'],shlex.join(remote)]

    def spawn(self):
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
