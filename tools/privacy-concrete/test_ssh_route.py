"""Admission and binding tests require no SSH daemon, private credentials, native FHE or model."""
import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from full_plan import FullPlan, canonical, PROTOCOL
from protocol import check
from ssh_route import SSHRoute, SSHDeployment, public_key, validate


def fixture(**overrides):
    raw=struct.pack('!I',11)+b'ssh-ed25519'+struct.pack('!I',32)+bytes(range(32))
    plan=FullPlan(canonical(dict(modelId='b'*64)))
    value=dict(schema='doppler.encrypted-ssh-route/v1',operatorId='research-provider',host='provider.invalid',port=2222,
               user='research',hostKey=dict(algorithm='ssh-ed25519',base64=base64.b64encode(raw).decode()),
               basePlanId=plan.identity,modelId='b'*64,backendProtocol=PROTOCOL,remotePython='/usr/bin/python3',
               remoteProvider='/srv/doppler/full_provider.py',remoteCache='/srv/doppler/public-cache',remoteManifest='/srv/doppler/route.json')
    return plan,dict(value,**overrides)


class SSHRouteTests(unittest.TestCase):
    def test_complete_endpoint_route_is_immutable_and_bound_before_disclosure(self):
        plan,value=fixture();route=SSHRoute(validate(value))
        bound=route.bind(plan,('research-provider',))
        self.assertNotEqual(bound.identity,plan.identity)
        expected=bound.frame('session','key',1,'result','tile')
        self.assertEqual(bound.protocol_id,expected['schema'])
        self.assertNotEqual(bound.protocol_id,plan.protocol_id)
        check(expected,expected)
        for field,changed in [('host','other.invalid'),('port',2223),('user','other'),('operatorId','other-operator'),('remoteProvider','/srv/doppler/other.py')]:
            _,alternative=fixture(**{field:changed})
            other=SSHRoute(validate(alternative)).bind(plan,(alternative['operatorId'],))
            with self.assertRaises(ValueError):check(other.frame('session','key',1,'result','tile'),expected)
        with self.assertRaises(ValueError):SSHRoute(bytearray(route.encoded))

    def test_host_key_participant_and_plan_changes_cannot_expand_permission(self):
        plan,value=fixture();route=SSHRoute(validate(value))
        with self.assertRaises(ValueError):route.bind(plan,[])
        with self.assertRaises(ValueError):route.bind(FullPlan(canonical(dict(modelId='c'*64))),('research-provider',))
        with self.assertRaises(ValueError):SSHRoute(validate(dict(value,backendProtocol='unqualified')))
        with self.assertRaises(ValueError):SSHRoute(validate(dict(value,attested=True)))
        first=public_key(value['hostKey']);raw=base64.b64decode(value['hostKey']['base64'])
        altered=dict(algorithm='ssh-ed25519',base64=base64.b64encode(raw[:-1]+b'X').decode())
        self.assertNotEqual(first,public_key(altered))
        other=SSHRoute(validate(dict(value,hostKey=altered))).bind(plan,('research-provider',))
        with self.assertRaises(ValueError):check(other.frame('s','k',1,'result','t'),route.bind(plan,('research-provider',)).frame('s','k',1,'result','t'))

    def test_options_paths_and_untrusted_fields_reject_before_any_process(self):
        for change in [dict(host='-oProxyCommand=evil'),dict(host='provider.invalid\nother'),dict(port=True),dict(user='root;evil'),
                       dict(remotePython='/srv/../bin/python'),dict(remoteManifest='/'),dict(remoteCache='/srv/cache;evil'),
                       dict(hostKey={'algorithm':'ssh-ed25519','base64':'bad'}),dict(extra='untrusted')]:
            with patch('ssh_route.subprocess.Popen') as spawn:
                _,value=fixture(**change)
                with self.assertRaises(ValueError):SSHRoute(validate(value))
                spawn.assert_not_called()

    def test_duplicate_manifest_fields_are_rejected(self):
        _,value=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'route.json'
            path.write_text(json.dumps(value)[:-1]+',"host":"replacement.invalid"}')
            with self.assertRaises(ValueError):SSHRoute.from_file(path)

    def test_ssh_command_disables_ambient_configuration_forwarding_and_auth_fallback(self):
        plan,value=fixture()
        with tempfile.TemporaryDirectory() as directory:
            identity=Path(directory)/'identity';identity.write_text('public test placeholder');identity.chmod(0o600)
            deployment=SSHDeployment(SSHRoute(validate(value)),str(identity),('research-provider',))
            bound=deployment.bind(plan);command=deployment.command('/explicit/known_hosts')
            self.assertEqual(command[:3],['/usr/bin/ssh','-F','/dev/null'])
            for option in ['StrictHostKeyChecking=yes','IdentityAgent=none','ForwardAgent=no','ClearAllForwardings=yes',
                           'PasswordAuthentication=no','KbdInteractiveAuthentication=no','BatchMode=yes']:
                self.assertIn(option,command)
            self.assertIn(bound.identity,command[-1]);self.assertNotIn('scp',command)
            identity.chmod(0o644)
            with self.assertRaises(ValueError):deployment.bind(plan)

    def test_spawn_cannot_bypass_requester_or_model_binding(self):
        plan,value=fixture()
        with tempfile.TemporaryDirectory() as directory:
            identity=Path(directory)/'identity';identity.touch(mode=0o600)
            with patch('ssh_route.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):SSHDeployment(SSHRoute(validate(value)),str(identity),()).spawn(plan)
                wrong=FullPlan(canonical(dict(modelId='c'*64)))
                with self.assertRaises(ValueError):SSHDeployment(SSHRoute(validate(value)),str(identity),('research-provider',)).spawn(wrong)
                spawn.assert_not_called()

    def test_failed_spawn_cleans_public_pin_without_fallback(self):
        plan,value=fixture()
        with tempfile.TemporaryDirectory() as directory:
            identity=Path(directory)/'identity';identity.touch(mode=0o600)
            deployment=SSHDeployment(SSHRoute(validate(value)),str(identity),('research-provider',))
            captured=[]
            def reject(command,**options):
                captured.append(Path(next(x.split('=',1)[1] for x in command if x.startswith('UserKnownHostsFile='))))
                self.assertEqual(options['env'],{})
                raise OSError('connection failed')
            with patch('ssh_route.subprocess.Popen',side_effect=reject) as spawn:
                with self.assertRaises(OSError):deployment.spawn(plan)
                self.assertEqual(spawn.call_count,1)
            self.assertFalse(captured[0].exists())


if __name__=='__main__':unittest.main()
