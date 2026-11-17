"""Admission and binding tests require no SSH daemon, private credentials, native FHE or model."""
import base64
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace
from full_plan import FullPlan, canonical, PROTOCOL
from protocol import check
from ssh_route import SSHRoute, SSHDeployment, public_key, validate, ENDPOINT_FIELDS, prepare_route


def fixture(**overrides):
    raw=struct.pack('!I',11)+b'ssh-ed25519'+struct.pack('!I',32)+bytes(range(32))
    plan=FullPlan(canonical(dict(modelId='b'*64)))
    value=dict(schema='doppler.encrypted-ssh-route/v1',operatorId='research-provider',host='provider.invalid',port=2222,
               user='research',hostKey=dict(algorithm='ssh-ed25519',base64=base64.b64encode(raw).decode()),
               basePlanId=plan.identity,modelId='b'*64,backendProtocol=PROTOCOL,remotePython='/usr/bin/python3',
               remoteProvider='/srv/doppler/full_provider.py',remoteCache='/srv/doppler/public-cache',remoteManifest='/srv/doppler/route.json')
    return plan,dict(value,**overrides)


class SSHRouteTests(unittest.TestCase):
    def test_route_preparation_derives_complete_cache_binding_and_preserves_backend(self):
        from plan_backends import QWEN_PROFILES,qwen_backend
        base,value=fixture();endpoint={name:value[name] for name in ENDPOINT_FIELDS}
        for profile in QWEN_PROFILES:
            settings,_,_,_=qwen_backend(profile)
            plan=SimpleNamespace(identity=base.identity,protocol_id=settings['protocol'],definition=base.definition)
            load=Mock(return_value=plan)
            with patch('ssh_route.qwen_backend',return_value=(settings,load,None,None)),patch('ssh_route.subprocess.Popen') as spawn:
                route=SSHRoute.for_cache('installed-cache',profile,endpoint,('research-provider',))
                load.assert_called_once_with('installed-cache');spawn.assert_not_called()
            actual=route.definition()
            self.assertEqual(actual['basePlanId'],base.identity);self.assertEqual(actual['modelId'],'b'*64)
            self.assertEqual(actual['backendProfile'],profile);self.assertEqual(actual['backendProtocol'],settings['protocol'])
            self.assertEqual(actual['schema'],'doppler.encrypted-ssh-check-route/v1' if settings.get('integrity') is not None else 'doppler.encrypted-ssh-route/v2')
            original_key=endpoint['hostKey']['base64'];endpoint['host']='different.invalid';endpoint['hostKey']['base64']='changed'
            self.assertEqual(route.definition()['host'],'provider.invalid')
            self.assertEqual(route.definition()['hostKey']['base64'],original_key)
            endpoint['host']='provider.invalid';endpoint['hostKey']['base64']=original_key

    def test_route_preparation_rejects_disallowed_or_malformed_endpoint_before_cache_or_process(self):
        _,value=fixture();endpoint={name:value[name] for name in ENDPOINT_FIELDS}
        for endpoint_value,allowed in [(endpoint,()),(dict(endpoint,host='-oProxyCommand=evil'),('research-provider',)),
                (dict(endpoint,attested=True),('research-provider',)),(dict(endpoint,basePlanId='a'*64),('research-provider',)),
                (endpoint,('research-provider',None))]:
            with patch('ssh_route.qwen_backend') as backend,patch('ssh_route.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):SSHRoute.for_cache('unused','qwen-all-linear',endpoint_value,allowed)
                backend.assert_not_called();spawn.assert_not_called()
        with self.assertRaises(ValueError):SSHRoute.for_cache('unused','advertised-backend',endpoint,('research-provider',))

    def test_private_route_output_is_atomic_create_only_and_preserves_existing_symlink(self):
        import stat
        _,value=fixture();route=SSHRoute(validate(value))
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'route.json';route.write_file(target)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode),0o600)
            self.assertEqual(SSHRoute.from_file(target),route)
            with self.assertRaises(FileExistsError):route.write_file(target)
            alias=Path(directory)/'alias.json';alias.symlink_to(target)
            with self.assertRaises(FileExistsError):route.write_file(alias)
            self.assertTrue(alias.is_symlink());self.assertEqual(target.read_bytes(),route.encoded)
            self.assertEqual(set(path.name for path in Path(directory).iterdir()),{'route.json','alias.json'})
            with patch('ssh_route.os.link',side_effect=OSError('simulated publication failure')):
                with self.assertRaises(OSError):route.write_file(Path(directory)/'failed.json')
            self.assertEqual(set(path.name for path in Path(directory).iterdir()),{'route.json','alias.json'})

    def test_preparation_cli_uses_duplicate_rejecting_input_and_never_reads_credentials(self):
        from plan_backends import qwen_backend
        plan,value=fixture();endpoint={name:value[name] for name in ENDPOINT_FIELDS}
        settings,_,_,_=qwen_backend('qwen-all-linear')
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'endpoint.json';target=Path(directory)/'route.json'
            arguments=['--cache-dir','installed','--backend-profile','qwen-all-linear','--endpoint',str(source),
                       '--allowed-operator','research-provider','--output',str(target)]
            source.write_text(json.dumps(endpoint)[:-1]+',"host":"replacement.invalid"}')
            with patch('ssh_route.qwen_backend') as backend:
                with self.assertRaises(ValueError):prepare_route(arguments)
                backend.assert_not_called();self.assertFalse(target.exists())
            source.write_text(json.dumps(endpoint))
            with patch('ssh_route.qwen_backend',return_value=(settings,Mock(return_value=plan),None,None)), \
                 patch.object(SSHDeployment,'private_identity',side_effect=AssertionError('credential access')), \
                 patch('ssh_route.subprocess.Popen') as spawn:
                route=prepare_route(arguments);spawn.assert_not_called()
            self.assertEqual(route,SSHRoute.from_file(target))

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

    def test_version_two_pins_each_installed_qwen_profile_without_expanding_legacy_routes(self):
        from plan_backends import QWEN_PROFILES,qwen_backend
        for profile in QWEN_PROFILES:
            with self.subTest(profile=profile),tempfile.TemporaryDirectory() as directory:
                base,value=fixture();settings,_,_,_=qwen_backend(profile)
                plan=SimpleNamespace(identity=base.identity,protocol_id=settings['protocol'],definition=base.definition,frame=base.frame,schedule=base.schedule)
                checked=settings.get('integrity') is not None
                schema='doppler.encrypted-ssh-check-route/v1' if checked else 'doppler.encrypted-ssh-route/v2'
                value.update(schema=schema,backendProfile=profile,backendProtocol=settings['protocol'])
                # A checked backend cannot ride the replay-named route, nor the reverse.
                other='doppler.encrypted-ssh-route/v2' if checked else 'doppler.encrypted-ssh-check-route/v1'
                with self.assertRaises(ValueError):validate(dict(value,schema=other))
                route=SSHRoute(validate(value));bound=route.bind(plan,('research-provider',))
                identity=Path(directory)/'identity';identity.touch(mode=0o600)
                command=SSHDeployment(route,str(identity),('research-provider',)).command('/explicit/known_hosts')
                self.assertIn('--profile '+profile,command[-1]);self.assertIn(bound.identity,command[-1])
                self.assertEqual(bound.protocol_id,'concrete-glwe-qwen-research-ssh-linear-check/v1' if checked else 'concrete-glwe-qwen-research-ssh-replay/v2')
                self.assertEqual(bound.protocol_id.endswith('-linear-check/v1'),checked)
                self.assertEqual(bound.frame('s','k',1,'evaluate','t')['schema'],bound.protocol_id)
                with self.assertRaises(ValueError):route.bind(plan,())
                if profile!='qwen-all-linear':
                    with self.assertRaises(ValueError):validate(dict(value,schema='doppler.encrypted-ssh-route/v1'))
                    with self.assertRaises(ValueError):validate(dict(fixture()[1],backendProtocol=settings['protocol']))

    def test_version_two_rejects_profile_changes_unknown_backends_and_protocol_downgrade(self):
        from plan_backends import qwen_backend
        base,value=fixture();settings,_,_,_=qwen_backend('qwen-all-linear-ntt12')
        value.update(schema='doppler.encrypted-ssh-route/v2',backendProfile='qwen-all-linear-ntt12',backendProtocol=settings['protocol'])
        for changes in [dict(backendProfile='qwen-all-linear'),dict(backendProtocol=PROTOCOL),dict(backendProfile='flash-first-ple-ntt'),dict(backendProfile=None),dict(backendProfile='advertised')]:
            with self.subTest(changes=changes),patch('ssh_route.subprocess.Popen') as spawn:
                with self.assertRaises(ValueError):SSHRoute(validate(dict(value,**changes)))
                spawn.assert_not_called()
        with self.assertRaises(ValueError):validate({key:entry for key,entry in value.items() if key!='backendProfile'})
        with self.assertRaises(ValueError):validate(dict(fixture()[1],backendProfile='qwen-all-linear'))
        route=SSHRoute(validate(value))
        with self.assertRaises(ValueError):route.bind(base,('research-provider',))

    def test_host_key_participant_and_plan_changes_cannot_expand_permission(self):
        plan,value=fixture();route=SSHRoute(validate(value))
        with self.assertRaises(ValueError):route.bind(plan,[])
        with self.assertRaises(ValueError):route.bind(FullPlan(canonical(dict(modelId='c'*64))),('research-provider',))
        from types import SimpleNamespace
        mismatched=SimpleNamespace(identity=plan.identity,protocol_id='different-native-protocol',definition=plan.definition)
        with self.assertRaises(ValueError):route.bind(mismatched,('research-provider',))
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
