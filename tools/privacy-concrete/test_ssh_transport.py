"""Opt-in real SSH loopback authentication/framing; no model or FHE workload."""
import base64
import getpass
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import time
import unittest
from protocol import send, receive
from ssh_route import SSHDeployment, SSHRoute, validate
from test_ssh_route import fixture


@unittest.skipUnless(os.environ.get('DOPPLER_SSH_INTEGRATION')=='1','opt-in isolated SSH daemon')
class SSHTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory=tempfile.TemporaryDirectory(prefix='doppler-ssh-test-')
        cls.addClassCleanup(cls.directory.cleanup)
        cls.root=Path(cls.directory.name)
        for name in ['host','client','unallowed-client']:
            subprocess.run(['/usr/bin/ssh-keygen','-q','-t','ed25519','-N','','-f',str(cls.root/name)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        (cls.root/'authorized_keys').write_bytes((cls.root/'client.pub').read_bytes())
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));cls.port=sock.getsockname()[1]
        cls.log=(cls.root/'server.log').open('wb')
        cls.addClassCleanup(cls.log.close)
        config=cls.root/'sshd_config'
        config.write_text('\n'.join(['ListenAddress 127.0.0.1','Port '+str(cls.port),'HostKey '+str(cls.root/'host'),
            'PidFile '+str(cls.root/'pid'),'AuthorizedKeysFile '+str(cls.root/'authorized_keys'),
            'AllowUsers '+getpass.getuser(),'StrictModes no','UsePAM no','PasswordAuthentication no',
            'KbdInteractiveAuthentication no','PubkeyAuthentication yes','AllowAgentForwarding no',
            'AllowTcpForwarding no','X11Forwarding no','ForceCommand /bin/cat','LogLevel ERROR'])+'\n')
        daemon=shutil.which('sshd') or '/usr/sbin/sshd'
        cls.server=subprocess.Popen([daemon,'-D','-e','-f',str(config)],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=cls.log)
        cls.addClassCleanup(cls.stop_server)
        for _ in range(100):
            if cls.server.poll() is not None:
                cls.log.flush()
                raise RuntimeError('Isolated SSH daemon failed: '+(cls.root/'server.log').read_text()[:2000])
            try:
                with socket.create_connection(('127.0.0.1',cls.port),timeout=.1):break
            except OSError:time.sleep(.02)
        else:raise RuntimeError('Isolated SSH daemon did not become ready')
        algorithm,key=(cls.root/'host.pub').read_text().split()[:2]
        cls.plan,cls.value=fixture(host='127.0.0.1',port=cls.port,user=getpass.getuser(),hostKey=dict(algorithm=algorithm,base64=key))

    @classmethod
    def stop_server(cls):
        if cls.server.poll() is None:cls.server.terminate()
        cls.server.wait(timeout=5)

    def deployment(self,value=None,identity='client'):
        return SSHDeployment(SSHRoute(validate(self.value if value is None else value)),str(self.root/identity),('research-provider',))

    def test_real_pinned_ssh_transports_bound_opaque_frames(self):
        deployment=self.deployment();bound=deployment.bind(self.plan)
        child=deployment.spawn(self.plan)
        try:
            header=bound.frame('public-session','public-key-context',1,'evaluate','public-tile')
            payload=b'opaque-public-ciphertext-fixture\x00\xff'
            send(child.stdin,header,payload,timeout=10)
            echoed,data,_=receive(child.stdout,timeout=10)
            self.assertEqual(echoed,header);self.assertEqual(data,payload)
            child.stdin.close();self.assertEqual(child.wait(timeout=5),0)
            self.assertFalse(Path(child.directory.name).exists())
        finally:
            if child.poll() is None:child.kill()
            child.wait()
            for stream in [child.stdin,child.stdout]:
                if not stream.closed:stream.close()

    def test_version_two_research_profile_transports_bound_frames_with_real_authentication(self):
        from types import SimpleNamespace
        from plan_backends import QWEN_PROFILES,qwen_backend
        for profile in QWEN_PROFILES:
            with self.subTest(profile=profile):
                settings,_,_,_=qwen_backend(profile)
                plan=SimpleNamespace(identity=self.plan.identity,protocol_id=settings['protocol'],definition=self.plan.definition,frame=self.plan.frame,schedule=self.plan.schedule)
                checked=settings.get('integrity') is not None
                schema='doppler.encrypted-ssh-check-route/v1' if checked else 'doppler.encrypted-ssh-route/v2'
                value=dict(self.value,schema=schema,backendProfile=profile,backendProtocol=settings['protocol'])
                deployment=self.deployment(value);bound=deployment.bind(plan);child=deployment.spawn(plan)
                try:
                    header=bound.frame('public-session','public-key-context',1,'evaluate','public-tile')
                    # A checked backend's frames never carry the replay protocol name.
                    self.assertEqual(header['schema'].endswith('-linear-check/v1'),checked)
                    payload=b'opaque-public-v2-ciphertext-fixture'
                    send(child.stdin,header,payload,timeout=10);echoed,data,_=receive(child.stdout,timeout=10)
                    self.assertEqual(echoed,header);self.assertEqual(data,payload)
                    child.stdin.close();self.assertEqual(child.wait(timeout=5),0)
                finally:
                    if child.poll() is None:child.kill()
                    child.wait()
                    for stream in [child.stdin,child.stdout]:
                        if not stream.closed:stream.close()

    def test_wrong_host_pin_fails_without_remote_output_or_local_fallback(self):
        raw=base64.b64decode(self.value['hostKey']['base64'])
        wrong=dict(self.value,hostKey=dict(algorithm='ssh-ed25519',base64=base64.b64encode(raw[:-1]+bytes([raw[-1]^1])).decode()))
        self.reject(self.deployment(wrong))

    def test_unallowed_client_key_fails_authentication_without_remote_output(self):
        self.reject(self.deployment(identity='unallowed-client'))

    def reject(self,deployment):
        child=deployment.spawn(self.plan)
        try:
            child.stdin.close()
            self.assertNotEqual(child.wait(timeout=10),0)
            self.assertEqual(child.stdout.read(),b'')
            self.assertFalse(Path(child.directory.name).exists())
        finally:
            if child.poll() is None:child.kill()
            child.wait();child.stdout.close()


if __name__=='__main__':unittest.main()
