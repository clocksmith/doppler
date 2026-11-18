"""Tests for authenticated encrypted network transport and distinct-owner identity verification in shares_network.js."""
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from full_status import ExecutionRejected
from public_weights import store_public_matrix
from shares import POLICY_ID, ProcessEvaluator, ShareRoute, build_anchor, build_network_commands, parse_evaluator_commands
from weight_encoding import encoding_for_bits

HERE = Path(__file__).resolve().parent
NET_JS = str(HERE / 'shares_network.js')
EVAL_PY = str(HERE / 'shares_evaluator.py')
PLAN_ID = '95c1f06bcbc08954a29696412115c0e5794e4fdf4fa930deaabdee905c1677cd'
MODEL_ID = '4223289199791c6713ba2c98eb8f9d43a37597b09db0b2c778434dec431a6d24'


def make_cache(directory: Path, seed: int = 7, in_w: int = 32, out_w: int = 24):
    rng = np.random.default_rng(seed)
    encoding = encoding_for_bits(12)
    weights = rng.integers(-500, 500, size=(in_w, out_w), dtype=np.int16)
    weight_path, definition = store_public_matrix(
        directory, 'proj.0', in_w, out_w, dict(weightEncoding=encoding['id'], quantizationBits=12), weights.tobytes()
    )
    manifest = dict(
        schema='doppler.two-server-shares-manifest/v1',
        projections=[dict(
            name='proj.0',
            inputWidth=in_w,
            outputWidth=out_w,
            weightId=definition['weightId'],
            file=weight_path.name,
            dtype='<i2',
        )],
    )
    manifest_path = directory / 'shares-manifest.json'
    manifest_path.write_text(json.dumps(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return weights, manifest_sha


def keygen(directory: Path, computer_id: str, participant_id: str, owner_id: str):
    directory.mkdir(parents=True, exist_ok=True)
    priv = directory / f'{computer_id}.priv.json'
    pub = directory / f'{computer_id}.pub.json'
    subprocess.run(
        ['node', NET_JS, 'keygen',
         '--computer-id', computer_id,
         '--participant-id', participant_id,
         '--owner-id', owner_id,
         '--out-private', str(priv),
         '--out-public', str(pub)],
        check=True,
        capture_output=True,
        text=True,
    )
    return priv, json.loads(pub.read_text())


def start_provider_server(priv_path: Path, cache_dir: Path, max_sessions: int = 1):
    proc = subprocess.Popen(
        ['node', NET_JS, 'serve',
         '--listen', '127.0.0.1:0',
         '--identity', str(priv_path),
         '--cache-dir', str(cache_dir),
         '--max-sessions', str(max_sessions),
         '--', sys.executable, EVAL_PY, '--cache-dir', str(cache_dir)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = proc.stdout.readline()
    if not line:
        err = proc.stderr.read()
        proc.kill()
        proc.stdout.close()
        proc.stderr.close()
        raise RuntimeError(f'Provider server failed to start: {err}')
    info = json.loads(line)
    return proc, f"127.0.0.1:{info['port']}"


def stop_provider_server(proc: subprocess.Popen):
    proc.kill()
    try:
        proc.wait(timeout=5)
    finally:
        if proc.stdout: proc.stdout.close()
        if proc.stderr: proc.stderr.close()


class SharesNetworkIdentityTests(unittest.TestCase):
    def test_enrollment_rejects_same_owner_duplicate_key_torsion_keys_and_requester_owner(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            _, pub_a = keygen(tmp, 'comp-a', 'peer-a', 'owner-alpha')
            _, pub_b_same_owner = keygen(tmp, 'comp-b', 'peer-b', 'owner-alpha')
            _, pub_c = keygen(tmp, 'comp-c', 'peer-c', 'owner-beta')

            same_owner_file = tmp / 'same-owner.json'
            same_owner_file.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, pub_b_same_owner],
            )))
            res = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(same_owner_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-b'],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(res.returncode, 0)
            self.assertIn('share enrolled ownerId', res.stderr)

            dup_key_entry = dict(pub_c, publicJwk=pub_a['publicJwk'])
            dup_key_file = tmp / 'dup-key.json'
            dup_key_file.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, dup_key_entry],
            )))
            res_dup = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(dup_key_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-c'],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(res_dup.returncode, 0)
            self.assertIn('Duplicate enrolled Ed25519 public key', res_dup.stderr)

            # Pure torsion, mixed-order (A + (0,-1)), & non-canonical Ed25519 keys must all be rejected.
            identity_pt = base64.urlsafe_b64encode(b'\x01' + b'\x00' * 31).decode().rstrip('=')
            non_canon_pt = base64.urlsafe_b64encode(bytes.fromhex('ee' + 'ff' * 30 + '7f')).decode().rstrip('=')
            raw_a = bytearray(base64.urlsafe_b64decode(pub_a['publicJwk']['x'] + '=='))
            sign_a = (raw_a[31] >> 7) & 1
            raw_a[31] &= 0x7f
            y_a = int.from_bytes(raw_a, 'little')
            p_ed = (1 << 255) - 19
            y_mixed = (p_ed - y_a) % p_ed
            mixed_bytes = bytearray(y_mixed.to_bytes(32, 'little'))
            mixed_bytes[31] |= ((1 - sign_a) << 7)
            mixed_order_pt = base64.urlsafe_b64encode(bytes(mixed_bytes)).decode().rstrip('=')
            for bad_x in (identity_pt, non_canon_pt, mixed_order_pt):
                torsion_file = tmp / 'torsion.json'
                torsion_file.write_text(json.dumps(dict(
                    schema='doppler.two-server-shares-enrollment/v1',
                    operatorIndependenceAttested=False,
                    computers=[pub_a, dict(pub_c, publicJwk=dict(kty='OKP', crv='Ed25519', x=bad_x))],
                )))
                res_bad = subprocess.run(
                    ['node', NET_JS, 'verify-enrollment', '--enrollment', str(torsion_file),
                     '--provider-a', 'comp-a', '--provider-b', 'comp-c'],
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(res_bad.returncode, 0)

            valid_file = tmp / 'valid.json'
            valid_file.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, pub_c],
            )))
            res_req = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(valid_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-c', '--requester-owner', 'owner-alpha'],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(res_req.returncode, 0)
            self.assertIn('matches requesterOwnerId', res_req.stderr)

            # Flag-swallowing / unknown-flag / duplicate-flag regressions must fail closed.
            res_swallow = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(valid_file),
                 '--requester-owner', '--provider-a', 'comp-a', '--provider-b', 'comp-c'],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(res_swallow.returncode, 0)

            res_unknown = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(valid_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-c', '--requester-owner-id', 'owner-alpha'],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(res_unknown.returncode, 0)

            res_ok = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(valid_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-c', '--requester-owner', 'owner-requester'],
                capture_output=True, text=True, check=True,
            )
            parsed = json.loads(res_ok.stdout)
            self.assertTrue(parsed['distinctEnrolledOwnersVerified'])
            self.assertTrue(parsed['requesterOwnerExcluded'])
            self.assertFalse(parsed['nonCollusionEstablished'])

            att_file = tmp / 'attested.json'
            att_file.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=True,
                computers=[pub_a, pub_c],
            )))
            res_att = subprocess.run(
                ['node', NET_JS, 'verify-enrollment', '--enrollment', str(att_file),
                 '--provider-a', 'comp-a', '--provider-b', 'comp-c'],
                capture_output=True, text=True, check=True,
            )
            self.assertTrue(json.loads(res_att.stdout)['nonCollusionEstablished'])

    def test_authenticated_encrypted_route_survives_port_scan_and_produces_exact_product(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cache = tmp / 'cache'
            cache.mkdir()
            # Qwen-sized > 390 KB pipelined payload (24 rows * 2048 cols * 8 bytes) exercises FrameReader backpressure.
            weights, manifest_sha = make_cache(cache, in_w=2048, out_w=512)
            priv_a, pub_a = keygen(tmp / 'a', 'comp-a', 'peer-a', 'owner-a')
            priv_b, pub_b = keygen(tmp / 'b', 'comp-b', 'peer-b', 'owner-b')
            enrollment = tmp / 'enrollment.json'
            enrollment.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, pub_b],
            )))

            srv_a, ep_a = start_provider_server(priv_a, cache, max_sessions=1)
            srv_b, ep_b = start_provider_server(priv_b, cache, max_sessions=1)
            try:
                # Unauthenticated TCP port scan must NOT consume max_sessions=1.
                host_a, port_a = ep_a.rsplit(':', 1)
                with socket.create_connection((host_a, int(port_a)), timeout=2):
                    pass

                cmds, verification = build_network_commands(
                    enrollment,
                    [f'comp-a@{ep_a}', f'comp-b@{ep_b}'],
                    PLAN_ID,
                    MODEL_ID,
                    manifest_sha,
                    requester_owner='owner-req',
                )
                self.assertTrue(verification['distinctEnrolledOwnersVerified'])
                self.assertFalse(verification['nonCollusionEstablished'])

                seed = os.urandom(32)
                anchors = {'proj.0': build_anchor(seed, 'proj.0', weights)}
                route = ShareRoute(
                    tuple(ProcessEvaluator(cmd, label) for cmd, label in zip(cmds, 'ab')),
                    seed,
                    anchors,
                    geometry={'proj.0': (2048, 512)},
                )
                try:
                    inputs = (np.arange(8 * 2048, dtype=np.uint64) % 4096).reshape(8, 2048)
                    product = route.linear('proj.0', inputs, 512, 12)
                    route.finish()
                finally:
                    route.close()
                route.require_integrity()
                expected = inputs.astype(np.int64) @ weights.astype(np.int64)
                np.testing.assert_array_equal(product, expected)
            finally:
                stop_provider_server(srv_a)
                stop_provider_server(srv_b)

    def test_forged_key_swapped_endpoint_owner_mismatch_and_manifest_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cache = tmp / 'cache'
            cache.mkdir()
            weights, manifest_sha = make_cache(cache)
            priv_a, pub_a = keygen(tmp / 'legit-a', 'comp-a', 'peer-a', 'owner-a')
            priv_impostor, _ = keygen(tmp / 'impostor-a', 'comp-a', 'peer-a', 'owner-a')
            priv_wrong_owner = tmp / 'wrong-owner-a.priv.json'
            wrong_owner_doc = json.loads(priv_a.read_text())
            wrong_owner_doc['ownerId'] = 'owner-b'
            priv_wrong_owner.write_text(json.dumps(wrong_owner_doc))
            os.chmod(priv_wrong_owner, 0o600)
            priv_b, pub_b = keygen(tmp / 'legit-b', 'comp-b', 'peer-b', 'owner-b')
            enrollment = tmp / 'enrollment.json'
            enrollment.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, pub_b],
            )))

            # Case 1: Provider A serves with an unenrolled private key (forged signature).
            srv_bad, ep_bad = start_provider_server(priv_impostor, cache)
            srv_b, ep_b = start_provider_server(priv_b, cache)
            try:
                seed = os.urandom(32)
                anchors = {'proj.0': build_anchor(seed, 'proj.0', weights)}
                cmds, _ = build_network_commands(
                    enrollment, [f'comp-a@{ep_bad}', f'comp-b@{ep_b}'], PLAN_ID, MODEL_ID, manifest_sha
                )
                route = ShareRoute(
                    tuple(ProcessEvaluator(cmd, label) for cmd, label in zip(cmds, 'ab')),
                    seed,
                    anchors,
                    geometry={'proj.0': (32, 24)},
                )
                try:
                    with self.assertRaises(ExecutionRejected):
                        route.linear('proj.0', np.ones((2, 32), dtype=np.uint64), 24, 12)
                finally:
                    route.close()
            finally:
                stop_provider_server(srv_bad)
                stop_provider_server(srv_b)

            # Case 2: Swapped endpoints & manifest mismatch against legitimate priv_a, plus self-declared ownerId mismatch.
            srv_a2, ep_a2 = start_provider_server(priv_a, cache, max_sessions=2)
            srv_b2, ep_b2 = start_provider_server(priv_b, cache, max_sessions=2)
            srv_wo, ep_wo = start_provider_server(priv_wrong_owner, cache, max_sessions=1)
            try:
                res_swap = subprocess.run(
                    ['node', NET_JS, 'connect', '--endpoint', ep_b2, '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', manifest_sha],
                    input=b'', capture_output=True, timeout=5, check=False,
                )
                self.assertNotEqual(res_swap.returncode, 0)

                wrong_manifest = '0' * 64
                res_manifest = subprocess.run(
                    ['node', NET_JS, 'connect', '--endpoint', ep_a2, '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', wrong_manifest],
                    input=b'', capture_output=True, timeout=5, check=False,
                )
                self.assertNotEqual(res_manifest.returncode, 0)
                self.assertIn(b'manifest binding', res_manifest.stderr)

                res_owner = subprocess.run(
                    ['node', NET_JS, 'connect', '--endpoint', ep_wo, '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', manifest_sha],
                    input=b'', capture_output=True, timeout=5, check=False,
                )
                self.assertNotEqual(res_owner.returncode, 0)
                self.assertIn(b'owner or manifest binding', res_owner.stderr)
            finally:
                stop_provider_server(srv_a2)
                stop_provider_server(srv_b2)
                stop_provider_server(srv_wo)

    def test_post_handshake_truncation_tamper_and_malformed_stream_survival(self):
        with tempfile.TemporaryDirectory() as raw:
            tmp = Path(raw)
            cache = tmp / 'cache'
            cache.mkdir()
            _, manifest_sha = make_cache(cache)
            priv_a, pub_a = keygen(tmp / 'a', 'comp-a', 'peer-a', 'owner-a')
            _, pub_b = keygen(tmp / 'b', 'comp-b', 'peer-b', 'owner-b')
            enrollment = tmp / 'enrollment.json'
            enrollment.write_text(json.dumps(dict(
                schema='doppler.two-server-shares-enrollment/v1',
                operatorIndependenceAttested=False,
                computers=[pub_a, pub_b],
            )))

            # Malformed SHQ1 payload followed by trailing bytes must NOT crash runServe with EPIPE.
            srv, ep = start_provider_server(priv_a, cache, max_sessions=2)
            try:
                bad_payload = b'BAD!' + b'\x00' * 18 + (b'X' * 200_000)
                res_bad = subprocess.run(
                    ['node', NET_JS, 'connect', '--endpoint', ep, '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', manifest_sha],
                    input=bad_payload, capture_output=True, timeout=5, check=False,
                )
                # Server must stay alive for a subsequent valid SHC1 close session.
                res_clean = subprocess.run(
                    ['node', NET_JS, 'connect', '--endpoint', ep, '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', manifest_sha],
                    input=b'SHC1', capture_output=True, timeout=5, check=False,
                )
                self.assertEqual(res_clean.returncode, 0, f'Server died after malformed session: {res_clean.stderr!r}')
            finally:
                stop_provider_server(srv)

            # TCP proxy that flips 1 bit in the first post-ServerHello encrypted frame -> Poly1305 tag rejection.
            srv2, ep2 = start_provider_server(priv_a, cache, max_sessions=1)
            target_host, target_port = ep2.rsplit(':', 1)
            proxy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            proxy_sock.bind(('127.0.0.1', 0))
            proxy_sock.listen(1)
            proxy_port = proxy_sock.getsockname()[1]
            try:
                proc = subprocess.Popen(
                    ['node', NET_JS, 'connect', '--endpoint', f'127.0.0.1:{proxy_port}', '--enrollment', str(enrollment),
                     '--computer-id', 'comp-a', '--role', 'a', '--plan-id', PLAN_ID, '--model-id', MODEL_ID,
                     '--policy-id', POLICY_ID, '--manifest-sha256', manifest_sha],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                client_conn, _ = proxy_sock.accept()
                upstream = socket.create_connection((target_host, int(target_port)), timeout=5)
                try:
                    # Forward ClientHello -> upstream, ServerHello -> client
                    hdr = client_conn.recv(4)
                    clen = struct.unpack('<I', hdr)[0]
                    upstream.sendall(hdr + client_conn.recv(clen))
                    shdr = upstream.recv(4)
                    slen = struct.unpack('<I', shdr)[0]
                    sbody = b''
                    while len(sbody) < slen:
                        sbody += upstream.recv(slen - len(sbody))
                    client_conn.sendall(shdr + sbody)
                    # Read REC_READY from client, flip last tag bit, forward -> upstream must reject & close.
                    rhdr = client_conn.recv(4)
                    rlen = struct.unpack('<I', rhdr)[0]
                    rbody = bytearray()
                    while len(rbody) < rlen:
                        rbody.extend(client_conn.recv(rlen - len(rbody)))
                    rbody[-1] ^= 0x01
                    upstream.sendall(rhdr + bytes(rbody))
                    # Close upstream without REC_EOF -> client must fail with truncation/auth error.
                    upstream.close()
                    client_conn.close()
                finally:
                    proxy_sock.close()
                _, err = proc.communicate(input=b'SHC1', timeout=5)
                self.assertNotEqual(proc.returncode, 0)
            finally:
                stop_provider_server(srv2)

            with self.assertRaises(ValueError):
                build_network_commands(enrollment, ['comp-a@127.0.0.1:9000', 'comp-b@127.0.0.1:9000'], PLAN_ID, MODEL_ID, manifest_sha)
            with self.assertRaises(ValueError):
                parse_evaluator_commands(['[]', '["ok"]'])


if __name__ == '__main__':
    unittest.main()
