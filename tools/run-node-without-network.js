#!/usr/bin/env node
import { spawn } from 'node:child_process';

// Qualification-only Linux wrapper. libseccomp blocks creation of IPv4/IPv6
// sockets in Node and its descendants; local GPU/driver Unix sockets still work.
// Python and libseccomp are probe prerequisites, not application dependencies.
const source = `
import ctypes, errno, os, socket, sys
lib = ctypes.CDLL('libseccomp.so.2', use_errno=True)
class Compare(ctypes.Structure):
    _fields_ = [('arg', ctypes.c_uint), ('op', ctypes.c_uint), ('a', ctypes.c_uint64), ('b', ctypes.c_uint64)]
lib.seccomp_init.argtypes = [ctypes.c_uint32]
lib.seccomp_init.restype = ctypes.c_void_p
lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Compare)]
lib.seccomp_load.argtypes = [ctypes.c_void_p]
lib.seccomp_release.argtypes = [ctypes.c_void_p]
ctx = lib.seccomp_init(0x7fff0000)
if not ctx: raise RuntimeError('seccomp initialization failed')
try:
    syscall = lib.seccomp_syscall_resolve_name(b'socket')
    if syscall < 0: raise RuntimeError('socket syscall unavailable')
    for family in [socket.AF_INET, socket.AF_INET6]:
        compare = Compare(0, 4, family, 0)
        if lib.seccomp_rule_add_array(ctx, 0x00050000 | errno.EPERM, syscall, 1, ctypes.byref(compare)) != 0:
            raise RuntimeError('seccomp socket rule failed')
    if lib.seccomp_load(ctx) != 0: raise RuntimeError('seccomp filter installation failed')
finally:
    lib.seccomp_release(ctx)
for family in [socket.AF_INET, socket.AF_INET6]:
    try: socket.socket(family)
    except PermissionError: pass
    else: raise RuntimeError('network isolation self-test failed')
os.execv(sys.argv[1], sys.argv[1:])
`;
if (!process.argv[2]) throw new Error('Usage: node tools/run-node-without-network.js <node-script> [...args]');
const child = spawn('python3', ['-c', source, process.execPath, ...process.argv.slice(2)], { stdio: 'inherit' });
child.on('error', error => { console.error(error.message); process.exitCode = 1; });
child.on('exit', (code, signal) => { process.exitCode = signal ? 1 : code; });
