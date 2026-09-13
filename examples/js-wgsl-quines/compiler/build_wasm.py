"""Three direct-code backends for the same nineteen-instruction Sprout.

Loaded by build_compiler.py after its JS/WGSL templates. This bootstrap is not
needed by any descendant. Binary templates describe instructions and generic
runtime helpers, never a precompiled copy of the compiler.
"""
import gzip
import hashlib
import re


def replace_once(pattern, replacement, value):
    result, count = re.subn(pattern, replacement, value)
    assert count == 1, (pattern, count)
    return result


# Only compiler descendants need binary output. The other four constructions
# continue to embed the original, strictly ASCII WebGPU host unchanged.
COMPILER_HOST = replace_once(
    r'function decodeWords\(words\)',
    'function decodeWords(words, binary = false)', HOST)
COMPILER_HOST = replace_once(
    r'code\s*>\s*127', 'code > (binary ? 255 : 127)', COMPILER_HOST)
COMPILER_HOST = replace_once(
    r'return new TextDecoder\("utf-8",\s*\{\s*fatal:\s*true\s*\}\)\.decode\(bytes\);',
    'return binary ? bytes : new TextDecoder("utf-8", { fatal: true }).decode(bytes);',
    COMPILER_HOST)
COMPILER_HOST = replace_once(
    r'(async function dispatchText\([^)]*input\s*=\s*null)\)',
    r'\1, binary = false)', COMPILER_HOST)
COMPILER_HOST = replace_once(
    r'return\s*\{\s*text:\s*decodeWords\(words\),\s*words\s*\};',
    'return binary ? { bytes: decodeWords(words, true), words } : { text: decodeWords(words), words };',
    COMPILER_HOST)

JS_PARTS[2] = JS_PARTS[2].replace('function execute(', 'function executeBytes(', 1)
JS_PARTS[2] = replace_once(
    r'target\s*!==\s*0\s*&&\s*target\s*!==\s*1',
    'target!==0&&target!==1&&target!==2', JS_PARTS[2])
JS_PARTS[2] = JS_PARTS[2].replace('Use target 0 (JS) or 1 (WGSL).',
                                      'Use target 0 (JS), 1 (WGSL), or 2 (Wasm).')
JS_PARTS[2] = replace_once(r'c\s*>\s*127', 'c>255', JS_PARTS[2])
# Quote Sprout words, not compiled code, with five payload bits per character.
# ASCII 48..112 excluding backslash needs no escaping in a JS string. The high
# half of the alphabet continues a word. Keep the complete launcher in the seed.
JS_PARTS[1] = ';\nconst P = Object.freeze(unpack("'
assert JS_PARTS[2].startswith(']);\n')
JS_PARTS[2] = '"));\n' + r'''function unpack(text) {
  const words = [];
  let word = 0, place = 1;
  for (const ch of text) {
    let digit = ch.charCodeAt(0) - 48;
    if (digit > 43) digit--;
    word += (digit & 31) * place;
    if (digit < 32) { words.push(word); word = 0; place = 1; }
    else place *= 32;
  }
  return words;
}
''' + JS_PARTS[2][4:]
JS_PARTS[3] = r'''
  return Uint8Array.from(output);
}
function execute(input=P,target=0) {
  const bytes=executeBytes(input,target);
  return target===2?bytes:new TextDecoder("utf-8",{fatal:true}).decode(bytes);
}
''' + COMPILER_HOST + (ROOT / 'lib/wasm.js').read_text() + PRINT_JS + r'''
const targetNumber=target=>["js","wgsl","wasm"].indexOf(target);
const api={id:"compiler",program:P,execute,executeBytes,
  self:()=>execute(P,0),wgsl:()=>execute(P,1),wasm:()=>execute(P,2),
  compile:(program,target="js")=>execute(program,targetNumber(target)),
  fromWasm:instantiateSprout,
  run:(target="js",input=null,shader=null)=>{
    const selected=targetNumber(target);
    if(selected<0)throw new Error("Use js, wgsl, or wasm.");
    if(input!==null)validateProgram(input);
    return withDevice(device=>dispatchText(device,shader??execute(P,1),"main",
      {TARGET:selected,EXTERNAL:input===null?0:1},input??[],selected===2));
  }
};
const DEFAULT_GPU=false;
''' + AUTORUN.replace(
    'const task=',
    'const task=args.includes("--wasm")?Promise.resolve({text:api.wasm()}):', 1) + '\n})();\n'

binary_runtime = replace_once(r'c\s*>\s*127u', 'c > 255u', WG_RUNTIME)
binary_runtime = replace_once(r'TARGET\s*>\s*1u', 'TARGET > 2u', binary_runtime)
WG_PARTS[2] = WG_PARTS[2].replace(WG_RUNTIME, binary_runtime)


def uleb(n):
    result = bytearray()
    while n >= 128:
        result.append((n & 127) | 128)
        n >>= 7
    result.append(n)
    return bytes(result)


def sleb(n):
    n = n if n < 0x80000000 else n - 0x100000000
    result = bytearray()
    while True:
        part = n & 127
        n >>= 7
        done = (n == 0 and not part & 64) or (n == -1 and part & 64)
        result.append(part if done else part | 128)
        if done:
            return bytes(result)


def padded(n, signed=False):
    top = n >> 28
    return bytes([((n >> shift) & 127) | 128 for shift in (0, 7, 14, 21)] +
                 [top | (112 if signed and top >= 8 else 0)])


def section(kind, payload):
    return bytes([kind]) + uleb(len(payload)) + payload


def get(n):
    return b'\x20' + uleb(n)


def set_(n):
    return b'\x21' + uleb(n)


def const(n):
    return b'\x41' + sleb(n)


def binary(left, right, opcode):
    return left + right + bytes([opcode])


def if_(condition, yes, no=None, result=False):
    return condition + bytes([4, 127 if result else 64]) + yes + (
        b'\x05' + no if no is not None else b'') + b'\x0b'


def while_(condition, body):
    return b'\x02\x40\x03\x40' + condition + b'\x45\x0d\x01' + body + b'\x0c\x00\x0b\x0b'


def trap_if(condition):
    return if_(condition, b'\x00')


def load(address):
    return address + b'\x28\x02\x00'


def store_word(address, value):
    return address + value + b'\x36\x02\x00'


def function_body(local_count, body):
    locals_ = b'\x01' + uleb(local_count) + b'\x7f' if local_count else b'\x00'
    body = locals_ + body + b'\x0b'
    return uleb(len(body)) + body


OUT, INPUT, STACK = 262144, 524288, 786432
ADD, SUB, MUL, DIV, REM = 0x6a, 0x6b, 0x6c, 0x6e, 0x70
EQ, NE, LT, GT, LE, GE, AND, OR = 0x46, 0x47, 0x49, 0x4b, 0x4d, 0x4f, 0x71, 0x72

put_body = (
    trap_if(binary(get(0), const(255), GT)) +
    trap_if(binary(b'\x23\x00', const(OUT + 262143), GE)) +
    b'\x23\x00' + get(0) + b'\x3a\x00\x00' +
    b'\x23\x00' + const(1) + b'\x6a\x24\x00')
decimal_body = const(1) + set_(1) + while_(
    binary(binary(get(0), get(1), DIV), const(10), GE),
    binary(get(1), const(10), MUL) + set_(1)) + while_(get(1),
    binary(binary(get(0), get(1), DIV), const(48), ADD) + b'\x10\x00' +
    binary(get(0), get(1), REM) + set_(0) +
    binary(get(1), const(10), DIV) + set_(1))

# Validator parameters: target, pointer, length. Locals: pc, data length,
# code length, opcode, a, b, c, depth, top. It inspects but never executes code.
word = lambda index: load(binary(get(1), binary(index, const(4), MUL), ADD))
top_address = binary(const(STACK), binary(binary(get(10), const(1), SUB), const(4), MUL), ADD)
push_address = binary(const(STACK), binary(get(10), const(4), MUL), ADD)
eq_op = lambda n: binary(get(6), const(n), EQ)
is_between = lambda reg, lo, hi: binary(binary(get(reg), const(lo), GE), binary(get(reg), const(hi), LE), AND)
validate_body = (
    trap_if(binary(get(0), const(2), GT)) +
    trap_if(binary(binary(get(1), const(0), NE), binary(get(1), const(INPUT), NE), AND)) +
    trap_if(binary(binary(get(2), const(3), LT), binary(get(2), const(60000), GT), OR)) +
    trap_if(binary(word(const(0)), const(21331), NE)) +
    word(const(1)) + set_(4) + word(const(2)) + set_(5) +
    trap_if(binary(binary(get(4), get(2), GT), binary(get(5), get(2), GT), OR)) +
    trap_if(binary(get(5), const(4), REM)) +
    trap_if(binary(binary(binary(const(3), get(4), ADD), get(5), ADD), get(2), NE)) +
    binary(const(3), get(4), ADD) + set_(3))
validate_loop = b''.join(word(binary(get(3), const(i), ADD)) + set_(6 + i) for i in range(4))
validate_loop += (
    trap_if(binary(binary(get(6), const(18), GT), binary(get(7), const(64), GE), OR)) +
    trap_if(binary(is_between(6, 2, 11), binary(get(8), const(64), GE), AND)) +
    trap_if(binary(is_between(6, 3, 9), binary(get(9), const(64), GE), AND)) +
    if_(binary(eq_op(14), eq_op(16), OR),
        trap_if(binary(get(10), const(64), GE)) +
        store_word(push_address, get(6)) + binary(get(10), const(1), ADD) + set_(10),
        if_(eq_op(17),
            trap_if(get(10) + b'\x45') + trap_if(binary(load(top_address), const(16), NE)) +
            store_word(top_address, const(17)),
            if_(binary(eq_op(15), eq_op(18), OR),
                trap_if(get(10) + b'\x45') + load(top_address) + set_(11) +
                if_(eq_op(15), trap_if(binary(get(11), const(14), NE)),
                    trap_if(binary(binary(get(11), const(16), NE), binary(get(11), const(17), NE), AND))) +
                binary(get(10), const(1), SUB) + set_(10)))) +
    binary(get(3), const(4), ADD) + set_(3))
validate_body += while_(binary(get(3), get(2), LT), validate_loop) + trap_if(get(10))

types = b'\x03\x60\x01\x7f\x00\x60\x03\x7f\x7f\x7f\x00\x60\x03\x7f\x7f\x7f\x01\x7f'
globals_ = b'\x03' + b''.join(bytes([127, mutable]) + const(value) + b'\x0b'
                             for mutable, value in [(1, OUT), (0, OUT), (0, INPUT)])
exports = b'\x04' + b''.join(uleb(len(name)) + name.encode('ascii') + bytes([kind, index])
    for name, kind, index in [('memory', 2, 0), ('run', 0, 3), ('output', 3, 1), ('input', 3, 2)])
prefix = (b'\x00asm\x01\x00\x00\x00' + section(1, types) + section(3, b'\x04\x00\x00\x01\x02') +
          section(5, b'\x01\x00\x10') + section(6, globals_) + section(7, exports) + b'\x0a')
helpers = b'\x04' + function_body(0, put_body) + function_body(1, decimal_body) + function_body(9, validate_body)
WASM_PARTS = [prefix, helpers,
    b'\x01\x40\x7f' + get(1) + b'\x45\x04\x40\x41',
    set_(2) + b'\x0b' + const(OUT) + b'\x24\x00' + get(0) + get(1) + get(2) + b'\x10\x02' +
        get(0) + set_(3) + get(2) + set_(4),
    b'\x23\x00' + const(OUT) + b'\x6b\x0b']

# 0xff is a builder marker, never a Wasm opcode here: a/b/c are native local
# indices; i is an immediate; n is the embedded program's word count.
reg = lambda letter: b'\xff' + letter.encode('ascii')
rg = lambda letter: b'\x20' + reg(letter)
rs = lambda letter: b'\x21' + reg(letter)
WASM_OP = [
    b'\x01', b'\x41' + reg('i') + rs('a'), rg('b') + rs('a'),
    *[binary(rg('b'), rg('c'), op) + rs('a') for op in (ADD, SUB, MUL)],
    *[if_(rg('c'), binary(rg('b'), rg('c'), op), const(0), True) + rs('a') for op in (DIV, REM)],
    *[binary(rg('b'), rg('c'), op) + rs('a') for op in (EQ, LT)],
    if_(binary(rg('b'), b'\x41' + reg('n'), LT),
        load(binary(rg('b'), const(4), MUL)), const(0), True) + rs('a'),
    if_(binary(rg('b'), get(2), LT),
        load(binary(get(1), binary(rg('b'), const(4), MUL), ADD)), const(0), True) + rs('a'),
    rg('a') + b'\x10\x00', rg('a') + b'\x10\x01',
    b'\x02\x40\x03\x40' + rg('a') + b'\x45\x0d\x01',
    b'\x0c\x00\x0b\x0b', rg('a') + b'\x04\x40', b'\x05', b'\x0b']
assert len(WASM_OP) == len(OPS) == 19


def expand_wasm(template, operands, length):
    result = bytearray()
    i = 0
    while i < len(template):
        byte = template[i]
        i += 1
        if byte != 255:
            result.append(byte)
            continue
        marker = chr(template[i])
        i += 1
        if marker in 'abc':
            result.append(operands[ord(marker) - ord('a')] + 3)
        else:
            result.extend(padded(operands[1] if marker == 'i' else length, True))
    return bytes(result)


WASM_LENGTHS = bytes(len(expand_wasm(template, (0, 0, 0), 0)) for template in WASM_OP)
ALL_TABLES = JS_PARTS + WG_PARTS + JS_OP + WG_OP + WASM_PARTS + WASM_OP + [WASM_LENGTHS]
DESC_WORDS = 2 * len(ALL_TABLES)
data = [0] * DESC_WORDS
for index, template in enumerate(ALL_TABLES):
    value = template.encode('ascii') if isinstance(template, str) else template
    data[index * 2:index * 2 + 2] = [3 + len(data), len(value)]
    data.extend(value)

code, listing = [], []


def ins(op, a=0, b=0, c=0):
    code.extend([OPNUM[op], a, b, c])
    listing.append(f'{op} {a} {b} {c}')


def note(text):
    listing.append('; ' + text)


def emit_piece(k):
    ins('set', 5, 2 * k)
    ins('add', 5, 4, 5)
    ins('data', 6, 5)
    ins('add', 5, 5, 3)
    ins('data', 7, 5)
    ins('set', 8, 0)
    ins('lt', 9, 8, 7)
    ins('while', 9)
    ins('add', 11, 6, 8)
    ins('data', 10, 11)
    ins('put', 10)
    ins('add', 8, 8, 3)
    ins('lt', 9, 8, 7)
    ins('end')


def input_start():
    ins('input', 13, 3)
    ins('set', 11, 3)
    ins('add', 13, 13, 11)


def instruction_start():
    for offset, destination in enumerate((14, 15, 16, 17)):
        ins('set', 11, offset)
        ins('add', 11, 13, 11)
        ins('input', destination, 11)
    ins('mul', 5, 14, 21)
    ins('add', 5, 5, 12)
    ins('data', 6, 5)
    ins('add', 5, 5, 3)
    ins('data', 7, 5)
    ins('set', 8, 0)
    ins('lt', 9, 8, 7)
    ins('while', 9)
    next_character()


def next_character():
    ins('add', 11, 6, 8)
    ins('data', 10, 11)
    ins('add', 8, 8, 3)


def instruction_end():
    ins('lt', 9, 8, 7)
    ins('end')
    ins('add', 13, 13, 22)
    ins('lt', 18, 13, 1)
    ins('end')


def emit_leb(source, signed=False):
    ins('mov', 24, source)
    ins('set', 29, 4)
    ins('while', 29)
    ins('mod', 25, 24, 26)
    ins('add', 25, 25, 26)
    ins('put', 25)
    ins('div', 24, 24, 26)
    ins('sub', 29, 29, 3)
    ins('end')
    if signed:
        ins('lt', 30, 24, 31)
        ins('eq', 30, 30, 2)
        ins('if', 30)
        ins('add', 24, 24, 28)
        ins('fi')
    ins('put', 24)


def literal(values):
    for value in values:
        ins('set', 10, value)
        ins('put', 10)


note('The same 19-opcode compiler emits direct JS, WGSL, or Wasm.')
for register, value in ((2, 0), (3, 1), (21, 2), (22, 4)):
    ins('set', register, value)
ins('lt', 19, 0, 21)
ins('if', 19)
note('Text backends share decimal quotation and operand substitution.')
ins('set', 20, 126)
ins('set', 23, 8)
ins('mul', 4, 0, 23)
ins('set', 11, 3)
ins('add', 4, 4, 11)
ins('set', 12, 2 * len(OPS))
ins('mul', 12, 12, 0)
ins('set', 11, 19)
ins('add', 12, 12, 11)
emit_piece(0)
ins('decimal', 1)
emit_piece(1)
ins('set', 50, 32)
ins('set', 51, 44)
ins('set', 52, 48)
ins('set', 8, 0)
ins('lt', 9, 8, 1)
ins('while', 9)
ins('if', 0)
ins('if', 8)
literal([44])
ins('fi')
ins('input', 10, 8)
ins('decimal', 10)
ins('else')
ins('input', 47, 8)
ins('set', 48, 1)
ins('while', 48)
ins('mod', 49, 47, 50)
ins('div', 47, 47, 50)
ins('if', 47)
ins('add', 49, 49, 50)
ins('fi')
ins('lt', 48, 49, 51)
ins('eq', 48, 48, 2)
ins('if', 48)
ins('add', 49, 49, 3)
ins('fi')
ins('add', 49, 49, 52)
ins('put', 49)
ins('mov', 48, 47)
ins('end')
ins('fi')
ins('add', 8, 8, 3)
ins('lt', 9, 8, 1)
ins('end')
emit_piece(2)
input_start()
ins('lt', 18, 13, 1)
ins('while', 18)
instruction_start()
ins('eq', 19, 10, 20)
ins('if', 19)
next_character()
ins('mov', 32, 17)
for character, register in ((97, 15), (98, 16)):
    ins('set', 11, character)
    ins('eq', 19, 10, 11)
    ins('if', 19)
    ins('mov', 32, register)
    ins('fi')
ins('decimal', 32)
ins('else')
ins('put', 10)
ins('fi')
instruction_end()
emit_piece(3)
ins('else')
note('Native Wasm: count fixed template lengths, emit code, quote input words.')
for register, value in ((4, 3 + 46 * 2), (12, 3 + 51 * 2), (20, 255),
                        (26, 128), (28, 112), (31, 8), (33, 256), (39, 3)):
    ins('set', register, value)
ins('set', 11, 3 + 70 * 2)
ins('data', 37, 11)
ins('set', 34, len(WASM_PARTS[2]) + 5 + len(WASM_PARTS[3]) + len(WASM_PARTS[4]))
input_start()
ins('lt', 18, 13, 1)
ins('while', 18)
ins('input', 14, 13)
ins('add', 11, 37, 14)
ins('data', 38, 11)
ins('add', 34, 34, 38)
ins('add', 13, 13, 22)
ins('lt', 18, 13, 1)
ins('end')
ins('set', 35, len(WASM_PARTS[1]) + 5)
ins('add', 35, 35, 34)
emit_piece(0)
emit_leb(35)
emit_piece(1)
emit_leb(34)
emit_piece(2)
emit_leb(1, True)
emit_piece(3)
input_start()
ins('lt', 18, 13, 1)
ins('while', 18)
instruction_start()
ins('eq', 19, 10, 20)
ins('if', 19)
next_character()
ins('set', 11, 105)
ins('eq', 19, 10, 11)
ins('if', 19)
emit_leb(16, True)
ins('else')
ins('set', 11, 110)
ins('eq', 19, 10, 11)
ins('if', 19)
emit_leb(1, True)
ins('else')
ins('mov', 32, 17)
for character, register in ((97, 15), (98, 16)):
    ins('set', 11, character)
    ins('eq', 19, 10, 11)
    ins('if', 19)
    ins('mov', 32, register)
    ins('fi')
ins('add', 32, 32, 39)
ins('put', 32)
ins('fi')
ins('fi')
ins('else')
ins('put', 10)
ins('fi')
instruction_end()
emit_piece(4)
literal([11])
ins('mul', 35, 1, 22)
ins('set', 11, 10)
ins('add', 34, 35, 11)
emit_leb(34)
literal([1, 0, 65, 0, 11])
emit_leb(35)
ins('set', 13, 0)
ins('lt', 18, 13, 1)
ins('while', 18)
ins('input', 44, 13)
ins('set', 46, 4)
ins('while', 46)
ins('mod', 45, 44, 33)
ins('put', 45)
ins('div', 44, 44, 33)
ins('sub', 46, 46, 3)
ins('end')
ins('add', 13, 13, 3)
ins('lt', 18, 13, 1)
ins('end')
ins('fi')

PROGRAM = [21331, len(data), len(code)] + data + code
assert len(PROGRAM) <= 60000


def packed_words(words):
    result = []
    for word in words:
        while True:
            digit = word % 32
            word //= 32
            if word:
                digit += 32
            result.append(chr(48 + digit + (digit >= 44)))
            if not word:
                break
    return ''.join(result)


def bootstrap_compile(p, target):
    instructions = [p[i:i + 4] for i in range(3 + p[1], len(p), 4)]
    if target == 2:
        body = WASM_PARTS[2] + padded(len(p), True) + WASM_PARTS[3]
        body += b''.join(expand_wasm(WASM_OP[op], (a, b, c), len(p)) for op, a, b, c in instructions)
        body += WASM_PARTS[4]
        code_payload = WASM_PARTS[1] + padded(len(body)) + body
        quoted = b''.join(word.to_bytes(4, 'little') for word in p)
        data_payload = b'\x01\x00\x41\x00\x0b' + padded(len(quoted)) + quoted
        return WASM_PARTS[0] + padded(len(code_payload)) + code_payload + b'\x0b' + padded(len(data_payload)) + data_payload
    parts, templates = (JS_PARTS, JS_OP) if target == 0 else (WG_PARTS, WG_OP)
    body = ''.join(templates[op].replace('~a', str(a)).replace('~b', str(b)).replace('~c', str(c))
                   for op, a, b, c in instructions)
    quoted = packed_words(p) if target == 0 else numbers(p)
    return parts[0] + str(len(p)) + parts[1] + quoted + parts[2] + body + parts[3]


def write_program(stem, p, assembly=None):
    for target, extension in enumerate(('js', 'wgsl', 'wasm')):
        content = bootstrap_compile(p, target)
        (ROOT / f'{stem}.{extension}').write_bytes(content.encode('ascii') if isinstance(content, str) else content)
    if assembly is not None:
        (ROOT / f'{stem}.sprout').write_text(assembly)


def assembly_text(p, lines):
    chunks = [' '.join(str(word) for word in p[3 + i:3 + min(i + 32, p[1])]) for i in range(0, p[1], 32)]
    return '.data\n' + '\n'.join(chunks) + '\n.code\n' + '\n'.join(lines) + '\n'


write_program('quines/05-compiler', PROGRAM)
(ROOT / 'compiler/compiler-program.json').write_text(json.dumps(PROGRAM) + '\n')
(ROOT / 'compiler/compiler.sprout').write_text(assembly_text(PROGRAM, listing))


def sample(rows, values=()):
    words = []
    lines = []
    for row in rows:
        op, *args = row
        args += [0] * (3 - len(args))
        words.extend([OPNUM[op], *args])
        lines.append(op + ' ' + ' '.join(map(str, args)))
    p = [21331, len(values), len(words), *values, *words]
    return p, assembly_text(p, lines)


examples = {
    'fibonacci': ('6765\n', [('set', 2, 20), ('set', 3, 0), ('set', 4, 1), ('set', 5, 1),
        ('while', 2), ('add', 6, 3, 4), ('mov', 3, 4), ('mov', 4, 6), ('sub', 2, 2, 5),
        ('end',), ('decimal', 3), ('set', 7, 10), ('put', 7)]),
    'factorial': ('3628800\n', [('set', 2, 10), ('set', 3, 1), ('set', 4, 1), ('while', 2),
        ('mul', 3, 3, 2), ('sub', 2, 2, 4), ('end',), ('decimal', 3), ('set', 5, 10), ('put', 5)]),
    'hello': ('Hello from Sprout!\n', [('set', 2, 3), ('set', 3, 3 + len('Hello from Sprout!\n')),
        ('set', 4, 1), ('lt', 5, 2, 3), ('while', 5), ('data', 6, 2), ('put', 6),
        ('add', 2, 2, 4), ('lt', 5, 2, 3), ('end',)]),
    'branch': ('Y\n', [('set', 2, 7), ('set', 3, 7), ('eq', 4, 2, 3), ('if', 4),
        ('set', 5, 89), ('else',), ('set', 5, 78), ('fi',), ('put', 5), ('set', 5, 10), ('put', 5)]),
    'overflow': ('0\n', [('set', 2, 4294967295), ('set', 3, 1), ('add', 4, 2, 3),
        ('decimal', 4), ('set', 5, 10), ('put', 5)]),
}
example_records = []
for name, (expected, rows) in examples.items():
    p, assembly = sample(rows, expected.encode('ascii') if name == 'hello' else ())
    write_program('compiler/' + name, p, assembly)
    (ROOT / f'compiler/{name}.json').write_text(json.dumps(p) + '\n')
    example_records.append({'name': name, 'expected': expected})
(ROOT / 'compiler/examples.json').write_text(json.dumps(example_records, indent=2) + '\n')

measurements = {
    'schema': 'sprout-size-v1', 'opcodes': len(OPS), 'registers': 64,
    'instructions': len(code) // 4, 'dataWords': len(data), 'programWords': len(PROGRAM),
    'measurement': 'Byte counts and deterministic gzip, not Kolmogorov complexity.',
    'encoding': 'Five-byte LEB128 for dynamic sizes and i32 constants; direct native instructions.',
    'jsQuotation': 'Printable five-bit word encoding; full compiler and launcher remain in the seed.',
    'files': {},
}
for relative in ('compiler/compiler.sprout', 'compiler/compiler-program.json',
                 'quines/05-compiler.js', 'quines/05-compiler.wgsl', 'quines/05-compiler.wasm'):
    content = (ROOT / relative).read_bytes()
    measurements['files'][relative] = {'bytes': len(content), 'gzipBytes': len(gzip.compress(content, mtime=0)),
                                     'sha256': hashlib.sha256(content).hexdigest()}
(ROOT / 'compiler/complexity.json').write_text(json.dumps(measurements, indent=2) + '\n')
print(f'Sprout compiler: {len(PROGRAM)} words; {len(code) // 4} instructions; {len(data)} data words; 3 backends')
