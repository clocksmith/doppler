# Sprout

Sprout is a structured register language, not a host-language interpreter. Its
compiler is itself a Sprout program. All three backends emit direct executable
statements or native instructions.

## Program and machine

A program is an array of u32 words:

`[21331, data_word_count, code_word_count, ...data, ...code]`

Each instruction occupies four words: `[opcode, a, b, c]`. Unused operands are
zero in assembled source. There are exactly 19 opcodes and 64 u32 registers.
Registers begin at zero except `r0`, the requested target (0 JavaScript,
1 WGSL, 2 Wasm), and `r1`, the input program's word count. Arithmetic wraps
modulo 2^32. Comparisons and division are unsigned.

| Opcode | Source | Meaning |
| --- | --- | --- |
| 0 | `nop` | No operation. |
| 1 | `set a b` | Set r[a] to immediate b. |
| 2 | `mov a b` | Copy r[b] to r[a]. |
| 3 | `add a b c` | Add r[b] and r[c]. |
| 4 | `sub a b c` | Subtract r[c] from r[b]. |
| 5 | `mul a b c` | Multiply r[b] and r[c]. |
| 6 | `div a b c` | Unsigned quotient; zero divisor produces zero. |
| 7 | `mod a b c` | Unsigned remainder; zero divisor produces zero. |
| 8 | `eq a b c` | One if equal, otherwise zero. |
| 9 | `lt a b c` | One if r[b] is unsigned-less-than r[c], otherwise zero. |
| 10 | `data a b` | Read embedded program word P[r[b]]; out of bounds is zero. |
| 11 | `input a b` | Read supplied input word at r[b]; out of bounds is zero. |
| 12 | `put a` | Emit byte r[a], which must be in 0..255. |
| 13 | `decimal a` | Emit r[a] as unsigned decimal ASCII. |
| 14 | `while a` | Repeat while r[a] is nonzero. |
| 15 | `end` | End a while block. |
| 16 | `if a` | Enter a branch when r[a] is nonzero. |
| 17 | `else` | Optional alternative branch. |
| 18 | `fi` | End an if block. |

Assembly uses `.data` and `.code` sections and semicolon comments. Registers
are numeric indices. There are no functions, recursion, arbitrary strings,
indirect jumps, or floating-point operations. Existing ASCII programs keep the
same meaning; widening `put` to bytes permits emitting actual Wasm binaries.
Text helpers decode UTF-8; use raw-byte APIs for arbitrary binary output.

## Compilation and self-reproduction

With no external input, each compiled program reads its own embedded P through
both `data` and `input`. With external input, `data` still reads the embedded
program while `input` reads the supplied program. That distinction allows the
same compiler to reproduce itself and compile unrelated programs.

The JS API is `Quine.compile(program, 'js' | 'wgsl' | 'wasm')`. Text targets
return strings; Wasm returns a Uint8Array. `executeBytes(input, target)` returns
raw bytes. `Quine.fromWasm(bytes)` returns an instantiated native compiler with
`compile(program, target)`, `js()`, `wgsl()`, and `self()` (Wasm bytes).
Its adapter performs memory I/O, not compilation.

WGSL uses one compute invocation. Binding 0 is output: a length word followed by
one u32 per byte. Binding 1 is external input: a word count followed by words.
`EXTERNAL=0` selects embedded P and needs only a dummy input word;
`EXTERNAL=1` selects supplied input. `TARGET` is 0, 1, or 2.

Wasm has no imports. It exports `memory`, `run(target, pointer, wordCount)`,
and immutable `input`/`output` offset globals. A zero pointer selects embedded
P. External input begins at byte 524288 and uses little-endian u32 words. Output
begins at byte 262144; `run` returns its byte count. The module owns 16 memory
pages. Dynamic section lengths and i32 constants use legal padded five-byte
LEB128 to reduce the compiler's description rather than the output's byte count.

## Limits

Programs have at most 60,000 words, blocks nest at most 64 levels, and output is
limited to 262,143 bytes. Headers, operand ranges, and balanced block structure
are checked before execution. Invalid native inputs trap; JS throws; WGSL marks
output invalid. An otherwise valid program may exceed the output limit.
Nonterminating programs remain possible: a worker or GPU timeout is not a
security sandbox. Do not run untrusted programs as if it were one.

The example programs cover Fibonacci (6765), factorial (3628800), text,
branching, and unsigned overflow. The compiler's measured size is recorded in
`complexity.json`; no claim of provably minimal Kolmogorov complexity is made.
