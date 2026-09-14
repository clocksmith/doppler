# Self-reproducing JavaScript and WGSL programs

A standalone educational gallery of programs that reproduce their own source,
generate source across languages, render an image of their source, and bootstrap
a small compiler. A **quine** is a program that outputs its own source without
reading its source file.

This is Doppler's standalone JavaScript/WGSL quine gallery, located at
`examples/js-wgsl-quines/`. It demonstrates self-reproduction, cross-language
source generation, and exact CPU/GPU output comparison without depending on
Doppler's model inference or artifact runtime.

The current GPU runner uses standard browser WebGPU or the optional Node
`webgpu` binding. The retained reports in `verification/` came with the original
bundle; relocating this gallery does not establish new GPU execution evidence.

Imported from `~/Desktop/js-wgsl-five-quines`. The generated programs, manifest,
and existing verification records are preserved; the Desktop original remains
in place.

## Start with the small one

[`quines/00-minimal.js`](quines/00-minimal.js) is a JavaScript-only introduction
to the quotation mechanism behind the larger constructions:

```js
const s = "const s = $;\nconsole.log(s.replace('$', JSON.stringify(s)));";
console.log(s.replace('$', JSON.stringify(s)));
```

`s` holds a template with a hole where its own quoted value belongs.
`JSON.stringify(s)` supplies that quoted value, including escaped newlines and
quotes. `replace` fills only the first `$`, leaving the one inside the printed
code intact. `console.log` supplies the final newline, matching the source file.

```sh
node quines/00-minimal.js
npm run test:minimal
```

The check executes returned stdout as the next program in a fresh Node process,
without passing it a source-file path, and compares the complete bytes.
[`verification/minimal-results.json`](verification/minimal-results.json) records
the source hash, environment, and checked generations. This addition does not
change the original gallery's generated JS/WGSL programs or their manifest.

## A small, GPU-built JavaScript quine

[`quines/gpu-quine.js`](quines/gpu-quine.js) preserves the supplied standalone
program, including its final newline. Its first line quotes the executable body;
WGSL constructs both that declaration and the complete executable JavaScript.
JavaScript builds the shader and decodes its mapped output, but does not assemble
the finished descendant from the stored body. There is no source-file reading,
`eval()`, or `toString()` in the quine. Each execution prints one generation and
stops. This is a GPU-assisted JavaScript quine, not a standalone WGSL quine.

Serve this gallery with `npm run serve`, then open
[`quines/index.html`](quines/index.html) and inspect the developer console.
The launcher does not require the larger gallery's `Quine` interface.

```sh
npm run test:gpu-quine
```

The browser test first runs the uninstrumented launch page, then runs eight
generations in fresh workers. An observer delegates every WebGPU call to the
browser and captures actual compilation diagnostics, dispatches, mapped ASCII
bytes, and cleanup. Each mapped output becomes the next executable program.
Complete bytes must match the starting file, including the final newline;
the separately captured console string plus one LF must match that readback.
Only the external harness reads files for loading and comparison.

Set `CHROME_BIN` to choose Chrome. `GPU_QUINE_BACKEND=swiftshader` explicitly
selects software WebGPU; it is never a silent retry or hardware-GPU claim.
Every attempt writes its own adapter details, source/harness hashes, returned
programs, and pass/failure receipt under `verification/gpu-quine/`. Software or
unknown adapter identity is kept distinct from hardware-reported execution.
These checks do not establish performance or cryptographic GPU provenance.
The original generated constructions and their manifest remain unchanged.

## Construction notes

Five complete constructions, not pseudocode. Generated programs are under
`quines/`; the readable construction logic is in `build.py` and
`compiler/build_compiler.py`.

**Verification boundary:** the supplied JavaScript, independent character-emission
reference, and independent Sprout interpreter tests pass. All five also pass their
JavaScript checks in Chromium. Actual WGSL compilation and WebGPU dispatch have
**not** been verified in the construction environment: browser navigation was
blocked by administrator policy, and a Node WebGPU binding was unavailable.
`tests/gpu.mjs` and the browser demo perform real GPU tests and never substitute
CPU output for a GPU result. A reported software adapter is not physical GPU proof.

## Start here

The gallery is self-contained, with all five initial programs embedded. It does
not download libraries, models, fonts, or other runtime resources.

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://localhost:8000/demo.html` in a WebGPU-capable browser. Choose a program,
run its JavaScript checks, then run its GPU rounds. The gallery displays the
adapter, exact source hashes, and failure diagnostics. Successful descendants are
used as the next programs; a saved expected source is never substituted for output.
The visual example offers a lossless PNG after actual GPU execution.

The page is a controlled demo of bundled code, not a sandbox for arbitrary
programs. Stopping a worker and destroying a WebGPU device are best-effort timeout
mechanisms, not a security isolation guarantee.

## The five constructions

### 1. `01-application.js`: the complete application quine

```text
whole JavaScript application -> WGSL -> identical whole JavaScript application
```

The output includes the authored WebGPU launcher, shader builder, readback,
error handling, and reproduction mechanism. The browser, JavaScript engine, GPU
driver, and optional Node WebGPU binding are the environment, not reproduced code.
The separate gallery HTML is not part of this program's identity.

An embedded integer array `Q` describes the program **template**, not the finished
program. The WGSL emitted by that program contains `Q` in its own constants. Its
character writer expands the template and inserts the numeric representation of
`Q` at the quotation hole. This constructs the complete child JavaScript.
The host binds only an initially empty output buffer, not an input source buffer.

Default execution performs the GPU round and writes the resulting JavaScript.
`--cpu` exposes the independent JavaScript reproduction path; `--wgsl` writes the
shader without attempting GPU execution.

### 2. `02-polyglot.js` and `.wgsl`: one file, two languages

```text
S as JavaScript -> S
S as WGSL       -> S
```

These two files are byte-for-byte identical. Neither interpreter receives an
extracted region. The whole shared file is valid JavaScript according to Node
and Chromium. The included nested-comment parser confirms the WGSL-active region;
an actual WebGPU compiler must still validate it.

The syntactic skeleton is:

```text
/* /* */
JavaScript-only code
// */all WGSL-only code on one physical line
```

WGSL allows nested block comments; JavaScript block comments do not nest. The
space in `// */` matters: `//*/` contains another `/*` and would reopen a nested
comment. The builder also refuses accidental block-comment delimiters in the
JavaScript-only region. Both languages expand the same self-description.

The JS default prints `S`. The WGSL `main` entry writes `S` as a length followed
by character codes in storage binding 0. `--gpu` dispatches the same source.

### 3. `03-multiquine`: all four transitions

```text
JavaScript A -> A or B
WGSL B       -> A or B
```

One embedded data array contains templates for both sources, separated by `CUT`.
Either language can expand either region and reconstruct the complete pair.

JavaScript: `Quine.emit("js")` or `Quine.emit("wgsl")`.
WGSL: `main` emits WGSL; `javascript` emits JavaScript.

Both default to self-reproduction. The GPU test first executes B to obtain B,
then executes that **returned** B to obtain A, then runs the returned A.
The mode/entry-point selector never supplies a desired answer or source bytes.

### 4. `04-visual`: source is both program and portrait

The first GPU dispatch reconstructs the whole JS program. A second dispatch reads
that same GPU output buffer and renders its bytes with a hand-drawn bitmap alphabet.
The CPU does not rasterize the characters or upload its expected source as pixels.

Characters wrap every 128 cells; newlines have a visible return glyph. The output
is a source tapestry, not a layout-preserving text editor. A few lowercase glyphs
use small-cap shapes, while lossless pixel metadata preserves their actual bytes.

Every cell is 8 by 10 pixels. Its bottom-right pixel stores a byte and marker.
Cell zero stores the source length. `Quine.decodeImage()` recovers the original
ASCII source, including whitespace, from those pixels. The GPU runner checks the
recovered source against the independently returned text buffer.

The host computes buffer dimensions from the expected source length for allocation
only. It never supplies those source bytes to either shader.

`verification/visual-reference.png`, when present, is an independently rendered
**CPU reference**, not evidence of GPU execution. Lossless PNG preserves the
metadata; resizing, JPEG compression, or editing pixels may not.

### 5. `05-compiler`: an actual self-hosting compiler

This is a compiler for **Sprout**, a deliberately small structured register
language, not a compiler for arbitrary JavaScript or WGSL.

The compiler is itself a 19-opcode Sprout program. The generated JS and
WGSL versions contain that source as numeric data and its **compiled executable
statements**. They emit either backend for themselves or another Sprout program.
Generated programs do not interpret Sprout opcodes at runtime. Only the compiler
program parses opcodes, because parsing and translating programs is its job.

```text
compiler source P
  -> compiled JS compiler C_JS
  -> compiled WGSL compiler C_WGSL
  -> identical C_JS
```

JS defaults to compiling P to its own JS. WGSL defaults to compiling P to its own
WGSL. Set the WGSL `TARGET` override to 0 for JS, or 1 for WGSL.

Both backends also compile unrelated Fibonacci, factorial, text-output, branch,
and unsigned-overflow programs. The independent interpreter compiles the compiler
and all five examples to byte-identical generated sources.

The GPU compiler supports a separate `EXTERNAL` override. When zero, it reads
only embedded P; no source input is used. Its bind-group interface still includes
binding 1, to which the self-reproduction harness binds a single zero word.
When one, binding 1 contains `[word_count, ...Sprout_program]` for compiling other
programs. That is a separate ordinary compiler operation, not the quine's input.

The diagnostic wrapper is shared by all compiled programs. The independent
examples are normal programs, not themselves quines; only compiling the compiler
source gives the self-reproducing fixed point.

## Run without WebGPU

No npm dependencies are needed for these checks:

```bash
npm test
node quines/02-polyglot.js > /tmp/polyglot-child.js
cmp quines/02-polyglot.js /tmp/polyglot-child.js

node quines/05-compiler.js > /tmp/compiler-child.js
cmp quines/05-compiler.js /tmp/compiler-child.js
node quines/05-compiler.js --wgsl > /tmp/compiler-child.wgsl
cmp quines/05-compiler.wgsl /tmp/compiler-child.wgsl

node compiler/compile.mjs compiler/fibonacci.sprout --target js > /tmp/fibonacci.js
node /tmp/fibonacci.js
# 6765

node compiler/compile.mjs compiler/factorial.sprout --target wgsl > /tmp/factorial.wgsl
```

For every construction, `--cpu` prints its JS self-reproduction and `--wgsl`
prints the complete shader. For the first and fourth, default execution instead
requires WebGPU because their principal construction runs on the GPU.

## Actual WebGPU verification

The browser demo requires no installation beyond a supported browser. Node needs
a provider; it does not normally expose WebGPU by itself:

```bash
npm install --no-save webgpu
npm run test:gpu
node quines/01-application.js > /tmp/gpu-child.js
cmp quines/01-application.js /tmp/gpu-child.js
```

The tests use normal provider selection. Whether a particular machine has a usable
hardware or software provider is environment-dependent. Reports include adapter
information and retain failures. No performance or GPU provenance claim follows
from a correct string: a CPU can compute these strings too.

GPU tests cover every reproduction path, actual returned descendants, both
compiler targets, GPU compilation and execution of all five independent Sprout
examples, and exact equality of the visual GPU image with the CPU image reference.
A failure is never relabeled as a CPU-reference pass.

All text-producing shader entries write one u32 length at `output[0]`, then one
ASCII character per u32. They need one invocation, never multiple workgroups.
The visual `paint` entry is different: it uses an 8-by-8 workgroup and binds the
already generated source plus a pixel output buffer.

## Rebuild and inspect

```bash
python3 build.py
npm test
```

Rebuilding requires Python 3 and Node. Runtime reproduction needs neither the
builder nor any original source file. `manifest.json` records all ten source-file
sizes and SHA-256 hashes. It is external verification metadata, not an ingredient
used by the quines.

`build.py` builds the first four fixed points and embeds the complete launcher.
`compiler/build_compiler.py` contains the bootstrap backend templates and builds
the Sprout compiler. `compiler/compiler.sprout` is the actual self-hosted source,
with its generic emission templates as data. `compiler/assemble.mjs` assembles that
text form. `tests/reference.mjs` independently interprets the numeric language.

The Sprout language has unsigned 32-bit registers, explicit data and input reads,
arithmetic, comparisons, loops, branches, and byte/decimal output. Division and
remainder by zero produce zero in both targets. Data reads outside the embedded
program or supplied input return zero. See `compiler/LANGUAGE.md`.

## References

The constructions were written for this bundle. These references establish the
language/platform details, not validation of these particular programs:

- WGSL comments and shader semantics: https://www.w3.org/TR/WGSL/
- JavaScript comment grammar: https://tc39.es/ecma262/multipage/ecmascript-language-lexical-grammar.html#sec-comments
- Dawn's Node WebGPU binding: https://dawn.googlesource.com/dawn/+/refs/heads/main/src/dawn/node/README.md

No code reads its own file, extracts source from script elements, or uses
`Function.prototype.toString()`. The host reads starting programs only to execute
and compare them, exactly as an interpreter or test runner ordinarily would.


## Sprout's third backend: native Wasm

The compiler now emits JavaScript, WGSL, or an import-free Wasm binary. It still
uses 19 opcodes and 64 u32 registers. Generated Wasm contains direct native
instructions, not a Sprout interpreter. Its embedded numeric source supplies
quotation, just as it does for the two text backends. No descendant reads its
source file, calls the Python builder, or embeds a precompiled compiler module.

The one output-domain change is that `put` accepts bytes 0..255 rather than only
ASCII. Text targets remain strings; the Wasm target returns a `Uint8Array`.
`Quine.executeBytes()` exposes raw program output. `Quine.fromWasm(bytes)`
instantiates the supplied module and only transports input/output through its
memory; compilation runs inside that module.

`compiler/complexity.json` records instruction count, source sizes, deterministic
gzip sizes, and hashes. These are description-size measurements, not a proof of
minimal Kolmogorov complexity. Dynamic binary sizes and i32 constants use legal
five-byte LEB128: fewer compiler rules, somewhat larger emitted binaries.

`node compiler/compile.mjs compiler/fibonacci.sprout --target wasm` emits a real
Wasm module on stdout. `node quines/05-compiler.js --wasm` emits the compiler's
Wasm form. The default JS and WGSL commands remain unchanged.

Run `npm run test:wasm` for native self-reproduction, independent interpreter
agreement, all example programs, byte boundaries, and malformed-input rejection.
Run `npm run test:browser` for the physical-GPU proof in Chrome. Set `CHROME_BIN`
if Chrome is not at the platform's default path. No Node WebGPU shim is required.

The browser proof consumes actual descendants:

`JS -> GPU WGSL -> Wasm -> byte-identical JS`

It also checks WGSL and Wasm self-reproduction, repeats the cycle three times,
and executes all five unrelated programs through every compiler/runtime pairing.
A software fallback does not qualify as physical-GPU evidence. Fresh receipts
and returned descendants go in
`verification/sprout-three-backend-2026-09-12/`; the original CPU/blocked-GPU
receipts describe the earlier two-backend construction, not this new run.
