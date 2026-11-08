import json, html, math, sys
S = sys.argv[1]
data = {r['id']: r for r in json.load(open(S + '/data.json'))}
st = json.load(open(S + '/stages.json'))
cat = {m['modelId']: m for m in json.load(open(sys.argv[2] + '/models/catalog.json'))['models']}
E = html.escape
BPW = {'F32': 32, 'F16': 16, 'BF16': 16, 'Q4_K_M': 4.5, 'Q4_0': 4.5, 'Q4_1': 5, 'Q5_0': 5.5, 'Q8_0': 8.5, 'Q6_K': 6.5625, 'W4A16': 4.5}
QUICKSTART = {'gemma-3-270m-it-q4k-ehf16-af32', 'gemma-3-1b-it-q4k-ehf16-af32', 'google-embeddinggemma-300m-q4k-ehf16-af32',
              'gemma-4-e2b-it-q4k-ehf16-af32', 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple', 'qwen-3-5-0-8b-q4k-ehaf16', 'qwen-3-5-2b-q4k-ehaf16'}

def n(x):
    return f'{x:,}' if isinstance(x, int) else ('—' if x is None else str(x))

def size(b):
    if b is None: return '—'
    if b >= 2**30: return f'{b/2**30:.2f} GiB'
    if b >= 2**20: return f'{b/2**20:.1f} MiB'
    return f'{b/2**10:.0f} KiB'

def fmtname(d):
    return {'Q4_K_M': 'Q4_K'}.get(d, d)

def kv_bytes_per_value(r):
    return 4 if r.get('kvDtype') == 'f32' else 2

# ---- KV recompute with the exact rules (sliding ring only for mixed geometry outside DiffusionGemma)
def kv_info(r, s):
    if s.get('sequence'):
        return None
    a = r.get('arch')
    if not isinstance(a, dict) or not a.get('numKeyValueHeads'):
        return None
    lt = r.get('layerTypes') or {}
    L = a['numLayers']; kvh = a['numKeyValueHeads']; hd = a['headDim']
    ghd = a.get('globalHeadDim') or hd; gkvh = a.get('numGlobalKeyValueHeads') or kvh
    shared = a.get('numKvSharedLayers') or 0
    cap = r.get('kvCapacityTokens')
    sw = s.get('slidingWindow')
    mixed = bool(lt) and ((ghd != hd) or shared > 0)
    slide = lt.get('sliding_attention', 0); full = lt.get('full_attention', 0); lin = lt.get('linear_attention', 0); conv = lt.get('conv', 0)
    if not lt:
        full = L
    out = {}
    bpv = kv_bytes_per_value(r)
    if mixed:
        ring = not s.get('diffusionGemma')
        sl_per = slide * 2 * kvh * hd; gl_per = full * 2 * gkvh * ghd
        sl_cap = min(sw, cap) if (ring and sw) else cap
        out['layout'] = 'mixed geometry, ' + ('sliding layers in a ring of %s tokens' % n(sl_cap) if ring and sw else 'all layers full length')
        out['perToken'] = sl_per + gl_per
        out['bytes'] = (sl_per * sl_cap + gl_per * cap) * bpv
        out['detail'] = f'{slide} sliding × 2 × {kvh}×{hd} + {full} global × 2 × {gkvh}×{ghd}'
    else:
        layers = L - lin - conv
        per = layers * 2 * kvh * hd
        out['layout'] = 'contiguous, every attention layer full length'
        out['perToken'] = per
        out['bytes'] = per * cap * bpv
        out['detail'] = f'{layers} layers × 2 × {kvh}×{hd}' + (f' ({lin} linear layers excluded)' if lin else '') + (f' ({conv} conv layers excluded since 8815443c)' if conv else '')
    out['cap'] = cap; out['capSource'] = r.get('kvCapSource', ''); out['dtype'] = r.get('kvDtype')
    return out

# ---- model groups (one architecture each; lanes are stored variants)
G = [
 dict(key='gemma3-270m', name='Gemma 3 270M IT', lanes=['gemma-3-270m-it-q4k-ehf16-af32', 'gemma-3-270m-it-f16-af32'], pipes=['gen'],
      notes=['Smallest text model; both lanes verified. The Q4_K lane graph has no fused Q4_K kernel, so the loader expands Q4_K to F16 at load.']),
 dict(key='gemma3-1b', name='Gemma 3 1B IT', lanes=['gemma-3-1b-it-q4k-ehf16-af32', 'gemma-3-1b-it-f16-af32', 'create-gemma-3-1b-qualification'], pipes=['gen'],
      notes=['create-gemma-3-1b-qualification is a local copy of the Q4_K lane used by Create qualification, not a separate model.', 'The F16 lane exists on amd-host only and is not in the catalog.']),
 dict(key='translategemma-1b', name='TranslateGemma 4B→1B en-es', lanes=['translategemma-4b-1b-enes-q4k-ehf16-af32'], pipes=['gen', 'tr'],
      notes=['Same 1B geometry as Gemma 3 1B. Its manifest uses the plain gemma chat template, so translation is driven by the prompt, not the structured translategemma template.']),
 dict(key='translategemma-4b', name='TranslateGemma 4B IT', lanes=['translategemma-4b-it-q4k-ehf16-af32'], pipes=['gen', 'tr'],
      notes=['Uses the translategemma template: string prompts are rejected; each message must carry source_lang_code and target_lang_code.']),
 dict(key='embeddinggemma', name='EmbeddingGemma 300M', lanes=['google-embeddinggemma-300m-q4k-ehf16-af32'], pipes=['emb'],
      notes=['Output: mean pool → dense 768→3072 → dense 3072→768 → L2 normalize, giving a 768-value vector.']),
 dict(key='gemma4-e2b', name='Gemma 4 E2B IT', lanes=['gemma-4-e2b-it-q4k-ehf16-af32', 'gemma-4-e2b-it-q4k-ehf16-af32-int4ple', 'gemma-4-e2b-it-q4k-ehf16-af16-int4ple'], pipes=['gen', 'img', 'aud', 'vid'],
      notes=['Per-layer inputs (PLE): every token also gathers 35 × 256 = 8,960 values from a 262,144-row table, one 256-value slice per layer.',
             'The last 20 layers reuse earlier layers’ keys and values (shared KV) and the last 20 layers use a double-width MLP (12,288).',
             'int4ple lanes store the PLE table as 4-bit rows with one F32 scale each: 6.01 GiB of F16-labelled data becomes 2.73 GiB (7.52 GB → 4.00 GB on disk).',
             'The manifest enables vision and audio. The audio upload and layout code was repaired in a76f1add; full audio inference is not qualified; the video width bug is fixed in 3a58230f but no GPU video run has confirmed it.']),
 dict(key='gemma4-12b', name='Gemma 4 12B IT (text)', lanes=['gemma-4-12b-it-text-q4k-ehf16-af32', 'gemma-4-12b-it-text-q4k-ehf16-af16', 'gemma-4-12b-it-text-w4a16-ct-ehf16-af16'], pipes=['gen'],
      notes=['Global layers use 512-wide heads, sliding layers 256-wide, so the KV cache uses mixed geometry.',
             'The W4A16 lane imports compressed-tensors 4-bit weights (group 32, one scale per row per group); its BF16 tensors are the scales.']),
 dict(key='gemma4-31b', name='Gemma 4 31B IT (text)', lanes=['gemma-4-31b-it-text-q4k-ehf16-af32', 'gemma-4-31b-it-text-q4k-ehf16-af16'], pipes=['gen'],
      notes=['16 KV heads on sliding layers and 4 on global layers; the largest KV cache per token among the Gemma models.']),
 dict(key='diffusiongemma', name='DiffusionGemma 26B-A4B', lanes=['diffusiongemma-26b-a4b-it-q4k-ehf16-af16'], pipes=['dg'],
      notes=['Writes 256-token canvases by denoising instead of one token at a time.',
             'Each layer runs a dense FFN and a mixture-of-experts branch (128 experts, 8 per token) and sums them.',
             'Logits per denoising step are 256 × 262,144 F32 values = 256 MiB.']),
 dict(key='qwen3-emb', name='Qwen3 Embedding 0.6B', lanes=['qwen-3-embedding-0-6b-q4k-ehf16-af32'], pipes=['emb'],
      notes=['Output: last-token pool → L2 normalize, giving a 1,024-value vector.']),
 dict(key='qwen3-rr', name='Qwen3 Reranker 0.6B', lanes=['qwen-3-reranker-0-6b-f16-af32', 'qwen-3-reranker-0-6b-q4k-ehf16-af32'], pipes=['rr'],
      notes=['Scores a query–document pair from two logits only: "yes" minus "no"; probability = sigmoid of that difference.']),
 dict(key='qwen35-08b', name='Qwen 3.5 0.8B', lanes=['qwen-3-5-0-8b-q4k-ehaf16'], pipes=['gen', 'img'],
      notes=['Hybrid: 18 linear-attention layers with fixed state, 6 full-attention layers with KV cache.',
             'Its manifest carries a Qwen3-VL vision encoder, reachable only through embedImage (not transcription), with open gaps.',
             'The name says -ehaf16, but the head is Q4_K and activations are F32.']),
 dict(key='qwen35-2b', name='Qwen 3.5 2B', lanes=['qwen-3-5-2b-q4k-ehaf16'], pipes=['gen'],
      notes=['Same layer mix and state size as 0.8B with a 2,048-wide hidden state.', 'The name says -ehaf16, but the head is Q4_K and activations are F32.']),
 dict(key='qwen36-27b', name='Qwen 3.6 27B', lanes=['qwen-3-6-27b-q4k-ehaf16', 'qwen-3-6-27b-q4k-eaf16'], pipes=['gen'],
      notes=['The eaf16 lane is a manifest that borrows the ehaf16 lane’s shards (weightsRef); only activations and math change to F16.',
             'The ehaf16 name is wrong in the same way as 0.8B: Q4_K head, F32 activations.']),
 dict(key='qwen36-35b', name='Qwen 3.6 35B-A3B (MoE)', lanes=['qwen-3-6-35b-a3b-text-source-q4_0-af32'], pipes=['gen'],
      notes=['Counts come from the GGUF header on amd-host: 40 layers (plus 1 multi-token-prediction layer), 10 full-attention, 30 linear.',
             '256 experts with 8 used per token, plus one shared expert. Experts hold 33.0 B of the 35.5 B weights, so a token touches about 1.0 B expert weights plus 2.5 B shared weights.',
             'The "Q4_0" file is a mix: Q4_0 for most weights, Q4_1 and Q8_0 in expert down-projections, Q8_0 shared experts and K/V, Q6_K output head.',
             'Runs through the opt-in profile qwen3-6-35b-a3b-q4_0-candidate. A Q4_K conversion config exists, but no converted artifact was found.']),
 dict(key='qwen38-27b', name='Qwen 3.8 27B', lanes=['qwen3-8-27b-text-f16-af32', 'qwen3-8-27b-text-q4k-ehaf16'], pipes=['gen'],
      notes=['Same geometry as Qwen 3.6 27B. The F16 lane passes source-token parity in the catalog; the Q4_K artifact on mac-host is not in the catalog.',
             'A Q4_0 passthrough config exists (qwen3.8-27b-q4_0-passthrough.json); no artifact found.']),
 dict(key='flash-next', name='Qwen 3.8 Flash-Next', lanes=['qwen-3-8-flash-next-text-q4k-ehf16-af32'], pipes=['gen'],
      notes=['177 billion weights: 512 experts with 10 used per token, plus a shared expert of width 640.',
             'Hyper-connection layers with indexed (QSA) attention over selected KV rows; requires F32, GPU, and no LoRA, embedding, prefix-KV or batch decode.',
             'The 51.2 B BF16 weights are n-gram embedding tables (128 tensors).',
             'A capability rule widens its KV cache to F32 even though the manifest says F16.', 'On amd-host only; not in the catalog.']),
 dict(key='lfm2', name='LFM2.5 1.2B Instruct', lanes=['lfm2-5-1-2b-instruct-q4k-ehf16-af32'], pipes=['gen'],
      notes=['10 short-convolution layers and 6 attention layers. Batched GPU decode refuses conv models, so it decodes one token per step.', 'Before 8815443c the conv layers also got KV buffers (128 MiB instead of 48 MiB at 4,096 tokens).',
             'Catalog: experimental, not tested.']),
 dict(key='minicpm', name='MiniCPM4 0.5B', lanes=['minicpm4-0-5b-f16-af32'], pipes=['gen'],
      notes=['LongRoPE position scaling and an explicit ×12 embedding scale. Catalog: experimental, tests failed.']),
 dict(key='muse', name='Muse Glimmer 30B (text)', lanes=['muse-glimmer-30b-text-f16-af32', 'muse-glimmer-30b-text-bf16-af32'], pipes=['gen'],
      notes=['Dense 27.9 B weights stored at 16 bits (55.7 GB). Its manifest caps the KV cache at 131,072 tokens, which allocates 6.5 GiB up front.',
             'Catalog: experimental, tests failed on both lanes.']),
 dict(key='glmocr', name='GLM-OCR', lanes=['glm-ocr-f16-af32', 'glm-ocr-f16-af32-pad-token-v1'], pipes=['img', 'gen'],
      notes=['Image-to-text OCR: GLM vision encoder (14-pixel patches, 2×2 merge) feeding a 16-layer text decoder with 3-axis rotary positions.',
             'Two local artifacts; not in the catalog.']),
 dict(key='amplify', name='AMPLIFY 120M (protein)', lanes=['amplify-120m-f16-af32'], pipes=['seq', 'emb'],
      notes=['Protein language model: one bidirectional pass returns per-residue embeddings, a mean-pooled embedding, and per-position logits (masked-token prediction).']),
 dict(key='esm2', name='ESM-2 t12 35M (protein)', lanes=['esm2-t12-35m-ur50d-f32-af32'], pipes=['seq', 'emb'],
      notes=['Per-residue and mean-pooled embeddings; no logits. Embedding scale 0.88; LayerNorm.']),
 dict(key='esmc', name='ESM C 300M (protein)', lanes=['esmc-300m-f32-af32'], pipes=['seq', 'emb'],
      notes=['Per-residue and mean-pooled embeddings; no logits.']),
 dict(key='nt', name='Nucleotide Transformer v2 50M (DNA)', lanes=['nucleotide-transformer-v2-50m-f32-af32'], pipes=['seq', 'emb'],
      notes=['DNA sequence encoder: per-token and mean-pooled embeddings; no logits.']),
 dict(key='sana', name='Sana Sprint 0.6B (image)', lanes=['sana-sprint-0.6b-wf16-ef16-hf16-f16'], pipes=['imgdiff'],
      notes=['Text-to-image model (Gemma 2 2B text encoder, 28-layer linear-attention transformer, DC-AE decoder). The artifact exists on amd-host, but the runtime rejects layout "sana" at load.']),
]

PIPES = [
 ('gen', 'Text generation'), ('tr', 'Translation'), ('emb', 'Embedding'), ('seq', 'Sequence encoding'), ('rr', 'Rerank'),
 ('img', 'Image → text'), ('aud', 'Audio → text'), ('vid', 'Video → text'), ('dg', 'Diffusion text'), ('imgdiff', 'Image generation'),
]
PIPE_STATE = {  # (group, pipe) -> label overrides
 ('gemma4-e2b', 'aud'): 'fixed, not run', ('gemma4-e2b', 'vid'): 'fixed, not run', ('qwen35-08b', 'img'): 'partial', ('sana', 'imgdiff'): 'rejected',
}

def lane_status(lid):
    c = cat.get(lid)
    if lid == 'qwen-3-6-35b-a3b-text-source-q4_0-af32':
        return ('candidate', 'opt-in profile; qualification open')
    if not c:
        return ('local', 'not in catalog')
    lc = (c.get('lifecycle') or {}).get('status') or {}
    rt, te = lc.get('runtime'), lc.get('tested')
    if te == 'verified': return ('ok', f'{rt}, verified')
    if te == 'failed': return ('bad', f'{rt}, tests failed')
    return ('warn', f'{rt}, tests {te}')

def formats_line(r):
    parts = []
    for d, v in sorted(r['dtypes'].items(), key=lambda kv: -kv[1]['elements']):
        if v['elements'] == 0: continue
        b = v.get('bytes') or int(v['elements'] * BPW.get(d, 16) / 8)
        parts.append(f'<span class="fmt"><b>{E(fmtname(d))}</b> {n(v["elements"])} <i>{size(b)}</i></span>')
    return ' '.join(parts)

def where(r):
    w = list(r.get('where') or [])
    return ', '.join(w) if w else '—'

def q4k_handling(lid):
    s = st.get(lid) or {}
    files = ' '.join(s.get('kernelFiles') or [])
    r = data[lid]
    has_q4k = 'Q4_K_M' in r['dtypes']
    if lid.endswith('q4_0-af32'):
        return 'GGUF blocks kept packed; native Q4_0/Q4_1/Q5_0/Q8_0/Q6_K kernels decode them on the fly.'
    if 'W4A16' in r['dtypes']:
        return 'W4A16 kept packed for a fused kernel when the GPU has shader-f16; otherwise expanded to F16 on the CPU at load.'
    if not has_q4k:
        return '16/32-bit weights; BF16 is cast to F16 or F32 on the GPU unless a kernel needs BF16.'
    if 'fused_matmul_q4' in files:
        return 'Q4_K kept packed and decoded inside fused matmul kernels; a tensor that an F16 step also reads gets a dense copy too.'
    return 'Graph has no fused Q4_K kernel, so the loader expands Q4_K to F16 at load (more GPU memory than the file size).'

def lanes_table(g):
    rows = []
    for lid in g['lanes']:
        r = data[lid]; s = st.get(lid, {})
        cls, txt = lane_status(lid)
        qs = ' · quickstart' if lid in QUICKSTART else ''
        ref = ' · borrows shards (weightsRef)' if r.get('weightsRef') else ''
        rows.append(f'''<tr><td id="{E(lid)}"><code>{E(lid)}</code><div class="sub">{formats_line(r)}</div></td>
<td class="num">{n(r["weights"])}</td><td class="num">{size(r.get("bytes"))}</td>
<td>{E(str(r.get("activationDtype") or "—"))} / {E(str(r.get("kvDtype") or "—"))}</td>
<td><span class="chip {cls}">{E(txt)}</span><div class="sub">{E(where(r))}{qs}{ref}</div></td></tr>''')
    return f'''<div class="scroll"><table class="lanes"><thead><tr><th>Stored lane and its number formats</th><th class="num">Weights</th><th class="num">On disk</th><th>Activations / KV</th><th>Status · location</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>'''

def walk(g):
    lid = g['lanes'][0]; r = data[lid]; s = st.get(lid, {}); a = r.get('arch') if isinstance(r.get('arch'), dict) else {}
    kv = kv_info(r, s)
    lt = r.get('layerTypes') or {}
    pipes = g['pipes']
    rows = []
    def row(label, body):
        rows.append(f'<tr><th>{label}</th><td>{body}</td></tr>')
    pipe_names = ', '.join(f'<a href="#p-{p}">{dict(PIPES)[p]}</a>' + (f' <span class="chip {"bad" if PIPE_STATE.get((g["key"], p)) in ("broken","rejected") else "warn"}">{PIPE_STATE[(g["key"], p)]}</span>' if (g['key'], p) in PIPE_STATE else '') for p in pipes)
    mt = r.get('modelType')
    route = {'transformer': 'InferencePipeline', 'gemma4': 'InferencePipeline', 'embedding': 'EmbeddingPipeline', 'diffusion_gemma': 'DiffusionGemmaPipeline', 'diffusion': 'DiffusionPipeline (experimental)'}.get(mt, mt)
    row('Pipelines', pipe_names)
    row('1 · Route', f'modelType <code>{E(str(mt))}</code> → {E(route)}')
    if lid.endswith('q4_0-af32'):
        path = '<a href="#path-gguf">GGUF passthrough</a>: llama.cpp blocks copied byte-for-byte into the manifest’s shards'
    elif mt == 'diffusion':
        path = '<a href="#path-rdrr">Converted RDRR</a> from a diffusers directory'
    else:
        path = '<a href="#path-rdrr">Converted RDRR</a> from SafeTensors'
    if any(data[x].get('weightsRef') for x in g['lanes']): path += '; one lane is a <a href="#path-ref">weightsRef</a> variant'
    if any(x in QUICKSTART for x in g['lanes']): path += '; quickstart ID downloads a pinned Hugging Face revision'
    row('2 · Weights in', path + '. ' + E(q4k_handling(lid)))
    if mt == 'diffusion':
        row('3 · Stages', 'Text encoder → 28-layer transformer → DC-AE decoder. Load stops at the layout check, so no stage runs.')
        return '<table class="walk">' + ''.join(rows) + '</table>'
    vocab = r.get('vocab') or a.get('vocabSize')
    tmpl = s.get('chatTemplate')
    seqcfg = s.get('sequence')
    if seqcfg:
        tok = f'{E(seqcfg.get("alphabet","").replace("_"," "))} alphabet, {n(vocab)} token IDs; no chat template'
    else:
        tok = f'{n(vocab)} token IDs; chat template <code>{E(str(tmpl))}</code>' + (' with thinking prefix' if s.get('thinking') else '') if tmpl else f'{n(vocab)} token IDs; no chat template'
    row('3 · Tokenize', tok)
    H = r.get('hidden') or a.get('hiddenSize')
    qi = (r.get('quant') or {}) if not lid.endswith('q4_0-af32') else {'embeddings': 'q4_0', 'lmHead': 'q6k'}
    scale = '× √hidden' if s.get('scaleEmbeddings') else (f'× {s["embeddingScale"]}' if s.get('embeddingScale') not in (None, 1) else 'unscaled')
    row('4 · Embed', f'Gather {n(H)} values per token from a {n(vocab)} × {n(H)} {E(str(qi.get("embeddings","")).upper())} table, {scale} → [T, {n(H)}]')
    # mixer
    nH = a.get('numAttentionHeads'); nKV = a.get('numKeyValueHeads'); D = a.get('headDim')
    parts = []
    L = a.get('numLayers')
    if lt:
        parts.append(' + '.join(f'{v} {k.replace("_attention"," attention").replace("conv","short-conv")}' for k, v in lt.items()) + f' = {L} layers')
    else:
        parts.append(f'{L} layers, all full attention' + ('' if s.get('causal') else ' (bidirectional)'))
    att = f'Attention: {nH} query heads, {nKV} KV heads × {D}'
    if a.get('globalHeadDim') and a.get('globalHeadDim') != D:
        att += f'; global layers {a.get("numGlobalKeyValueHeads") or nKV} KV heads × {a["globalHeadDim"]}'
    if s.get('slidingWindow') and lt.get('sliding_attention'): att += f'; sliding window {n(s["slidingWindow"])}'
    extras = []
    if s.get('qkNorm'): extras.append('Q/K RMSNorm')
    if s.get('outputGate'): extras.append('sigmoid output gate')
    if s.get('mrope'): extras.append('multi-axis RoPE')
    if s.get('partialRotary') and s['partialRotary'] != 1: extras.append(f'RoPE on {int(s["partialRotary"]*100)}% of dims')
    if s.get('ropeLocalTheta'): extras.append('separate local RoPE table')
    if s.get('ropeScaling'): extras.append(f'{s["ropeScaling"]} scaling')
    if s.get('attnSoftcap'): extras.append(f'attention softcap {s["attnSoftcap"]}')
    if extras: att += ' · ' + ', '.join(extras)
    parts.append(att)
    if lt.get('linear_attention'):
        nk = a.get('linearNumKeyHeads'); nv = a.get('linearNumValueHeads'); dk = a.get('linearKeyHeadDim')
        conv = 2 * nk * dk + nv * dk
        parts.append(f'Linear attention: {nk} key heads, {nv} value heads × {dk}; conv width {n(conv)} (kernel 4); gated-delta recurrence')
    row('5 · Mixer', '<br>'.join(E(p) for p in parts))
    I = r.get('ffn') or a.get('intermediateSize')
    moe = r.get('moe')
    act = s.get('activation')
    if moe:
        f = f'Mixture of experts: router [T, {n(moe["numExperts"])}] → top {moe["numExpertsPerToken"]} → experts of width {n(moe.get("expertIntermediateSize"))}'
        if moe.get('sharedExpertIntermediateSize'): f += f' + shared expert {n(moe["sharedExpertIntermediateSize"])}'
        if s.get('ffnBranch') == 'dense_plus_moe': f += f'; plus a dense {act} FFN of width {n(I)}, summed'
    else:
        f = f'Dense {"gated " if s.get("gated") else ""}{act}: [T, {n(H)}] → [T, {n(I)}] → [T, {n(H)}]'
        if s.get('doubleWide'): f += f'; last {a.get("numKvSharedLayers")} layers double width ({n(2*I)})'
    norms = 'LayerNorm' if s.get('normType') == 'layernorm' else ('RMSNorm (1 + w)' if s.get('weightOffset') else 'RMSNorm')
    if s.get('sandwich'): norms += ', sandwich norms around attention and FFN'
    row('6 · FFN and norms', E(f) + '<br>' + E(norms))
    if r.get('perLayerInputPerToken'):
        row('7 · Per-layer extras', 'Per-layer inputs: ' + f'{n(r["perLayerInputPerToken"])} values per token ({L} × {a.get("hiddenSizePerLayerInput")}): gather + projection, gated into each layer’s residual')
    elif g['key'] == 'flash-next':
        row('7 · Per-layer extras', 'Hyper-connections: a root mixer replaces the final norm; the QSA indexer selects KV rows per query')
    elif lt.get('conv'):
        row('7 · Per-layer extras', 'Short-conv layers keep a persistent convolution state')
    else:
        row('7 · Per-layer extras', 'None')
    # output
    pipes_set = set(pipes)
    if 'emb' in pipes_set and not seqcfg:
        ep = s.get('embPost') or {}
        projs = ep.get('projections') or []
        dim = projs[-1]['outputSize'] if projs else H
        out = f'Final norm → {ep.get("poolingMode","mean")} pool → ' + (' → '.join(f'dense {p["inputSize"]}→{p["outputSize"]}' for p in projs) + ' → ' if projs else '') + f'{ep.get("normalize","none")} → {n(dim)} values'
    elif seqcfg:
        out = f'Final norm → per-token embeddings [T, {n(H)}] + mean pool (excluding IDs {seqcfg["pooledEmbedding"]["excludeTokenIds"]})' + (f' + per-position logits [T, {n(vocab)}]' if seqcfg.get('logits') else '')
    elif 'rr' in pipes_set:
        out = f'Final norm → LM head for 2 token IDs only ("yes", "no") → score = difference → sigmoid'
    else:
        head = str(qi.get('lmHead', '')).upper()
        out = f'Final norm → LM head {"(tied to embedding) " if s.get("tied") else ""}{E(head)} [{n(vocab)} × {n(H)}] → {n(vocab)} F32 logits'
        if s.get('finalSoftcap'): out += f', softcap {s["finalSoftcap"]}'
        if 'dg' in pipes_set: out += '; per denoising step [256, 262,144]'
    row('8 · Output', out)
    # state
    st_parts = []
    if kv:
        st_parts.append(f'KV cache ({E(str(kv["dtype"]))}): {n(kv["perToken"])} values per token = {E(kv["detail"])}; {E(kv["layout"])}; sized for {n(kv["cap"])} tokens ({E(kv["capSource"])}) → <b>{size(kv["bytes"])}</b> allocated at load')
    if r.get('linearStateValues'):
        st_parts.append(f'Linear-attention state (F32, fixed): {n(r["linearStateValues"])} values = <b>{size(r["linearStateValues"]*4)}</b> across {r["linearLayers"]} layers')
    if lt.get('conv'):
        st_parts.append('Short-conv state per conv layer (persistent)')
    if seqcfg:
        st_parts.append('No KV allocation: the loader honors skipEmbeddingKVCacheWrites for sequence encoders (a76f1add)')
    elif 'emb' in pipes_set:
        st_parts.append('Used as the attention store for the single pass, but sized for the full token cap, not the input')
    elif 'rr' in pipes_set:
        st_parts.append('Used: the rerank harness prefills the shared prompt prefix once and rewinds to it per document')
    row('9 · State', '<br>'.join(st_parts) or '—')
    if not (pipes_set & {'gen', 'dg'}):
        row('10 · Decode', 'No decode loop: one pass per input')
    if 'gen' in pipes_set or 'dg' in pipes_set:
        bs = s.get('batchSize'); ri = s.get('readbackInterval')
        dec = f'Batch {bs} tokens per command buffer, read back every {ri} batch(es)' if bs else 'Decode-loop settings come from the runtime profile'
        if lt.get('conv'): dec = 'Batched decode refuses conv models: one token per step'
        if 'dg' in pipes_set: dec = 'No token loop: up to 48 denoising steps per 256-token canvas'
        row('10 · Decode', dec)
    # partition rule
    reasons = []
    if moe: reasons.append('MoE')
    if a.get('numKvSharedLayers'): reasons.append('shared KV')
    if r.get('perLayerInputPerToken'): reasons.append('per-layer inputs')
    if a.get('globalHeadDim') and a.get('globalHeadDim') != D: reasons.append('mixed KV geometry')
    if lt.get('conv'): reasons.append('conv layers')
    if not pipes_set & {'gen', 'tr'}: reasons.append('not a token-generation model')
    if g['key'] == 'flash-next': reasons.append('hyper-connections')
    row('Partitions', ('Blocked by ' + ', '.join(reasons)) if reasons else 'Allowed by the code rules (token-only causal layers); a rule check, not a run')
    return '<table class="walk">' + ''.join(rows) + '</table>'

def counts(g):
    r = data[g['lanes'][0]]; s = st.get(g['lanes'][0], {}); a = r.get('arch') if isinstance(r.get('arch'), dict) else {}
    kv = kv_info(r, s)
    vlabel = 'Logits per token' if set(g['pipes']) & {'gen','tr','dg'} else 'Vocabulary'
    items = [('Weights', n(r['weights'])), ('Hidden per token', n(r.get('hidden') or a.get('hiddenSize'))), (vlabel, n(r.get('vocab') or a.get('vocabSize')) if (r.get('vocab') or a.get('vocabSize')) else '—')]
    if kv: items.append(('KV values per token', n(kv['perToken'])))
    if r.get('linearStateValues'): items.append(('Fixed state values', n(r['linearStateValues'])))
    return '<dl class="counts">' + ''.join(f'<div><dt>{k}</dt><dd>{v}</dd></div>' for k, v in items) + '</dl>'

cards = []
for g in G:
    notes = ''.join(f'<li>{E(x)}</li>' for x in g['notes'])
    cards.append(f'''<article class="model" id="m-{g["key"]}"><header><h3>{E(g["name"])}</h3>{counts(g)}</header>
<ul class="notes">{notes}</ul>{lanes_table(g)}<h4>Pipeline walk</h4>{walk(g)}</article>''')

# matrix
mrows = []
for g in G:
    cells = []
    for p, _ in PIPES:
        if p in g['pipes']:
            stt = PIPE_STATE.get((g['key'], p), 'yes')
            cls = {'yes': 'ok', 'partial': 'warn', 'broken': 'bad', 'rejected': 'bad', 'fixed, not run': 'warn'}[stt]
            cells.append(f'<td><span class="dot {cls}">{stt}</span></td>')
        else:
            cells.append('<td></td>')
    lid = g['lanes'][0]; r = data[lid]
    mrows.append(f'<tr><th><a href="#m-{g["key"]}">{E(g["name"])}</a></th><td class="num">{n(r["weights"])}</td>{"".join(cells)}</tr>')
matrix = f'''<div class="scroll"><table class="matrix"><thead><tr><th>Model</th><th class="num">Weights</th>{"".join(f"<th>{E(t)}</th>" for _, t in PIPES)}</tr></thead><tbody>{"".join(mrows)}</tbody></table></div>'''

open(S + '/cards.html', 'w').write('\n'.join(cards))
open(S + '/matrix.html', 'w').write(matrix)
print('cards', len(cards), 'bytes', sum(len(c) for c in cards))
