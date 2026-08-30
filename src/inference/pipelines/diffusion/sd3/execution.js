import { getDevice } from '../../../../gpu/device.js';
import { createTensor } from '../../../../gpu/tensor.js';
import { getBuffer } from '../../../../gpu/weight-buffer.js';
import { log } from '../../../../debug/index.js';
import { createSD3WeightResolver } from '../sd3-weights.js';
import {
  createDiffusionBufferDestroyer,
  createDiffusionBufferReleaser,
  createDiffusionIndexBuffer,
} from '../runtime-resources.js';
import { expectDiffusionWeight } from '../weight-contract.js';
import {
  createSD3PositionPlan,
  createSD3TransformerPlan,
  resolveSD3EmbeddingDtype,
  resolveSD3ModulationOffsets,
  resolveSD3ModulationSegments,
} from './plan.js';

export async function executeSD3Transformer(
  latents,
  context,
  timeText,
  weightsEntry,
  modelConfig,
  runtime,
  options = {},
  bindings
) {
  const {
    applyAdaLayerNorm,
    applyGate,
    applyQKNorm,
    buildModulation,
    concatKV,
    createBiasTensorWithDtype,
    createKernelOps,
    createVectorBuffer,
    runFFN,
    runMatmulResolved,
    runQKV,
  } = bindings;
  const device = getDevice();
  if (!device) {
    throw new Error('SD3 transformer requires a WebGPU device.');
  }

  const resolver = createSD3WeightResolver(weightsEntry, modelConfig);
  const recorder = options.recorder ?? null;
  const ops = createKernelOps(recorder);
  const release = createDiffusionBufferReleaser(recorder);
  const destroy = createDiffusionBufferDestroyer(recorder);
  const matmul = (input, weight, name, M, N, K, options = {}) =>
    runMatmulResolved(input, weight, resolver, name, M, N, K, { ...options, recorder });
  const config = modelConfig?.components?.transformer?.config || {};
  const transformerPlan = createSD3TransformerPlan(config, runtime, latents.shape);
  const {
    hiddenSize,
    numHeads,
    headDim,
    patchSize,
    layerNormEps,
    latentChannels,
    latentHeight,
    latentWidth,
    gridHeight,
    gridWidth,
    tokenCount,
    numLayers,
    dualAttentionLayers,
    attn2Layers: plannedAttn2Layers,
  } = transformerPlan;

  const projWeight = expectDiffusionWeight(resolver.get('pos_embed.proj.weight'), 'pos_embed.proj.weight');
  const projBias = resolver.get('pos_embed.proj.bias');

  const conv = await ops.conv2d(latents, projWeight, projBias, {
    inChannels: latentChannels,
    outChannels: hiddenSize,
    height: latentHeight,
    width: latentWidth,
    kernelH: patchSize,
    kernelW: patchSize,
    stride: patchSize,
    pad: 0,
  });

  const tokens = await ops.transpose(conv, hiddenSize, tokenCount);
  release(conv.buffer);

  const posEmbed = expectDiffusionWeight(resolver.get('pos_embed.pos_embed'), 'pos_embed.pos_embed');
  const posShape = resolver.shape('pos_embed.pos_embed') || [1, tokenCount, hiddenSize];
  const positionPlan = createSD3PositionPlan(gridHeight, gridWidth, posShape[1]);
  const { maxTokens } = positionPlan;
  if (!positionPlan.square) {
    log.warn('Diffusion', 'pos_embed size is not square; using sequential indices.');
  }
  const posIndices = Uint32Array.from(positionPlan.indices);

  const posBuffer = createDiffusionIndexBuffer(device, posIndices, 'sd3_pos_idx');
  const posEmbedKey = resolver.key('pos_embed.pos_embed');
  const posEmbedDtype = resolveSD3EmbeddingDtype(posEmbed?.dtype, weightsEntry?.dtypes?.get(posEmbedKey), runtime);
  const pos = await ops.gather(
    posBuffer,
    getBuffer(posEmbed),
    tokenCount,
    hiddenSize,
    maxTokens,
    {
      embeddingDtype: posEmbedDtype,
      outputDtype: tokens.dtype,
      transpose: false,
    }
  );
  destroy(posBuffer);

  const xCombined = await ops.residualAdd(tokens, pos, tokenCount * hiddenSize, { useVec4: true });
  release(tokens.buffer);
  release(pos.buffer);

  let x = createTensor(xCombined.buffer, xCombined.dtype, [tokenCount, hiddenSize], 'sd3_tokens');
  let ctx = context;
  let ctxOwned = false;

  const ones = new Float32Array(hiddenSize).fill(1.0);
  const zeros = new Float32Array(hiddenSize);
  const onesBuf = createVectorBuffer(device, ones, 'sd3_ln_weight');
  const zerosBuf = createVectorBuffer(device, zeros, 'sd3_ln_bias');

  const dualLayers = new Set(dualAttentionLayers);
  const attn2Layers = plannedAttn2Layers == null
    ? null
    : new Set(plannedAttn2Layers);

  for (let layerIdx = 0; layerIdx < numLayers; layerIdx++) {
    const modWeightName = `transformer_blocks.${layerIdx}.norm1.linear.weight`;
    const modBiasName = `transformer_blocks.${layerIdx}.norm1.linear.bias`;
    const modWeight = expectDiffusionWeight(
      resolver.get(modWeightName),
      modWeightName
    );
    const modBias = resolver.get(modBiasName);
    const modSegments = resolveSD3ModulationSegments(modWeight?.shape || resolver.shape(modWeightName), hiddenSize, 9, modWeightName);
    if (modSegments < 6) {
      throw new Error(`Unsupported modulation segments=${modSegments} for ${modWeightName}`);
    }
    const modBiasTensor = createBiasTensorWithDtype(
      modBias,
      hiddenSize * modSegments,
      'sd3_mod_bias',
      resolver,
      modBiasName
    );
    const mod = await buildModulation(timeText, modWeight, modBiasTensor, hiddenSize, modSegments, runtime, matmul, modWeightName, ops);

    const offsets = resolveSD3ModulationOffsets(modSegments, hiddenSize);
    const attnOffsets = offsets.attn;
    const attn2Offsets = offsets.attn2;
    const ffOffsets = offsets.ff;

    let ctxMod = null;
    let ctxOffsets = null;
    let ctxAttnOffsets = null;
    let ctxFfOffsets = null;
    if (dualLayers.has(layerIdx)) {
      const ctxWeightName = `transformer_blocks.${layerIdx}.norm1_context.linear.weight`;
      const ctxBiasName = `transformer_blocks.${layerIdx}.norm1_context.linear.bias`;
      const ctxWeight = expectDiffusionWeight(
        resolver.get(ctxWeightName),
        ctxWeightName
      );
      const ctxBias = resolver.get(ctxBiasName);
      const ctxSegments = resolveSD3ModulationSegments(ctxWeight?.shape || resolver.shape(ctxWeightName), hiddenSize, 6, ctxWeightName);
      if (ctxSegments < 6) {
        throw new Error(`Unsupported modulation segments=${ctxSegments} for ${ctxWeightName}`);
      }
      const ctxBiasTensor = createBiasTensorWithDtype(
        ctxBias,
        hiddenSize * ctxSegments,
        'sd3_ctx_mod_bias',
        resolver,
        ctxBiasName
      );
      ctxMod = await buildModulation(timeText, ctxWeight, ctxBiasTensor, hiddenSize, ctxSegments, runtime, matmul, ctxWeightName, ops);
      ctxOffsets = resolveSD3ModulationOffsets(ctxSegments, hiddenSize);
      ctxAttnOffsets = ctxOffsets.attn;
      ctxFfOffsets = ctxOffsets.ff;
    }

    const xAttnIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      attnOffsets,
      runtime,
      ops,
      release,
      { numTokens: tokenCount, hiddenSize }
    );

    if (dualLayers.has(layerIdx)) {
      const ctxAttnIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxAttnOffsets,
        runtime,
        ops,
        release,
        { numTokens: ctx.shape[0], hiddenSize }
      );

      const attnWeightNames = {
        q: `transformer_blocks.${layerIdx}.attn.to_q.weight`,
        k: `transformer_blocks.${layerIdx}.attn.to_k.weight`,
        v: `transformer_blocks.${layerIdx}.attn.to_v.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn.qkv.weight`,
      };
      const attnWeights = {
        q: resolver.get(attnWeightNames.q),
        k: resolver.get(attnWeightNames.k),
        v: resolver.get(attnWeightNames.v),
        qkv: resolver.get(attnWeightNames.qkv),
      };
      const attnBiasNames = {
        q: `transformer_blocks.${layerIdx}.attn.to_q.bias`,
        k: `transformer_blocks.${layerIdx}.attn.to_k.bias`,
        v: `transformer_blocks.${layerIdx}.attn.to_v.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn.qkv.bias`,
      };
      const attnBias = {
        q: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.q),
          hiddenSize,
          'sd3_attn_q_bias',
          resolver,
          attnBiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.k),
          hiddenSize,
          'sd3_attn_k_bias',
          resolver,
          attnBiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.v),
          hiddenSize,
          'sd3_attn_v_bias',
          resolver,
          attnBiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(attnBiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn_qkv_bias',
          resolver,
          attnBiasNames.qkv
        ),
      };
      const addWeightNames = {
        q: `transformer_blocks.${layerIdx}.attn.add_q_proj.weight`,
        k: `transformer_blocks.${layerIdx}.attn.add_k_proj.weight`,
        v: `transformer_blocks.${layerIdx}.attn.add_v_proj.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn.add_qkv.weight`,
      };
      const addWeights = {
        q: resolver.get(addWeightNames.q),
        k: resolver.get(addWeightNames.k),
        v: resolver.get(addWeightNames.v),
        qkv: resolver.get(addWeightNames.qkv),
      };
      const addBiasNames = {
        q: `transformer_blocks.${layerIdx}.attn.add_q_proj.bias`,
        k: `transformer_blocks.${layerIdx}.attn.add_k_proj.bias`,
        v: `transformer_blocks.${layerIdx}.attn.add_v_proj.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn.add_qkv.bias`,
      };
      const addBias = {
        q: createBiasTensorWithDtype(
          resolver.get(addBiasNames.q),
          hiddenSize,
          'sd3_attn_add_q_bias',
          resolver,
          addBiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(addBiasNames.k),
          hiddenSize,
          'sd3_attn_add_k_bias',
          resolver,
          addBiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(addBiasNames.v),
          hiddenSize,
          'sd3_attn_add_v_bias',
          resolver,
          addBiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(addBiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn_add_qkv_bias',
          resolver,
          addBiasNames.qkv
        ),
      };

      const normWeights = {
        q: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_q.weight`),
        k: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_k.weight`),
        qAdd: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_added_q.weight`),
        kAdd: resolver.get(`transformer_blocks.${layerIdx}.attn.norm_added_k.weight`),
      };

      let { q: qx, k: kx, v: vx } = await runQKV(
        xAttnIn,
        attnWeights,
        attnBias,
        tokenCount,
        hiddenSize,
        `sd3_attn_${layerIdx}`,
        matmul,
        attnWeightNames,
        ops,
        release,
        recorder
      );

      let { q: qc, k: kc, v: vc } = await runQKV(
        ctxAttnIn,
        addWeights,
        addBias,
        ctx.shape[0],
        hiddenSize,
        `sd3_attn_add_${layerIdx}`,
        matmul,
        addWeightNames,
        ops,
        release,
        recorder
      );

      if (normWeights.q) {
        const normed = await applyQKNorm(qx, normWeights.q, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(qx.buffer);
        qx = normed;
      }
      if (normWeights.k) {
        const normed = await applyQKNorm(kx, normWeights.k, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(kx.buffer);
        kx = normed;
      }
      if (normWeights.qAdd) {
        const normed = await applyQKNorm(qc, normWeights.qAdd, ctx.shape[0], numHeads, headDim, layerNormEps, ops);
        release(qc.buffer);
        qc = normed;
      }
      if (normWeights.kAdd) {
        const normed = await applyQKNorm(kc, normWeights.kAdd, ctx.shape[0], numHeads, headDim, layerNormEps, ops);
        release(kc.buffer);
        kc = normed;
      }

      const kAll = await concatKV(kx, kc, tokenCount, ctx.shape[0], hiddenSize, recorder);
      const vAll = await concatKV(vx, vc, tokenCount, ctx.shape[0], hiddenSize, recorder);

      const attnX = await ops.attention(qx, kAll, vAll, null, numHeads, headDim, {
        seqLen: tokenCount,
        kvLen: tokenCount + ctx.shape[0],
        numKVHeads: numHeads,
        causal: false,
      });

      const attnC = await ops.attention(qc, kAll, vAll, null, numHeads, headDim, {
        seqLen: ctx.shape[0],
        kvLen: tokenCount + ctx.shape[0],
        numKVHeads: numHeads,
        causal: false,
      });

      const outWeightName = `transformer_blocks.${layerIdx}.attn.to_out.0.weight`;
      const outWeight = expectDiffusionWeight(
        resolver.get(outWeightName),
        outWeightName
      );
      const outBiasName = `transformer_blocks.${layerIdx}.attn.to_out.0.bias`;
      const outBias = resolver.get(outBiasName);
      const outAddWeightName = `transformer_blocks.${layerIdx}.attn.to_add_out.weight`;
      const outAddWeight = expectDiffusionWeight(
        resolver.get(outAddWeightName),
        outAddWeightName
      );
      const outAddBiasName = `transformer_blocks.${layerIdx}.attn.to_add_out.bias`;
      const outAddBias = resolver.get(outAddBiasName);

      let attnOutX = await matmul(attnX, outWeight, outWeightName, tokenCount, hiddenSize, hiddenSize, {
        outputDtype: attnX.dtype,
        transposeB: 'auto',
      });
      if (outBias) {
        attnOutX = await ops.biasAdd(
          attnOutX,
          createBiasTensorWithDtype(outBias, hiddenSize, 'sd3_attn_out_bias', resolver, outBiasName),
          tokenCount,
          hiddenSize
        );
      }

      let attnOutC = await matmul(attnC, outAddWeight, outAddWeightName, ctx.shape[0], hiddenSize, hiddenSize, {
        outputDtype: attnC.dtype,
        transposeB: 'auto',
      });
      if (outAddBias) {
        attnOutC = await ops.biasAdd(
          attnOutC,
          createBiasTensorWithDtype(outAddBias, hiddenSize, 'sd3_attn_out_add_bias', resolver, outAddBiasName),
          ctx.shape[0],
          hiddenSize
        );
      }

      const gatedX = await applyGate(attnOutX, mod, attnOffsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
      const gatedC = await applyGate(attnOutC, ctxMod, ctxAttnOffsets, ops, release, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });

      const xRes = await ops.residualAdd(x, gatedX, tokenCount * hiddenSize, { useVec4: true });
      const cRes = await ops.residualAdd(ctx, gatedC, ctx.shape[0] * hiddenSize, { useVec4: true });

      release(xAttnIn.buffer);
      release(ctxAttnIn.buffer);
      release(qx.buffer);
      release(kx.buffer);
      release(vx.buffer);
      release(qc.buffer);
      release(kc.buffer);
      release(vc.buffer);
      release(kAll.buffer);
      release(vAll.buffer);
      release(attnX.buffer);
      release(attnC.buffer);
      release(gatedX.buffer);
      release(gatedC.buffer);
      release(x.buffer);
      if (ctxOwned) {
        release(ctx.buffer);
      }

      x = createTensor(xRes.buffer, xRes.dtype, [tokenCount, hiddenSize], 'sd3_x');
      ctx = createTensor(cRes.buffer, cRes.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');
      ctxOwned = true;

      const ctxFfIn = await applyAdaLayerNorm(
        ctx,
        onesBuf,
        zerosBuf,
        layerNormEps,
        ctxMod,
        ctxFfOffsets,
        runtime,
        ops,
        release,
        { numTokens: ctx.shape[0], hiddenSize }
      );

      const ffCtxWeightNames = {
        up: `transformer_blocks.${layerIdx}.ff_context.net.0.proj.weight`,
        down: `transformer_blocks.${layerIdx}.ff_context.net.2.weight`,
      };
      const ffCtxWeights = {
        up: expectDiffusionWeight(
          resolver.get(ffCtxWeightNames.up),
          ffCtxWeightNames.up
        ),
        down: expectDiffusionWeight(
          resolver.get(ffCtxWeightNames.down),
          ffCtxWeightNames.down
        ),
      };
      const ffCtxBiasNames = {
        up: `transformer_blocks.${layerIdx}.ff_context.net.0.proj.bias`,
        down: `transformer_blocks.${layerIdx}.ff_context.net.2.bias`,
      };
      const ffCtxBias = {
        up: createBiasTensorWithDtype(
          resolver.get(ffCtxBiasNames.up),
          ffCtxWeights.up.shape[0],
          'sd3_ff_ctx_up_bias',
          resolver,
          ffCtxBiasNames.up
        ),
        down: createBiasTensorWithDtype(
          resolver.get(ffCtxBiasNames.down),
          hiddenSize,
          'sd3_ff_ctx_down_bias',
          resolver,
          ffCtxBiasNames.down
        ),
      };
      const ffCtxOut = await runFFN(
        ctxFfIn,
        ffCtxWeights,
        ffCtxBias,
        ctx.shape[0],
        hiddenSize,
        runtime,
        matmul,
        ffCtxWeightNames,
        ops,
        release
      );
      const ffCtxGated = await applyGate(ffCtxOut, ctxMod, ctxFfOffsets, ops, release, { numTokens: ctx.shape[0], hiddenSize, zeroOffset: ctxMod.zeroOffset });
      const ctxRes2 = await ops.residualAdd(ctx, ffCtxGated, ctx.shape[0] * hiddenSize, { useVec4: true });

      release(ctxFfIn.buffer);
      release(ffCtxGated.buffer);
      if (ctxOwned) {
        release(ctx.buffer);
      }
      ctx = createTensor(ctxRes2.buffer, ctxRes2.dtype, [ctx.shape[0], hiddenSize], 'sd3_ctx');
      ctxOwned = true;

    } else {
      release(xAttnIn.buffer);
    }

    const hasAttn2 = attn2Layers ? attn2Layers.has(layerIdx) : config.dual_attention_layers ? dualLayers.has(layerIdx) : true;
    if (hasAttn2) {
      const xAttn2In = await applyAdaLayerNorm(
        x,
        onesBuf,
        zerosBuf,
        layerNormEps,
        mod,
        attn2Offsets,
        runtime,
        ops,
        release,
        { numTokens: tokenCount, hiddenSize }
      );

      const attn2WeightNames = {
        q: `transformer_blocks.${layerIdx}.attn2.to_q.weight`,
        k: `transformer_blocks.${layerIdx}.attn2.to_k.weight`,
        v: `transformer_blocks.${layerIdx}.attn2.to_v.weight`,
        qkv: `transformer_blocks.${layerIdx}.attn2.qkv.weight`,
      };
      const attn2Weights = {
        q: resolver.get(attn2WeightNames.q),
        k: resolver.get(attn2WeightNames.k),
        v: resolver.get(attn2WeightNames.v),
        qkv: resolver.get(attn2WeightNames.qkv),
      };
      const attn2BiasNames = {
        q: `transformer_blocks.${layerIdx}.attn2.to_q.bias`,
        k: `transformer_blocks.${layerIdx}.attn2.to_k.bias`,
        v: `transformer_blocks.${layerIdx}.attn2.to_v.bias`,
        qkv: `transformer_blocks.${layerIdx}.attn2.qkv.bias`,
      };
      const attn2Bias = {
        q: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.q),
          hiddenSize,
          'sd3_attn2_q_bias',
          resolver,
          attn2BiasNames.q
        ),
        k: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.k),
          hiddenSize,
          'sd3_attn2_k_bias',
          resolver,
          attn2BiasNames.k
        ),
        v: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.v),
          hiddenSize,
          'sd3_attn2_v_bias',
          resolver,
          attn2BiasNames.v
        ),
        qkv: createBiasTensorWithDtype(
          resolver.get(attn2BiasNames.qkv),
          hiddenSize * 3,
          'sd3_attn2_qkv_bias',
          resolver,
          attn2BiasNames.qkv
        ),
      };

      let { q: q2, k: k2, v: v2 } = await runQKV(
        xAttn2In,
        attn2Weights,
        attn2Bias,
        tokenCount,
        hiddenSize,
        `sd3_attn2_${layerIdx}`,
        matmul,
        attn2WeightNames,
        ops,
        release,
        recorder
      );

      const normQ2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_q.weight`);
      const normK2 = resolver.get(`transformer_blocks.${layerIdx}.attn2.norm_k.weight`);
      if (normQ2) {
        const normed = await applyQKNorm(q2, normQ2, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(q2.buffer);
        q2 = normed;
      }
      if (normK2) {
        const normed = await applyQKNorm(k2, normK2, tokenCount, numHeads, headDim, layerNormEps, ops);
        release(k2.buffer);
        k2 = normed;
      }

      const attn2 = await ops.attention(q2, k2, v2, null, numHeads, headDim, {
        seqLen: tokenCount,
        kvLen: tokenCount,
        numKVHeads: numHeads,
        causal: false,
      });

      const attn2OutWeightName = `transformer_blocks.${layerIdx}.attn2.to_out.0.weight`;
      const attn2OutWeight = expectDiffusionWeight(
        resolver.get(attn2OutWeightName),
        attn2OutWeightName
      );
      const attn2OutBiasName = `transformer_blocks.${layerIdx}.attn2.to_out.0.bias`;
      const attn2OutBias = resolver.get(attn2OutBiasName);
      let attn2Out = await matmul(attn2, attn2OutWeight, attn2OutWeightName, tokenCount, hiddenSize, hiddenSize, {
        outputDtype: attn2.dtype,
        transposeB: 'auto',
      });
      if (attn2OutBias) {
        attn2Out = await ops.biasAdd(
          attn2Out,
          createBiasTensorWithDtype(attn2OutBias, hiddenSize, 'sd3_attn2_out_bias', resolver, attn2OutBiasName),
          tokenCount,
          hiddenSize
        );
      }

      const gated2 = await applyGate(attn2Out, mod, attn2Offsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
      const xRes2 = await ops.residualAdd(x, gated2, tokenCount * hiddenSize, { useVec4: true });

      release(xAttn2In.buffer);
      release(q2.buffer);
      release(k2.buffer);
      release(v2.buffer);
      release(attn2.buffer);
      release(attn2Out.buffer);
      release(gated2.buffer);
      release(x.buffer);

      x = createTensor(xRes2.buffer, xRes2.dtype, [tokenCount, hiddenSize], 'sd3_x');
    }

    const xFfIn = await applyAdaLayerNorm(
      x,
      onesBuf,
      zerosBuf,
      layerNormEps,
      mod,
      ffOffsets,
      runtime,
      ops,
      release,
      { numTokens: tokenCount, hiddenSize }
    );

    const ffWeightNames = {
      up: `transformer_blocks.${layerIdx}.ff.net.0.proj.weight`,
      down: `transformer_blocks.${layerIdx}.ff.net.2.weight`,
    };
    const ffWeights = {
      up: expectDiffusionWeight(
        resolver.get(ffWeightNames.up),
        ffWeightNames.up
      ),
      down: expectDiffusionWeight(
        resolver.get(ffWeightNames.down),
        ffWeightNames.down
      ),
    };
    const ffBiasNames = {
      up: `transformer_blocks.${layerIdx}.ff.net.0.proj.bias`,
      down: `transformer_blocks.${layerIdx}.ff.net.2.bias`,
    };
    const ffBias = {
      up: createBiasTensorWithDtype(
        resolver.get(ffBiasNames.up),
        ffWeights.up.shape[0],
        'sd3_ff_up_bias',
        resolver,
        ffBiasNames.up
      ),
      down: createBiasTensorWithDtype(
        resolver.get(ffBiasNames.down),
        hiddenSize,
        'sd3_ff_down_bias',
        resolver,
        ffBiasNames.down
      ),
    };

    const ffOut = await runFFN(
      xFfIn,
      ffWeights,
      ffBias,
      tokenCount,
      hiddenSize,
      runtime,
      matmul,
      ffWeightNames,
      ops,
      release
    );
    const ffGated = await applyGate(ffOut, mod, ffOffsets, ops, release, { numTokens: tokenCount, hiddenSize, zeroOffset: mod.zeroOffset });
    const xRes3 = await ops.residualAdd(x, ffGated, tokenCount * hiddenSize, { useVec4: true });

    release(xFfIn.buffer);
    release(ffGated.buffer);
    release(x.buffer);

    x = createTensor(xRes3.buffer, xRes3.dtype, [tokenCount, hiddenSize], 'sd3_x');

    release(mod.tensor.buffer);
    if (ctxMod?.tensor?.buffer) {
      release(ctxMod.tensor.buffer);
    }
  }

  const normOutWeightName = 'norm_out.linear.weight';
  const normOutWeight = expectDiffusionWeight(resolver.get(normOutWeightName), normOutWeightName);
  const normOutBias = resolver.get('norm_out.linear.bias');
  const normOutSegments = resolveSD3ModulationSegments(normOutWeight?.shape || resolver.shape(normOutWeightName), hiddenSize, 2, normOutWeightName);
  const normOutBiasTensor = createBiasTensorWithDtype(
    normOutBias,
    hiddenSize * normOutSegments,
    'sd3_norm_out_bias',
    resolver,
    'norm_out.linear.bias'
  );
  const normOut = await buildModulation(timeText, normOutWeight, normOutBiasTensor, hiddenSize, normOutSegments, runtime, matmul, normOutWeightName, ops);

  const xNorm = await ops.layerNorm(x, onesBuf, zerosBuf, layerNormEps, { batchSize: tokenCount, hiddenSize, normWeightDtype: 'f32' });
  const xMod = await ops.modulate(xNorm, normOut.tensor, {
    numTokens: tokenCount,
    hiddenSize,
    scaleOffset: 0,
    shiftOffset: hiddenSize,
    gateOffset: 0,
    hasGate: false,
    addOne: true,
  });

  release(xNorm.buffer);
  release(x.buffer);
  release(normOut.tensor.buffer);
  if (ctxOwned) {
    release(ctx.buffer);
  }
  release(onesBuf);
  release(zerosBuf);

  const projOutWeightName = 'proj_out.weight';
  const projOutWeight = expectDiffusionWeight(resolver.get(projOutWeightName), projOutWeightName);
  const projOutBiasName = 'proj_out.bias';
  const projOutBias = resolver.get(projOutBiasName);
  let patch = await matmul(xMod, projOutWeight, projOutWeightName, tokenCount, projOutWeight.shape[0], hiddenSize, {
    outputDtype: xMod.dtype,
    transposeB: 'auto',
  });
  if (projOutBias) {
    patch = await ops.biasAdd(
      patch,
      createBiasTensorWithDtype(projOutBias, projOutWeight.shape[0], 'sd3_proj_out_bias', resolver, projOutBiasName),
      tokenCount,
      projOutWeight.shape[0]
    );
  }

  release(xMod.buffer);

  const patchChannels = projOutWeight.shape[0];
  const output = await ops.pixelShuffle(patch, {
    outChannels: latentChannels,
    outHeight: latentHeight,
    outWidth: latentWidth,
    gridWidth,
    gridHeight,
    patchSize,
    patchChannels,
  });

  release(patch.buffer);

  return output;
}
