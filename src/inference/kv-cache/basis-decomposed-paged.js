import { getDevice } from '../../gpu/device.js';
import { readBuffer } from '../../memory/buffer-pool.js';
import { KVCache } from './base.js';
import { f16ToF32Array, f32ToF16Array } from './types.js';
import { steamrollerRadixArgsort, generateBDPAData } from '../pipelines/text/bdpa-steamroller.js';

// ============================================================================
// BasisDecomposedPagedCache Class
// ============================================================================

export class BasisDecomposedPagedCache extends KVCache {
    /**
     * @param {import('./types.js').KVCacheConfig & { bdpaVocabSize?: number }} config
     */
    constructor(config) {
        super({
            ...config,
            // Force BDPA to identify as paged for downstream assertions, but we implement custom layout
            layout: 'bdpa_paged'
        });

        if (!config.useGPU) {
            throw new Error('BasisDecomposedPagedCache requires a GPU device.');
        }

        // Configurable BDPA hyperparameters
        this.basisVocabSize = config.bdpaVocabSize || 2048;
        this.pageSize = config.pageSize || 128;
        this.maxContextPages = Math.ceil(this.maxSeqLen / this.pageSize);

        // BDA Specific Memory Overrides
        this.basisDtype = 'f16';
        this.deltaDtype = 'int8'; // Or int4 packed

        const bytesPerBasis = this.headDim * 2; // f16
        const bytesPerDelta = this.headDim * 1; // int8

        this.memoryUsage = 0;
        this.layers = new Array(this.numLayers);
        this.tokenIds = new Int32Array(this.maxSeqLen);
        this.tokenIdsSet = new Uint8Array(this.maxSeqLen);

        // Allocate the 3-buffer system
        this._initializeBDPAStorage(bytesPerBasis, bytesPerDelta);
    }

    _initializeStorage() {
        // Override base storage. We allocate our own.
    }

    _initializeBDPAStorage(bytesPerBasis, bytesPerDelta) {
        const device = getDevice();
        if (!device) throw new Error('GPU Context missing during BDPA initialization');

        // Note: Node-WebGPU provides GPUBufferUsage globally, TS doesn't catch it locally without types.
        /** @ts-ignore */
        const standardUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

        for (let l = 0; l < this.numLayers; l++) {
            // 1. Vocabulary Basis Table (T_basis)
            const basisSize = this.basisVocabSize * bytesPerBasis * this.numHeads;
            const basisBufferK = device.createBuffer({
                label: `bdpa_basis_k_layer_${l}`,
                size: basisSize,
                usage: standardUsage,
            });
            const basisBufferV = device.createBuffer({
                label: `bdpa_basis_v_layer_${l}`,
                size: basisSize,
                usage: standardUsage,
            });

            // 2. Semantic Paged Cache (P_delta) (Int8 residuals packed into i32 lanes in WGSL).
            const rawPagedSize = this.maxContextPages * this.pageSize * bytesPerDelta * this.numHeads;
            const pagedSize = Math.ceil(rawPagedSize / 4) * 4;
            const pagedBufferK = device.createBuffer({
                label: `bdpa_paged_k_layer_${l}`,
                size: pagedSize,
                usage: standardUsage,
            });
            const pagedBufferV = device.createBuffer({
                label: `bdpa_paged_v_layer_${l}`,
                size: pagedSize,
                usage: standardUsage,
            });

            // 3. Execution Index (I_flat)
            // Structure: [BasisPtr (u32), DeltaPagePtr (u32), OriginalPos (u32)] x maxSeqLen
            const indexBytes = this.maxSeqLen * (3 * 4);
            const indexBuffer = device.createBuffer({
                label: `bdpa_index_layer_${l}`,
                size: indexBytes,
                usage: standardUsage,
            });

            this.layers[l] = {
                basisGPU: { k: basisBufferK, v: basisBufferV },
                pagedGPU: { k: pagedBufferK, v: pagedBufferV },
                indexGPU: indexBuffer,
                rawK: new Float32Array(this.maxSeqLen * this.kvSize),
                rawV: new Float32Array(this.maxSeqLen * this.kvSize),
                numBasisVectors: 0,
                seqLen: 0,
            };

            this.memoryUsage += (basisSize + pagedSize) * 2 + indexBytes;
        }
    }

    /**
   * @param {number} layerIdx
   * @returns {import('./types.js').BDPAGPUBuffersResult}
   */
    getGPUBuffers(layerIdx) {
        if (layerIdx < 0 || layerIdx >= this.numLayers) {
            throw new Error(`Invalid layer index: ${layerIdx}`);
        }
        const layer = this.layers[layerIdx];
        return {
            layout: 'bdpa',
            seqLen: layer.seqLen,
            basisGPU: layer.basisGPU,
            pagedGPU: layer.pagedGPU,
            indexGPU: layer.indexGPU,
            numBasisVectors: layer.numBasisVectors,
            pageSize: this.pageSize
        };
    }

    hasGPUCache() {
        const firstLayer = this.layers[0];
        return !!(firstLayer?.basisGPU?.k && firstLayer?.basisGPU?.v && firstLayer?.pagedGPU?.k && firstLayer?.pagedGPU?.v && firstLayer?.indexGPU);
    }

    clear() {
        this.currentSeqLen = 0;
        this.totalTokensSeen = 0;
        this.tokenIds.fill(0);
        this.tokenIdsSet.fill(0);
        for (let l = 0; l < this.numLayers; l++) {
            const layer = this.layers[l];
            layer.seqLen = 0;
            layer.numBasisVectors = 0;
        }
    }

    truncate(length) {
        if (length >= this.currentSeqLen) return;
        this.currentSeqLen = length;
        this.totalTokensSeen = Math.min(this.totalTokensSeen, length);
        for (let i = length; i < this.maxSeqLen; i++) {
            this.tokenIdsSet[i] = 0;
        }
        for (let l = 0; l < this.numLayers; l++) {
            const layer = this.layers[l];
            layer.seqLen = Math.min(layer.seqLen, length);
        }
    }

    recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens, tokenIds = null) {
        throw new Error(
            'BDPA cache does not support recorder-based KV ingestion. Disable command batching for BDPA runs.'
        );
    }

    async updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens, tokenIds = null) {
        this._assertLayerIndex(layerIdx);
        this._assertStartPos(startPos);
        if (!Number.isInteger(numTokens) || numTokens < 0) {
            throw new Error('BasisDecomposedPagedCache updateFromGPU requires a non-negative integer token count.');
        }
        if (numTokens === 0) {
            return;
        }
        if (startPos + numTokens > this.maxSeqLen) {
            throw new Error(`BDPA cache overflow: ${startPos + numTokens} > ${this.maxSeqLen}`);
        }
        if (!tokenIds || tokenIds.length !== numTokens) {
            throw new Error(`BDPA cache requires tokenIds for updateFromGPU (expected ${numTokens}, got ${tokenIds?.length ?? 0}).`);
        }

        const elems = numTokens * this.kvSize;
        const k = await this._decodeGPUFloatBuffer(keysBuffer, elems);
        const v = await this._decodeGPUFloatBuffer(valuesBuffer, elems);
        const layer = this.layers[layerIdx];
        const dstOffset = startPos * this.kvSize;
        layer.rawK.set(k, dstOffset);
        layer.rawV.set(v, dstOffset);
        this._writeTokenIds(tokenIds, startPos, numTokens);

        layer.seqLen = Math.max(layer.seqLen, startPos + numTokens);
        this.totalTokensSeen = Math.max(this.totalTokensSeen, startPos + numTokens);
        if (layerIdx === this.numLayers - 1) {
            this.currentSeqLen = Math.max(this.currentSeqLen, startPos + numTokens);
        }

        this._rebuildLayerBDPA(layerIdx);
    }

    _writeTokenIds(tokenIds, startPos, numTokens) {
        for (let i = 0; i < numTokens; i++) {
            const id = tokenIds[i];
            if (!Number.isInteger(id) || id < 0) {
                throw new Error(`Invalid token id at index ${i}: ${id}`);
            }
            const pos = startPos + i;
            this.tokenIds[pos] = id;
            this.tokenIdsSet[pos] = 1;
        }
    }

    async _decodeGPUFloatBuffer(buffer, elements) {
        const requiredBytes = elements * this.bytesPerElem;
        if (requiredBytes > buffer.size) {
            throw new Error(`BDPA update requires ${requiredBytes} bytes but source buffer has ${buffer.size} bytes.`);
        }
        const data = await readBuffer(buffer, requiredBytes);
        if (this.kvDtype === 'f16') {
            const u16 = new Uint16Array(data);
            return f16ToF32Array(u16);
        }
        return new Float32Array(data);
    }

    _rebuildLayerBDPA(layerIdx) {
        const layer = this.layers[layerIdx];
        const seqLen = layer.seqLen;
        if (seqLen <= 0) {
            layer.numBasisVectors = 0;
            return;
        }
        for (let i = 0; i < seqLen; i++) {
            if (this.tokenIdsSet[i] !== 1) {
                throw new Error(`BDPA token id missing at position ${i}.`);
            }
        }

        const kvElems = seqLen * this.kvSize;
        const tokens = this.tokenIds.subarray(0, seqLen);
        const sortedIndices = steamrollerRadixArgsort(tokens);
        const generated = generateBDPAData(
            sortedIndices,
            layer.rawK.subarray(0, kvElems),
            layer.rawV.subarray(0, kvElems),
            tokens,
            this.kvSize
        );

        if (generated.numBasisVectors > this.basisVocabSize) {
            throw new Error(
                `BDPA basis overflow at layer ${layerIdx}: ${generated.numBasisVectors} > basisVocabSize=${this.basisVocabSize}.`
            );
        }

        const device = getDevice();
        if (!device) {
            throw new Error('GPU context missing during BDPA rebuild.');
        }
        const basisKF16 = f32ToF16Array(generated.tBasisK);
        const basisVF16 = f32ToF16Array(generated.tBasisV);
        const packedDeltaK = this._packInt8ToInt32(generated.pDeltaK);
        const packedDeltaV = this._packInt8ToInt32(generated.pDeltaV);

        device.queue.writeBuffer(layer.basisGPU.k, 0, basisKF16);
        device.queue.writeBuffer(layer.basisGPU.v, 0, basisVF16);
        device.queue.writeBuffer(layer.pagedGPU.k, 0, packedDeltaK);
        device.queue.writeBuffer(layer.pagedGPU.v, 0, packedDeltaV);
        device.queue.writeBuffer(layer.indexGPU, 0, generated.iFlat);

        layer.numBasisVectors = generated.numBasisVectors;
    }

    _packInt8ToInt32(values) {
        const packedLen = Math.ceil(values.length / 4);
        const packed = new Int32Array(packedLen);
        for (let i = 0; i < values.length; i++) {
            const packedIdx = (i / 4) | 0;
            const lane = i % 4;
            const byte = values[i] & 0xFF;
            packed[packedIdx] |= (byte << (lane * 8));
        }
        return packed;
    }
}
