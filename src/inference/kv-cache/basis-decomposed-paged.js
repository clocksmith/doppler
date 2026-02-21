import { getDevice } from '../../gpu/device.js';
import { KVCache } from './base.js';

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

            // 2. Semantic Paged Cache (P_delta) (Int8 Residuals)
            const pagedSize = this.maxContextPages * this.pageSize * bytesPerDelta * this.numHeads;
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
                seqLen: 0
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
            seqLen: this.currentSeqLen,
            basisGPU: layer.basisGPU,
            pagedGPU: layer.pagedGPU,
            indexGPU: layer.indexGPU,
            pageSize: this.pageSize
        };
    }

    // BDPA updates bypassing the standard `recordUpdateFromGPU`.
    // The Steamroller algorithm is responsible for memory ingestion updates.
    async recordUpdateFromGPU(recorder, layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
        throw new Error('Linear recordUpdateFromGPU is disabled for BDPA. Use Steamroller ingestion bindings.');
    }

    updateFromGPU(layerIdx, keysBuffer, valuesBuffer, startPos, numTokens) {
        throw new Error('Linear updateFromGPU is disabled for BDPA. Use Steamroller ingestion bindings.');
    }
}
