import { isCpuWeightBuffer } from "../../../../../gpu/weight-buffer.js";
import { computeLogits, recordLogitsGPU, extractLastPositionLogits } from "../../logits/index.js";
import { getLogitsConfig, getLogitsWeights } from "../../logits/logits-config.js";
import { recordPrefillRecorderStats } from "../session/text.js";
import { selectRuleValue } from "../../../../../rules/rule-registry.js";
import { releaseBuffer, readBufferSlice } from "../../../../../memory/buffer-pool.js";
import { decodeReadback } from "../../debug-utils/readback-decoding.js";
import { log, trace } from "../../../../../debug/index.js";
import { getLogitsHealth } from "../../debug-utils/health-metrics.js";

export async function computePrefillLogits({
  config,
  currentRecorder,
  opts,
  selectedLogitTokenIds,
  numTokens,
  tracePrefillEnabled,
  prefillBuffers,
  recordProfile,
  debugCheckBuffer,
  preserveBufferAcrossRecorderSubmit,
  submitFinalRecorder,
  state
}) {
      let lastLogits;
      let logitsVocabSize = config.vocabSize;
      const lmHead = state.weights.get('lm_head');
      const canRecordLogits = !!currentRecorder
        && opts._returnGpuLogits !== true
        && selectedLogitTokenIds === null
        && !!lmHead
        && !isCpuWeightBuffer(lmHead)
        && !state.disableRecordedLogits
        && numTokens === 1;
      if (currentRecorder && canRecordLogits) {
        const logitsTraceStart = tracePrefillEnabled ? performance.now() : 0;
        const recorded = await recordLogitsGPU(
          currentRecorder,
          prefillBuffers.hidden,
          numTokens,
          getLogitsWeights(state),
          getLogitsConfig(state, 'prefill'),
          state.operatorDiagnostics,
          { applySoftcap: true }
        );
        logitsVocabSize = recorded.vocabSize;

        recordPrefillRecorderStats(state, currentRecorder);
        await currentRecorder.submitAndWait();
        await recordProfile(currentRecorder);

        const logitsBytes = selectRuleValue('shared', 'dtype', 'bytesFromDtype', { dtype: recorded.logitsDtype });
        const lastLogitsSize = logitsVocabSize * logitsBytes;
        const lastLogitsOffset = (numTokens - 1) * lastLogitsSize;
        const logitsData = await readBufferSlice(recorded.logitsBuffer, lastLogitsOffset, lastLogitsSize);
        releaseBuffer(recorded.logitsBuffer);
        lastLogits = decodeReadback(logitsData, recorded.logitsDtype);
        if (tracePrefillEnabled) {
          trace.perf('Prefill recorded logits complete', {
            numTokens,
            vocabSize: logitsVocabSize,
            elapsedMs: performance.now() - logitsTraceStart,
          });
        }

        const health = getLogitsHealth(lastLogits);
        if (health.nanCount > 0 || health.infCount > 0 || health.nonZeroCount === 0) {
          log.warn(
            'Logits',
            `Recorded logits invalid (nan=${health.nanCount} inf=${health.infCount} nonZero=${health.nonZeroCount}, maxAbs=${health.maxAbs.toFixed(3)}); recomputing without recorder.`
          );
          state.disableRecordedLogits = true;
          state.disableFusedDecode = true;
          const fallbackLogits = await computeLogits(
            prefillBuffers.hidden,
            numTokens,
            getLogitsWeights(state),
            getLogitsConfig(state, 'prefill'),
            state.useGPU,
            state.debugFlags,
            undefined,
            debugCheckBuffer,
            state.runtimeConfig.shared.debug.probes,
            { lastPositionOnly: true },
            state.operatorDiagnostics
          );
          const fallbackHealth = getLogitsHealth(fallbackLogits);
          if (fallbackHealth.nanCount > 0 || fallbackHealth.infCount > 0 || fallbackHealth.nonZeroCount === 0) {
            throw new Error(
              `[Logits] Fallback logits invalid (nan=${fallbackHealth.nanCount} inf=${fallbackHealth.infCount} nonZero=${fallbackHealth.nonZeroCount}, maxAbs=${fallbackHealth.maxAbs.toFixed(3)}). ` +
              'This indicates upstream kernel output is NaN/Inf (often prefill attention/matmul).'
            );
          }
          logitsVocabSize = config.vocabSize;
          lastLogits = fallbackLogits.length === logitsVocabSize
            ? fallbackLogits
            : extractLastPositionLogits(fallbackLogits, numTokens, logitsVocabSize);
        }

      } else {
        const logitsTraceStart = tracePrefillEnabled ? performance.now() : 0;
        if (currentRecorder) {
          prefillBuffers.hidden = preserveBufferAcrossRecorderSubmit(
            prefillBuffers.hidden,
            currentRecorder,
            'prefill_logits_carry'
          );
          recordPrefillRecorderStats(state, currentRecorder);
          await submitFinalRecorder(currentRecorder);
          await recordProfile(currentRecorder);
        }
        const logits = await computeLogits(
          prefillBuffers.hidden,
          numTokens,
          getLogitsWeights(state),
          getLogitsConfig(state, 'prefill'),
          state.useGPU,
          state.debugFlags,
          undefined,
          debugCheckBuffer,
          state.runtimeConfig.shared.debug.probes,
          { lastPositionOnly: opts._returnAllLogits !== true, selectedTokenIds: selectedLogitTokenIds, returnGpuBuffer: opts._returnGpuLogits === true },
          state.operatorDiagnostics
        );

        if (opts._returnAllLogits === true || opts._returnGpuLogits === true || selectedLogitTokenIds !== null) {
          lastLogits = logits;
        } else {
          lastLogits = logits.length === logitsVocabSize
            ? logits
            : extractLastPositionLogits(logits, numTokens, logitsVocabSize);
        }

        if (tracePrefillEnabled) {
          trace.perf('Prefill logits complete', {
            numTokens,
            vocabSize: logitsVocabSize,
            elapsedMs: performance.now() - logitsTraceStart,
          });
        }
      }

  return lastLogits;
}
