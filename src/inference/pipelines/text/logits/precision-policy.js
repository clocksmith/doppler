import {
  getKernelFilePrecisionPatch,
  resolveF16ToF32ActivationKernel,
} from '../../../../config/transforms/execution-graph-transforms.js';
import { selectRuleValue } from '../../../../rules/rule-registry.js';

function requireLogitsConfigValue(condition, label, value) {
  if (!condition) {
    throw new Error(`[Logits] stable F32 policy requires ${label}, got ${String(value)}.`);
  }
}

export function shouldForceStableF32Logits(config, inputDtype) {
  requireLogitsConfigValue(inputDtype === 'f16' || inputDtype === 'f32', 'input dtype "f16" or "f32"', inputDtype);
  requireLogitsConfigValue(config && typeof config === 'object', 'a logits config object', config);
  requireLogitsConfigValue(
    config.finalLogitSoftcapping === null || Number.isFinite(config.finalLogitSoftcapping),
    'config.finalLogitSoftcapping to be null or finite',
    config.finalLogitSoftcapping
  );
  requireLogitsConfigValue(
    typeof config.rmsNormWeightOffset === 'boolean',
    'config.rmsNormWeightOffset to be boolean',
    config.rmsNormWeightOffset
  );
  requireLogitsConfigValue(
    Number.isFinite(config.hiddenSize) && config.hiddenSize > 0,
    'config.hiddenSize to be a positive finite number',
    config.hiddenSize
  );

  return selectRuleValue('inference', 'config', 'stableF32Logits', {
    inputDtype,
    finalLogitSoftcapping: config.finalLogitSoftcapping,
    rmsNormWeightOffset: config.rmsNormWeightOffset,
    hiddenSize: config.hiddenSize,
  });
}

export function createStableF32LogitsKernelPath(kernelPath) {
  if (!kernelPath?.postLayer) {
    return kernelPath;
  }
  let changed = false;
  const postLayer = kernelPath.postLayer.map((step) => {
    if (step?.op === 'final_norm') {
      const precision = {
        ...(step.precision ?? {}),
        inputDtype: 'f32',
        outputDtype: 'f32',
      };
      if (
        step.precision?.inputDtype === precision.inputDtype
        && step.precision?.outputDtype === precision.outputDtype
      ) {
        return step;
      }
      changed = true;
      return {
        ...step,
        precision,
      };
    }
    if (step?.op !== 'lm_head' && step?.op !== 'lm_head_prefill') {
      return step;
    }
    const replacement = resolveF16ToF32ActivationKernel(step.kernel);
    const existingPrecision = getKernelFilePrecisionPatch(step.kernel);
    const kernelAlreadyF32 = existingPrecision?.inputDtype === 'f32'
      && existingPrecision?.outputDtype === 'f32';
    if (!replacement && !kernelAlreadyF32) {
      throw new Error(
        `[Logits] stable F32 policy cannot map LM-head kernel "${step.kernel}" to an F32 activation kernel.`
      );
    }
    const kernel = replacement ?? step.kernel;
    const precision = {
      ...(step.precision ?? {}),
      inputDtype: 'f32',
      outputDtype: 'f32',
    };
    if (
      kernel === step.kernel
      && step.precision?.inputDtype === precision.inputDtype
      && step.precision?.outputDtype === precision.outputDtype
    ) {
      return step;
    }
    changed = true;
    return {
      ...step,
      kernel,
      precision,
    };
  });
  if (!changed) {
    return kernelPath;
  }
  return {
    ...kernelPath,
    postLayer,
  };
}
