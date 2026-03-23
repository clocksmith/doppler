import { log } from '../debug/index.js';
import { evaluateHotSwapRollout, setHotSwapManifest } from './runtime.js';

function cleanupModel(model) {
  if (!model) return;
  try {
    if (typeof model.unload === 'function') {
      model.unload();
    } else if (typeof model.destroy === 'function') {
      model.destroy();
    } else if (typeof model.dispose === 'function') {
      model.dispose();
    }
  } catch (cleanupError) {
    log.warn('HotSwap', `Old model cleanup error: ${cleanupError?.message || cleanupError}`);
  }
}

export async function swapModel(currentModel, newModelLoader, policy = {}, context = {}) {
  const decision = evaluateHotSwapRollout(policy, context);
  if (!decision.allowed) {
    log.info('HotSwap', `Model swap not allowed: ${decision.reason}`);
    return { swapped: false, model: currentModel, decision };
  }

  let newModel = null;
  try {
    newModel = await newModelLoader(context);
    setHotSwapManifest(newModel?.manifest ?? null);
    log.info('HotSwap', 'Model swap succeeded');
  } catch (swapError) {
    log.warn('HotSwap', `Model swap failed: ${swapError?.message || swapError}`);
    if (newModel) {
      cleanupModel(newModel);
    }
    throw swapError;
  } finally {
    cleanupModel(currentModel);
  }

  return { swapped: true, model: newModel, decision };
}
