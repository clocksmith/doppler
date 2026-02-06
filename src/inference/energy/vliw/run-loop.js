import { DEFAULT_MLP_CONFIG, DEFAULT_WEIGHTS, ENGINE_ORDER } from './constants.js';
import { createRng } from './rng.js';
import { buildTaskIndex } from './tasks.js';
import { buildGraph } from './graph.js';
import { computeGraphMetrics } from './metrics.js';
import { buildBaselinePriorities } from './baseline.js';
import { scheduleFromBundles } from './schedule-bundles.js';
import { scheduleGraphWithPolicies } from './schedule-latency.js';
import { scheduleWithPriority, scheduleWithHeuristic } from './schedule-priority.js';
import { initPriorities, perturbPriorities } from './priorities.js';
import { initWeights, perturbWeights } from './weights.js';
import { createMlp, perturbMlp } from './mlp.js';
import { createMlpTrainer, mlpTrainDistillStep, disposeMlpTrainer } from './mlp-trainer.js';
import { resolveScheduleEnergy } from './energy.js';
import { SLOT_LIMITS } from '../vliw-shared.js';

function countTasksByEngine(tasks) {
  const counts = {};
  tasks.forEach((task) => {
    if (!task || task.engine === 'debug') return;
    counts[task.engine] = (counts[task.engine] || 0) + 1;
  });
  return counts;
}

function lowerBoundCycles(tasks, caps) {
  const counts = countTasksByEngine(tasks);
  let lb = 0;
  Object.entries(caps || {}).forEach(([engine, cap]) => {
    if (engine === 'debug') return;
    const total = counts?.[engine] ?? 0;
    if (!total) return;
    lb = Math.max(lb, Math.ceil(total / Math.max(1, cap || 0)));
  });
  return lb;
}

function resolveScoreMode(scoreMode) {
  if (scoreMode && scoreMode !== 'auto') return scoreMode;
  return 'graph';
}

export async function runVliwEnergyLoop({
  tasks,
  caps,
  dependencyModel,
  loop,
  search,
  seed,
  initMode,
  initScale,
  diagnostics,
  onProgress,
  onTrace,
}) {
  if (!Array.isArray(tasks) || !tasks.length) {
    throw new Error('VLIW demo requires a non-empty task list.');
  }
  const taskList = buildTaskIndex(tasks);
  const graph = buildGraph(taskList);
  const constraintMode = search?.mode === 'relaxed' ? 'relaxed' : 'parity';
  const scoreMode = resolveScoreMode(search?.scoreMode);
  const capsSource = search?.capsSource === 'slot_limits' ? 'slot_limits' : 'spec';
  const resolvedCaps = capsSource === 'slot_limits' ? SLOT_LIMITS : caps;
  const engineOrder = Array.isArray(search?.engineOrder) && search.engineOrder.length
    ? search.engineOrder
    : ENGINE_ORDER;
  const taskMeta = taskList.map((task) => ({
    id: task.id,
    engine: task.engine,
    bundle: task.bundle ?? null,
    deps: Array.isArray(graph.deps?.[task.id]) ? graph.deps[task.id].length : 0,
    reads: Array.isArray(task.reads) ? task.reads.length : 0,
    writes: Array.isArray(task.writes) ? task.writes.length : 0,
  }));
  const maxSteps = Math.max(1, Math.floor(loop?.maxSteps ?? 200));
  const minSteps = Math.max(1, Math.floor(loop?.minSteps ?? 1));
  const stepSize = Number.isFinite(loop?.stepSize) ? loop.stepSize : 0.25;
  const gradientScale = Number.isFinite(loop?.gradientScale) ? loop.gradientScale : 1.0;
  const convergenceThreshold = Number.isFinite(loop?.convergenceThreshold)
    ? loop.convergenceThreshold
    : null;

  const readbackEvery = Math.max(1, Math.floor(diagnostics?.readbackEvery ?? 5));
  const historyLimit = Math.max(1, Math.floor(diagnostics?.historyLimit ?? 200));

  const restarts = Math.max(1, Math.floor(search?.restarts ?? 1));
  let tempStart = Number.isFinite(search?.temperatureStart) ? search.temperatureStart : 2.5;
  const tempDecay = Number.isFinite(search?.temperatureDecay) ? search.temperatureDecay : 0.985;
  const mutationCount = Math.max(1, Math.floor(search?.mutationCount ?? Math.max(1, gradientScale * 4)));
  const policy = search?.policy === 'priorities'
    ? 'priorities'
    : (search?.policy === 'mlp' ? 'mlp' : 'weights');
  const mlpInputSize = 5;
  const mlpHiddenSize = policy === 'mlp'
    ? Math.max(1, Math.floor(
      Number.isFinite(search?.mlp?.hiddenSize)
        ? search.mlp.hiddenSize
        : DEFAULT_MLP_CONFIG.hiddenSize
    ))
    : null;
  const mlpTrainerConfig = policy === 'mlp'
    ? {
      lr: Number.isFinite(search?.mlp?.lr) ? search.mlp.lr : DEFAULT_MLP_CONFIG.lr,
      beta1: Number.isFinite(search?.mlp?.beta1) ? search.mlp.beta1 : DEFAULT_MLP_CONFIG.beta1,
      beta2: Number.isFinite(search?.mlp?.beta2) ? search.mlp.beta2 : DEFAULT_MLP_CONFIG.beta2,
      eps: Number.isFinite(search?.mlp?.eps) ? search.mlp.eps : DEFAULT_MLP_CONFIG.eps,
    }
    : null;
  const jitter = Number.isFinite(search?.jitter) ? search.jitter : 0;
  const schedulerSeed = Number.isFinite(search?.schedulerSeed) ? search.schedulerSeed : (seed ?? 0);
  const schedulerJitter = Number.isFinite(search?.schedulerJitter) ? search.schedulerJitter : 0;
  const schedulerPolicies = Array.isArray(search?.schedulerPolicies) && search.schedulerPolicies.length
    ? search.schedulerPolicies
    : ['mix'];
  const schedulerRestarts = Number.isFinite(search?.schedulerRestarts) ? search.schedulerRestarts : 1;
  const schedulerLabel = constraintMode === 'parity'
    ? scoreMode
    : (policy === 'priorities' ? 'priority' : (policy === 'mlp' ? 'mlp' : 'heuristic'));

  const rng = createRng(seed ?? 1337);
  const baselinePriorities = buildBaselinePriorities(taskList);
  const bundleSchedule = scheduleFromBundles(taskList, resolvedCaps);
  const bundleEnergy = bundleSchedule
    ? resolveScheduleEnergy(bundleSchedule, taskList.length)
    : Number.POSITIVE_INFINITY;
  const graphScheduleInfo = scheduleGraphWithPolicies(taskList, resolvedCaps, {
    policies: schedulerPolicies,
    restarts: schedulerRestarts,
    seed: schedulerSeed,
    jitter: schedulerJitter,
    dependencyModel,
    engineOrder,
  });
  const graphSchedule = graphScheduleInfo.schedule;
  const baselineSchedule = constraintMode === 'parity'
    ? (Number.isFinite(bundleEnergy) ? bundleSchedule : graphSchedule)
    : scheduleWithPriority(taskList, resolvedCaps, baselinePriorities, graph);
  const baselineEnergy = baselineSchedule
    ? resolveScheduleEnergy(baselineSchedule, taskList.length)
    : Number.POSITIVE_INFINITY;
  let bestEnergy = Number.isFinite(baselineEnergy) ? baselineEnergy : Number.POSITIVE_INFINITY;
  let bestSchedule = Number.isFinite(baselineEnergy) ? baselineSchedule : null;
  let bestHistory = [];
  let bestState = bestSchedule ? bestSchedule.grid : null;
  let bestShape = bestSchedule ? bestSchedule.gridShape : null;
  let bestSteps = 0;
  let totalSteps = 0;
  const candidates = [];
  const graphMetrics = constraintMode === 'relaxed' ? computeGraphMetrics(graph) : null;
  const lbCycles = lowerBoundCycles(taskList, resolvedCaps);
  let mlpTrainSteps = 0;
  let mlpTrainFailures = 0;
  let mlpFirstTrainError = null;

  const totalStart = performance.now();

  if (constraintMode === 'parity') {
    let chosenSchedule = null;
    let chosenCycles = Number.POSITIVE_INFINITY;
    let schedulerPolicy = graphScheduleInfo.policy || null;
    if (scoreMode === 'bundle') {
      chosenSchedule = bundleSchedule || graphSchedule;
      chosenCycles = chosenSchedule?.cycles ?? Number.POSITIVE_INFINITY;
      schedulerPolicy = 'bundle';
    } else if (scoreMode === 'lb') {
      chosenSchedule = bundleSchedule || graphSchedule;
      chosenCycles = lbCycles;
      schedulerPolicy = 'lower-bound';
    } else {
      chosenSchedule = graphSchedule || bundleSchedule;
      chosenCycles = chosenSchedule?.cycles ?? Number.POSITIVE_INFINITY;
    }
    if (!chosenSchedule || !chosenSchedule.grid || !chosenSchedule.gridShape) {
      throw new Error('VLIW parity mode failed to produce a schedule.');
    }
    bestSchedule = chosenSchedule;
    bestState = chosenSchedule.grid;
    bestShape = chosenSchedule.gridShape;
    bestEnergy = Number.isFinite(chosenCycles) ? chosenCycles : resolveScheduleEnergy(chosenSchedule, taskList.length);
    const totalTimeMs = performance.now() - totalStart;
    candidates.push({
      restart: 1,
      cycles: chosenCycles,
      utilization: chosenSchedule.utilization,
      violations: chosenSchedule.violations,
      steps: 0,
    });
    return {
      steps: 0,
      stepsPerRestart: 0,
      bestStep: 0,
      restarts: 1,
      energy: bestEnergy,
      energyHistory: [],
      state: bestState,
      shape: bestShape,
      metrics: {
        cycles: chosenCycles,
        utilization: chosenSchedule.utilization,
        violations: chosenSchedule.violations,
      },
      baseline: {
        cycles: baselineSchedule?.cycles ?? 0,
        utilization: baselineSchedule?.utilization ?? 0,
        violations: baselineSchedule?.violations ?? 0,
        scheduled: baselineSchedule?.scheduled ?? 0,
        energy: baselineEnergy,
      },
      schedule: {
        slotAssignments: chosenSchedule.slotAssignments,
        slotEngines: chosenSchedule.slotEngines,
        slotIndices: chosenSchedule.slotIndices,
        duplicates: chosenSchedule.duplicates ?? 0,
        missing: chosenSchedule.missing ?? 0,
      },
      candidates,
      taskMeta,
      totalTimeMs,
      scheduler: schedulerLabel,
      schedulerPolicy,
      schedulerPolicies,
      scoreMode,
      engineOrder,
      capsSource,
      mode: constraintMode,
    };
  }

  for (let restart = 0; restart < restarts; restart++) {
    const seedValue = Math.floor(rng() * 1e9);
    const priorities = initMode === 'baseline'
      ? new Float32Array(baselinePriorities)
      : initPriorities(taskList.length, initMode, seedValue, initScale);
    const weights = initWeights(initMode, seedValue, initScale, DEFAULT_WEIGHTS);
    let current = policy === 'priorities'
      ? priorities
      : (policy === 'mlp' ? createMlp(mlpInputSize, mlpHiddenSize, seedValue) : weights);

    let trainer = null;

    let currentSchedule = policy === 'priorities'
      ? scheduleWithPriority(taskList, resolvedCaps, current, graph)
      : scheduleWithHeuristic({
        tasks: taskList,
        caps: resolvedCaps,
        graph,
        features: graphMetrics,
        weights: policy === 'weights' ? current : weights,
        ...(policy === 'mlp' ? { mlpModel: current } : {}),
        basePriorities: baselinePriorities,
        rng,
        jitter,
      });
    let currentEnergy = resolveScheduleEnergy(currentSchedule, taskList.length);
    if (!Number.isFinite(currentEnergy) && policy === 'mlp') {
      const retrySeed = Math.floor(rng() * 1e9);
      const retryMlp = createMlp(mlpInputSize, mlpHiddenSize, retrySeed);
      const retrySchedule = scheduleWithHeuristic({
        tasks: taskList,
        caps: resolvedCaps,
        graph,
        features: graphMetrics,
        weights,
        mlpModel: retryMlp,
        basePriorities: baselinePriorities,
        rng,
        jitter,
      });
      const retryEnergy = resolveScheduleEnergy(retrySchedule, taskList.length);
      if (Number.isFinite(retryEnergy)) {
        current = retryMlp;
        currentSchedule = retrySchedule;
        currentEnergy = retryEnergy;
      }
    }
    let temperature = tempStart;
    const energyHistory = [];
    let restartBestEnergy = currentEnergy;
    let restartBestSchedule = currentSchedule;
    let restartBestSteps = 1;
    let stepsRun = 0;
    let disableMlpTraining = false;

    if (onTrace) {
      onTrace(0, currentEnergy, {
        cycles: currentSchedule.cycles,
        utilization: currentSchedule.utilization,
      });
    }

    try {
      for (let step = 0; step < maxSteps; step++) {
        stepsRun = step + 1;

        const scoredFeatures = policy === 'mlp' ? [] : null;
        const candidate = policy === 'priorities'
          ? perturbPriorities(current, rng, mutationCount, stepSize)
          : (policy === 'mlp' ? perturbMlp(current, rng, mutationCount, stepSize) : perturbWeights(current, rng, mutationCount, stepSize));

        const candidateSchedule = policy === 'priorities'
          ? scheduleWithPriority(taskList, resolvedCaps, candidate, graph)
          : scheduleWithHeuristic({
            tasks: taskList,
            caps: resolvedCaps,
            graph,
            features: graphMetrics,
            weights: policy === 'weights' ? candidate : weights,
            ...(policy === 'mlp' ? { mlpModel: candidate, scoredFeatures } : {}),
            basePriorities: baselinePriorities,
            rng,
            jitter,
          });

        const candidateEnergy = resolveScheduleEnergy(candidateSchedule, taskList.length);
        const delta = candidateEnergy - currentEnergy;
        const accept = (!Number.isFinite(delta) && candidateEnergy < currentEnergy)
          || delta <= 0
          || rng() < Math.exp(-delta / Math.max(temperature, 1e-6));

        if (accept) {
          if (policy === 'mlp' && !disableMlpTraining && scoredFeatures && scoredFeatures.length) {
            const featureBatch = new Float32Array(scoredFeatures);
            const canTrain = featureBatch.length >= mlpInputSize && featureBatch.length % mlpInputSize === 0;
            if (canTrain) {
              try {
                if (!trainer) {
                  trainer = createMlpTrainer(mlpInputSize, mlpHiddenSize, mlpTrainerConfig);
                }
                current = await mlpTrainDistillStep(trainer, current, featureBatch, candidate);
                currentSchedule = scheduleWithHeuristic({
                  tasks: taskList,
                  caps: resolvedCaps,
                  graph,
                  features: graphMetrics,
                  weights,
                  mlpModel: current,
                  basePriorities: baselinePriorities,
                  rng,
                  jitter,
                });
                currentEnergy = resolveScheduleEnergy(currentSchedule, taskList.length);
                mlpTrainSteps += 1;
              } catch (error) {
                mlpTrainFailures += 1;
                if (!mlpFirstTrainError) {
                  mlpFirstTrainError = error?.message || String(error);
                }
                disableMlpTraining = true;
                current = candidate;
                currentSchedule = candidateSchedule;
                currentEnergy = candidateEnergy;
              }
            } else {
              current = candidate;
              currentSchedule = candidateSchedule;
              currentEnergy = candidateEnergy;
            }
          } else {
            current = candidate;
            currentSchedule = candidateSchedule;
            currentEnergy = candidateEnergy;
          }
        }

        let observedEnergy = currentEnergy;
        let observedSchedule = currentSchedule;
        if (policy === 'mlp' && candidateEnergy < observedEnergy) {
          observedEnergy = candidateEnergy;
          observedSchedule = candidateSchedule;
        }

        if (observedEnergy < restartBestEnergy) {
          restartBestEnergy = observedEnergy;
          restartBestSchedule = observedSchedule;
          restartBestSteps = step + 1;
        }
        if (observedEnergy < bestEnergy) {
          bestEnergy = observedEnergy;
          bestSchedule = observedSchedule;
          bestState = observedSchedule.grid;
          bestShape = observedSchedule.gridShape;
          bestSteps = step + 1;
          if (energyHistory.length) {
            bestHistory = energyHistory.slice();
          }
        }

        if (step % readbackEvery === 0 || step === maxSteps - 1) {
          energyHistory.push(currentEnergy);
          if (energyHistory.length > historyLimit) {
            energyHistory.shift();
          }
        }
        if (onProgress) {
          onProgress({
            stage: 'energy',
            percent: (step + 1) / maxSteps,
            message: `VLIW search ${restart + 1}/${restarts} • step ${step + 1}/${maxSteps}`,
          });
        }
        if (convergenceThreshold != null && step >= minSteps && currentEnergy <= convergenceThreshold) {
          break;
        }
        temperature *= tempDecay;
      }
    } finally {
      if (trainer) {
        disposeMlpTrainer(trainer);
      }
    }
    totalSteps += stepsRun;

    candidates.push({
      restart: restart + 1,
      cycles: restartBestEnergy,
      utilization: restartBestSchedule.utilization,
      violations: restartBestSchedule.violations,
      steps: restartBestSteps,
    });
  }

  const totalTimeMs = performance.now() - totalStart;
  if (!bestSchedule || !bestState || !bestShape) {
    if (baselineSchedule?.grid && baselineSchedule?.gridShape) {
      bestSchedule = baselineSchedule;
      bestState = baselineSchedule.grid;
      bestShape = baselineSchedule.gridShape;
      bestEnergy = resolveScheduleEnergy(bestSchedule, taskList.length);
    } else {
      throw new Error('VLIW search failed to produce a schedule.');
    }
  }

  return {
    steps: totalSteps,
    stepsPerRestart: maxSteps,
    bestStep: bestSteps,
    restarts,
    energy: bestEnergy,
    energyHistory: bestHistory,
    state: bestState,
    shape: bestShape,
    metrics: {
      cycles: bestSchedule.cycles,
      utilization: bestSchedule.utilization,
      violations: bestSchedule.violations,
    },
    baseline: {
      cycles: baselineSchedule.cycles,
      utilization: baselineSchedule.utilization,
      violations: baselineSchedule.violations,
      scheduled: baselineSchedule.scheduled,
      energy: baselineEnergy,
    },
    schedule: {
      slotAssignments: bestSchedule.slotAssignments,
      slotEngines: bestSchedule.slotEngines,
      slotIndices: bestSchedule.slotIndices,
      duplicates: bestSchedule.duplicates ?? 0,
      missing: bestSchedule.missing ?? 0,
    },
    candidates,
    taskMeta,
    totalTimeMs,
    scheduler: schedulerLabel,
    schedulerPolicy: policy,
    schedulerPolicies,
    scoreMode,
    engineOrder,
    capsSource,
    mode: constraintMode,
    ...(policy === 'mlp'
      ? {
        mlpStats: {
          hiddenSize: mlpHiddenSize,
          lr: mlpTrainerConfig?.lr ?? DEFAULT_MLP_CONFIG.lr,
          trainSteps: mlpTrainSteps,
          trainFailures: mlpTrainFailures,
          firstError: mlpFirstTrainError,
        },
      }
      : {}),
  };
}
