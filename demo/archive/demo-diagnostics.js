async function handleDiagnosticsRun(mode) {
  const profileSelect = $('diagnostics-profile');
  const modelSelect = $('diagnostics-model');
  const runtimeProfileSelect = $('runtime-profile');
  const selections = state.diagnosticsSelections[state.uiMode] || {};
  const selectedProfileId = profileSelect?.value || selections.profile || '';
  const selectedProfile = decodeDiagnosticsProfileId(selectedProfileId);
  const workload = selectedProfile?.suite || selections.suite || getDiagnosticsDefaultSuite(state.uiMode);
  const modelId = modelSelect?.value || null;
  const runtimeProfile = selectedProfile?.runtimeProfile || selections.runtimeProfile || runtimeProfileSelect?.value || DEFAULT_RUNTIME_PROFILE;
  if (selectedProfile) {
    storeDiagnosticsSelection(state.uiMode, {
      profile: selectedProfileId,
      suite: selectedProfile.suite,
      runtimeProfile: selectedProfile.runtimeProfile,
    });
  }
  if (runtimeProfileSelect && runtimeProfileSelect.value !== runtimeProfile) {
    runtimeProfileSelect.value = runtimeProfile;
  }
  if (profileSelect && selectedProfileId && profileSelect.value !== selectedProfileId) {
    profileSelect.value = selectedProfileId;
  }
  const captureOutput = runtimeProfile === 'profiles/verbose-trace';
  const previousRuntime = cloneRuntimeConfig(getRuntimeConfig());
  let runtimeConfig = state.diagnosticsRuntimeConfig;

  updateDiagnosticsStatus(`${mode === 'verify' ? 'Verifying' : 'Running'} ${workload}...`);
  updateDiagnosticsReport('');
  clearDiagnosticsOutput();
  try {
    if (!runtimeConfig || state.diagnosticsRuntimeProfileId !== runtimeProfile) {
      runtimeConfig = await refreshDiagnosticsRuntimeConfig(runtimeProfile);
    }
    if (mode === 'verify') {
      const result = await controller.verifySuite(
        modelId ? { sources: { browser: { id: modelId } } } : null,
        {
          workload,
          runtimeProfile,
          modelId,
          runtimeConfig,
        }
      );
      state.lastReport = result.report;
      state.lastReportInfo = result.reportInfo ?? null;
      state.lastMetrics = result.metrics ?? null;
      state.lastDiagnosticsSuite = result.suite ?? workload;
      updateDiagnosticsStatus('Verified');
      updateDiagnosticsReport(result.report?.timestamp || new Date().toISOString());
      renderDiagnosticsOutput({ suite: workload, modelId, report: result.report }, workload, false);
      return;
    }

    if (state.activePipeline) {
      await unloadActivePipeline();
    }

    const options = {
      workload,
      runtimeProfile,
      modelId,
      runtimeConfig,
      captureOutput,
    };
    const result = await controller.runSuite(
      modelId ? { sources: { browser: { id: modelId } } } : null,
      { ...options, keepPipeline: true }
    );
    state.lastReport = result.report;
    state.lastReportInfo = result.reportInfo;
    state.lastMetrics = result.metrics ?? null;
    state.lastDiagnosticsSuite = result.suite;
    if (result.memoryStats) {
      state.lastMemoryStats = result.memoryStats;
    }
    if (result.pipeline !== undefined) {
      state.activePipeline = result.pipeline;
    }
    state.activeModelId = modelId || null;
    state.lastInferenceStats = result.pipeline?.getStats?.() ?? state.lastInferenceStats;
    if (state.lastInferenceStats) {
      state.runCounter += 1;
      recordRunLog(state.lastInferenceStats, `#${state.runCounter}`);
    }
    if (result.suite === 'diffusion' && result.metrics) {
      state.lastDiffusionRequest = {
        width: result.metrics.width,
        height: result.metrics.height,
        steps: result.metrics.steps,
      };
    }
    if (result.suite === 'energy' && result.metrics) {
      const shape = Array.isArray(result.metrics.shape) ? result.metrics.shape : null;
      if (shape) {
        state.lastEnergyRequest = {
          shape,
          height: shape[0],
          width: shape[1],
          channels: shape[2],
        };
      }
      if (Array.isArray(result.metrics.energyHistory)) {
        drawEnergyChart(result.metrics.energyHistory);
      }
      updateEnergyStats({
        steps: result.metrics.steps,
        energy: result.metrics.energy,
        dtype: result.metrics.dtype,
        shape,
        stateStats: result.metrics.stateStats,
      });
    }
    updateDiagnosticsStatus(`Complete (${result.suite || workload})`);
    if (result.reportInfo?.path) {
      updateDiagnosticsReport(result.reportInfo.path);
    } else if (result.report?.timestamp) {
      updateDiagnosticsReport(result.report.timestamp);
    }
    const snapshot = captureMemorySnapshot();
    updateMemoryPanel(snapshot);
    updatePerformancePanel(snapshot);
    updateMemoryControls();
    renderDiagnosticsOutput(result, workload, captureOutput);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    updateDiagnosticsStatus(message, true);
    const timestamp = new Date().toISOString();
    const report = {
      suite: workload,
      modelId,
      runtimeProfile,
      timestamp,
      results: [{ name: mode === 'verify' ? 'verify-config' : 'run', passed: false, error: message }],
      metrics: { error: true, mode },
      output: { error: message },
    };
    state.lastReport = report;
    state.lastReportInfo = null;
    state.lastMetrics = report.metrics;
    state.lastDiagnosticsSuite = workload;
    updateDiagnosticsReport(timestamp);
    renderDiagnosticsOutput({ suite: workload, modelId, report }, workload, captureOutput);
  } finally {
    setRuntimeConfig(previousRuntime);
    updateRunAutoLabels();
  }
}

function exportDiagnosticsReport() {
  if (!state.lastReport) {
    updateDiagnosticsStatus('No report available to export', true);
    return;
  }
  const timestamp = state.lastReport.timestamp || new Date().toISOString();
  const safeTimestamp = timestamp.replace(/[:]/g, '-');
  const filename = `doppler-report-${safeTimestamp}.json`;
  const blob = new Blob([JSON.stringify(state.lastReport, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function serializeTypedArray(value) {
  if (!value) return null;
  if (ArrayBuffer.isView(value)) return Array.from(value);
  return value;
}

function serializeSchedule(schedule) {
  if (!schedule) return null;
  return {
    slotAssignments: serializeTypedArray(schedule.slotAssignments),
    slotEngines: Array.isArray(schedule.slotEngines) ? schedule.slotEngines.slice() : schedule.slotEngines,
    slotIndices: Array.isArray(schedule.slotIndices) ? schedule.slotIndices.slice() : schedule.slotIndices,
  };
}

function serializeOps(ops) {
  if (!Array.isArray(ops)) return null;
  return ops.map((op) => ({
    id: op?.id ?? null,
    engine: op?.engine ?? null,
    slot: Array.isArray(op?.slot) ? op.slot.slice() : op?.slot ?? null,
    offloadable: !!op?.offloadable,
    meta: op?.meta ?? null,
  }));
}

function exportEnergyRun() {
  if (!state.lastEnergyResult) {
    updateEnergyStatus('No energy run available to export.');
    return;
  }
  const payload = {
    timestamp: new Date().toISOString(),
    problem: 'quintel',
    result: {
      backend: state.lastEnergyResult.backend ?? null,
      dtype: state.lastEnergyResult.dtype ?? null,
      shape: state.lastEnergyResult.shape ?? null,
      steps: state.lastEnergyResult.steps ?? null,
      energy: state.lastEnergyResult.energy ?? null,
      metrics: state.lastEnergyResult.metrics ?? null,
      baseline: state.lastEnergyResult.baseline ?? null,
      candidates: state.lastEnergyResult.candidates ?? null,
      energyHistory: state.lastEnergyResult.energyHistory ?? null,
      schedule: serializeSchedule(state.lastEnergyResult.schedule),
    },
  };
  const filename = `doppler-energy-export-${payload.timestamp.replace(/[:]/g, '-')}.json`;
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
// --- Exports ---
export {
  exportDiagnosticsReport,
  serializeTypedArray,
  serializeSchedule,
  serializeOps,
  exportEnergyRun,
};
