import {
  extractExecutionContractFacts,
  sanitizeLeanModuleName,
} from '../config/execution-contract-check.js';

export {
  extractExecutionContractFacts,
  sanitizeLeanModuleName,
} from '../config/execution-contract-check.js';

function asLeanString(value) {
  return JSON.stringify(String(value));
}

function renderExecutionStep(step) {
  return [
    '  {',
    `    id := ${asLeanString(step.id)},`,
    `    phase := .${step.phase},`,
    `    opClass := .${step.opClass},`,
    '  }',
  ].join('\n');
}

function renderCheckName(modelId, suffix) {
  return `${modelId}.${suffix}`;
}

export function renderExecutionContractLeanModule(facts, options = {}) {
  const moduleName = sanitizeLeanModuleName(options.moduleName ?? facts?.modelId ?? 'GeneratedExecutionContractCheck');
  const modelId = String(facts?.modelId ?? 'model').trim() || 'model';
  const session = facts?.session;
  const steps = Array.isArray(facts?.steps) ? facts.steps : [];
  if (!session || typeof session !== 'object') {
    throw new Error('lean execution contract: facts.session is required.');
  }

  const renderedSteps = steps.length > 0
    ? steps.map(renderExecutionStep).join(',\n')
    : '';
  const stepsLiteral = steps.length > 0
    ? `[\n${renderedSteps}\n]`
    : '[]';

  return [
    'import Doppler.ExecutionContract',
    '',
    `def extractedModelId : String := ${asLeanString(modelId)}`,
    '',
    'def extractedSession : SessionConfig := {',
    `  layout := .${session.layout},`,
    `  disableCommandBatching := ${session.disableCommandBatching ? 'true' : 'false'},`,
    `  decodeBatchSize := ${session.decodeBatchSize},`,
    `  headDim := ${session.headDim},`,
    `  kvLen := ${session.kvLen},`,
    `  coldQuantMode := .${session.coldQuantMode},`,
    '}',
    '',
    `def extractedSteps : List ExecutionStep := ${stepsLiteral}`,
    '',
    'def executionContractChecks : List (String × Bool) := [',
    `  (${asLeanString(renderCheckName(modelId, 'steps'))}, allStepsCompatible extractedSteps extractedSession),`,
    `  (${asLeanString(renderCheckName(modelId, 'session'))}, sessionConsistent extractedSession)`,
    ']',
    '',
    'def renderCheck (entry : String × Bool) : String :=',
    '  let status := if entry.2 then "pass" else "fail"',
    '  s!"{entry.1}: {status}"',
    '',
    'def renderedChecks : List String :=',
    '  executionContractChecks.map renderCheck',
    '',
    'def executionContractOverall : Bool :=',
    '  executionContractChecks.all (fun entry => entry.2)',
    '',
    '#eval s!"executionContractModule:' + moduleName + '"',
    '#eval s!"executionContractOverall:{if executionContractOverall then "pass" else "fail"}"',
    '#eval renderedChecks',
    '',
  ].join('\n');
}
