import { buildRuleFileStructureArtifact } from './rule-matcher.js';

// Structural contract for inference/layer-pattern.rules.json. Context keys
// mirror the selection contexts built in inference/pipelines/text/config.js;
// rule values are owned only by the JSON file.

const isNonEmptyString = (value) => typeof value === 'string' && value.length > 0;

const LAYER_PATTERN_RULE_SPECS = Object.freeze({
  patternKind: {
    contextKeys: Object.freeze(['patternType', 'globalPattern']),
    isValue: (value) => value === null || isNonEmptyString(value),
    valueType: 'null or a non-empty string',
    catchAll: true,
  },
  layerType: {
    contextKeys: Object.freeze(['patternKind', 'isEven', 'isStride']),
    isValue: isNonEmptyString,
    valueType: 'a non-empty string',
    catchAll: false,
  },
});

// layerType has no global catch-all, so every pattern kind that patternKind
// can produce needs its own terminal rule, and layerType may only reference
// produced kinds.
function checkLayerTypeCoverage(ruleFile) {
  const patternKindRules = Array.isArray(ruleFile?.patternKind) ? ruleFile.patternKind : [];
  const layerTypeRules = Array.isArray(ruleFile?.layerType) ? ruleFile.layerType : [];
  const producedKinds = new Set(
    patternKindRules.map((rule) => rule?.value).filter(isNonEmptyString)
  );
  const errors = [];
  for (const [index, rule] of layerTypeRules.entries()) {
    const kind = rule?.match?.patternKind;
    if (!producedKinds.has(kind)) {
      errors.push(`layerType[${index}].match.patternKind must be a kind produced by patternKind; got ${JSON.stringify(kind)}.`);
    }
  }
  for (const kind of producedKinds) {
    const hasTerminal = layerTypeRules.some((rule) => {
      const keys = Object.keys(rule?.match ?? {});
      return keys.length === 1 && rule.match.patternKind === kind;
    });
    if (!hasTerminal) {
      errors.push(`layerType must end pattern kind "${kind}" with a { "patternKind": "${kind}" } rule.`);
    }
  }
  return errors;
}

export function buildLayerPatternContractArtifact(ruleGroup) {
  return buildRuleFileStructureArtifact(ruleGroup, LAYER_PATTERN_RULE_SPECS, {
    idPrefix: 'inference.layerPattern',
    label: 'LayerPatternContract',
    extraChecks: [{ id: 'layerType.coverage', run: checkLayerTypeCoverage }],
  });
}
