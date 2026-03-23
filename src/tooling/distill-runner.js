import { log } from '../debug/index.js';

/**
 * Validate that teacher and student model configs are compatible for distillation.
 * Checks hidden size and vocab size match (required for knowledge distillation).
 *
 * @param {object} options
 * @param {object} options.teacherConfig - Teacher model configuration.
 * @param {object} options.studentConfig - Student model configuration.
 * @param {string} [options.label] - Optional label for log messages.
 * @returns {{ compatible: boolean, warnings: string[] }}
 */
export function validateDistillationCompatibility(options = {}) {
  const { teacherConfig, studentConfig, label } = options;
  const warnings = [];
  const prefix = label ? `${label}: ` : '';

  if (!teacherConfig || typeof teacherConfig !== 'object') {
    warnings.push(`${prefix}Teacher config is missing or invalid.`);
    log.warn('distill-runner', warnings[warnings.length - 1]);
    return { compatible: false, warnings };
  }

  if (!studentConfig || typeof studentConfig !== 'object') {
    warnings.push(`${prefix}Student config is missing or invalid.`);
    log.warn('distill-runner', warnings[warnings.length - 1]);
    return { compatible: false, warnings };
  }

  const teacherVocab = teacherConfig.vocabSize ?? teacherConfig.vocab_size ?? null;
  const studentVocab = studentConfig.vocabSize ?? studentConfig.vocab_size ?? null;
  if (teacherVocab != null && studentVocab != null && teacherVocab !== studentVocab) {
    const msg =
      `${prefix}Vocab size mismatch: teacher=${teacherVocab}, student=${studentVocab}. ` +
      'Distillation requires matching vocab sizes unless a projection layer is configured.';
    warnings.push(msg);
    log.warn('distill-runner', msg);
  }

  const teacherHidden = teacherConfig.hiddenSize ?? teacherConfig.hidden_size ?? null;
  const studentHidden = studentConfig.hiddenSize ?? studentConfig.hidden_size ?? null;
  if (teacherHidden != null && studentHidden != null && teacherHidden !== studentHidden) {
    const msg =
      `${prefix}Hidden size mismatch: teacher=${teacherHidden}, student=${studentHidden}. ` +
      'A projection layer is required when hidden sizes differ.';
    warnings.push(msg);
    log.warn('distill-runner', msg);
  }

  const compatible = warnings.length === 0;
  if (compatible) {
    log.debug('distill-runner', `${prefix}Teacher/student configs are compatible.`);
  }

  return { compatible, warnings };
}
