// Where a drawn value came from. Every mark in every view carries one of these
// kinds, so a projection or an estimate cannot be read as a measurement. A mark
// without a kind is a programming error and throws.

export const PROVENANCE = {
  measured: { label: 'Measured', what: 'Read from this run as it executed.' },
  derived: { label: 'Derived', what: 'Computed from measured values of this run, by a stated rule.' },
  projected: { label: 'Projected', what: 'A projection of measured values through another part of the model. An estimate, not something the model produced at that point.' },
  reconstructed: { label: 'Reconstructed', what: 'Rebuilt from other captured values; not captured directly.' },
  estimated: { label: 'Estimated', what: 'Not run. An estimate from a stated model of the cost.' },
  unrun: { label: 'Not run', what: 'Possible, but not run. What would follow is unknown until it is.' },
  missing: { label: 'Not recorded', what: 'This run did not record it. Nothing is shown in its place.' },
};

export function describeProvenance(kind) {
  const entry = PROVENANCE[kind];
  if (!entry) throw new Error(`Every mark needs a provenance kind; got ${kind === undefined ? 'none' : `"${kind}"`}. Kinds: ${Object.keys(PROVENANCE).join(', ')}.`);
  return { kind, ...entry };
}

// RD's badge, as the site's pill. site.css draws the border by kind: solid for
// measured and derived, dotted for projected, reconstructed and estimated,
// dashed for not run and not recorded.
export function provenanceMark(kind, detail = '') {
  const { label, what } = describeProvenance(kind);
  const mark = document.createElement('span');
  mark.className = 'badge';
  mark.dataset.provenance = kind;
  mark.title = detail ? `${what} ${detail}` : what;
  mark.textContent = label;
  return mark;
}
