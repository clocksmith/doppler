export function readOptionalNumber(el, { integer = false } = {}) {
  const raw = el?.value;
  if (raw === '' || raw == null) return undefined;
  const parsed = integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
  return Number.isFinite(parsed) ? parsed : undefined;
}
