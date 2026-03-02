export function $(id) {
  return document.getElementById(id);
}

export function setText(el, text) {
  if (!el) return;
  el.textContent = text;
}

export function setHidden(el, hidden) {
  if (el) el.hidden = hidden;
}
