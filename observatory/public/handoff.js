// Passing one thing from a page of this site to the next, in this tab only.
// Session storage can be missing or full; every use is optional and the pages
// work without it.

const key = (kind) => `doppler.observatory.handoff.${kind}`;

export function give(kind, value) {
  try { sessionStorage.setItem(key(kind), JSON.stringify(value)); return true; } catch { return false; }
}

// Reads what was handed over and clears it, so a reload does not repeat it.
export function take(kind) {
  try {
    const text = sessionStorage.getItem(key(kind));
    if (text === null) return null;
    sessionStorage.removeItem(key(kind));
    return JSON.parse(text);
  } catch { return null; }
}
