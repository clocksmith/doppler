const TEXT_ENCODER = new TextEncoder();

export function encodeUtf8(value) {
  return TEXT_ENCODER.encode(String(value ?? ''));
}
