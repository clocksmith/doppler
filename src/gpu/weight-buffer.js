


export function createWeightBuffer(
  buffer,
  dtype,
  layout,
  shape,
  label
) {
  return {
    buffer,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}


export function createCpuWeightBuffer(
  data,
  dtype,
  layout,
  shape,
  label
) {
  return {
    data,
    dtype,
    layout,
    shape: Object.freeze([...shape]),
    label,
  };
}


export function isColumnMajor(weight) {
  return weight.layout === 'column';
}


export function isWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'buffer' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}


export function isCpuWeightBuffer(value) {
  return (
    typeof value === 'object' &&
    value !== null &&
    'data' in value &&
    'dtype' in value &&
    'layout' in value &&
    'shape' in value
  );
}


export function getBuffer(weight) {
  return isWeightBuffer(weight) ? weight.buffer : weight;
}


export function getLayout(weight) {
  return isWeightBuffer(weight) ? weight.layout : null;
}


export function getWeightDtype(weight) {
  return isWeightBuffer(weight) ? weight.dtype : null;
}
