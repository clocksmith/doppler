


function silu(x) {
  return x / (1 + Math.exp(-x));
}


export function siluRef(input) {
  const output = new Float32Array(input.length);

  for (let i = 0; i < input.length; i++) {
    output[i] = silu(input[i]);
  }

  return output;
}


export function siluGatedRef(gate, up) {
  const output = new Float32Array(gate.length);

  for (let i = 0; i < gate.length; i++) {
    output[i] = silu(gate[i]) * up[i];
  }

  return output;
}


export function siluFusedRef(input) {
  const halfSize = input.length / 2;
  const output = new Float32Array(halfSize);

  for (let i = 0; i < halfSize; i++) {
    const gateVal = input[i];
    const upVal = input[halfSize + i];
    output[i] = silu(gateVal) * upVal;
  }

  return output;
}


export function siluInplaceRef(input) {
  for (let i = 0; i < input.length; i++) {
    input[i] = silu(input[i]);
  }
  return input;
}

export default siluRef;
