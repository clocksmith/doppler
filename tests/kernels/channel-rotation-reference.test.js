import assert from 'node:assert/strict';
import { rotationReference, applyRotationReference } from './reference/channel-rotation.js';
const options = { channels: 3, start: 1, stop: 2, contentStart: 1, contentStop: 3, epsilon: 1e-6 };
const source = [999, 999, 999, 1, 2, -1, 3, -2, 4, -999, -999, -999];
const rotation = rotationReference(source, options);
assert.equal(rotation.status, 1);
const input = [2, -3, 7, 0, 4, -2];
const near = (a, b) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-11));
for (const theta of [0, 0.05, -0.05, 0.5]) {
  const changed = applyRotationReference(input, rotation, { channels: 3, theta });
  const restored = applyRotationReference(changed, rotation, { channels: 3, theta: -theta });
  near(restored, input);
  for (let i = 0; i < input.length; i += 3) near([Math.hypot(...changed.slice(i, i + 3))], [Math.hypot(...input.slice(i, i + 3))]);
}
assert.deepEqual([...applyRotationReference(input, rotation, { channels: 3, theta: 0 })], input);
assert.equal(rotationReference([0,0,0,1,2,3], { ...options, start: 0, stop: 1, contentStart: 0, contentStop: 2 }).status, 0);
assert.equal(rotationReference([1,2,3,2,4,6], { ...options, start: 0, stop: 1, contentStart: 0, contentStop: 2 }).status, 0);
assert.equal(rotationReference([NaN,2,3,2,4,6], { ...options, start: 0, stop: 1, contentStart: 0, contentStop: 2 }).status, -1);
console.log('channel-rotation CPU identity, inverse, norm, degenerate and invalid checks passed');
