export const HASH_STAGES = [
  ['+', 0x7ED55D16, '+', '<<', 12],
  ['^', 0xC761C23C, '^', '>>', 19],
  ['+', 0x165667B1, '+', '<<', 5],
  ['+', 0xD3A2646C, '^', '<<', 9],
  ['+', 0xFD7046C5, '+', '<<', 3],
  ['^', 0xB55A4F09, '^', '>>', 16],
];

export function splitHashStages() {
  const linear = [];
  const bitwise = [];
  HASH_STAGES.forEach(([op1, val1, op2, op3, val3]) => {
    if (op1 === '+' && op2 === '+') {
      const mult = (1 + (1 << val3)) % (2 ** 32);
      linear.push({ mult, add: val1 });
    } else {
      bitwise.push({ op1, const: val1, op2, shift_op: op3, shift: val3 });
    }
  });
  return { linear, bitwise };
}
