const PROGRAM_LENGTH = 47u;
const P = array<u32, PROGRAM_LENGTH>(21331,0,44,1,2,7,0,1,3,7,0,8,4,2,3,16,4,0,0,1,5,89,0,17,0,0,0,1,5,78,0,18,0,0,0,12,5,0,0,1,5,10,0,12,5,0,0);
override TARGET: u32 = 1u;
override EXTERNAL: u32 = 0u;
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> supplied: array<u32>;
var<private> cursor: u32 = 1u;
var<private> bad: bool = false;
fn put(c: u32) {
  if (c > 255u || cursor >= arrayLength(&output)) { bad = true; return; }
  output[cursor] = c; cursor += 1u;
}
fn decimal(n: u32) {
  var divisor = 1u;
  while (n / divisor >= 10u) { divisor *= 10u; }
  var value = n;
  loop {
    put(48u + value / divisor); value %= divisor;
    if (divisor == 1u) { break; }
    divisor /= 10u;
  }
}
fn data_at(i: u32) -> u32 { if (i >= PROGRAM_LENGTH) { return 0u; } return P[i]; }
fn input_length() -> u32 { if (EXTERNAL == 0u) { return PROGRAM_LENGTH; } return supplied[0]; }
fn input_at(i: u32) -> u32 {
  if (EXTERNAL == 0u) { return data_at(i); }
  if (i >= supplied[0] || i + 1u >= arrayLength(&supplied)) { return 0u; }
  return supplied[i + 1u];
}
fn safe_div(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a / b; }
fn safe_mod(a: u32, b: u32) -> u32 { if (b == 0u) { return 0u; } return a % b; }
fn valid_input() -> bool {
  let n = input_length();
  if (TARGET > 2u || n < 3u || n > 60000u || input_at(0u) != 21331u) { return false; }
  if (EXTERNAL != 0u && n >= arrayLength(&supplied)) { return false; }
  let data_length = input_at(1u); let code_length = input_at(2u);
  if (data_length > n || code_length > n || code_length % 4u != 0u || 3u + data_length + code_length != n) { return false; }
  var stack: array<u32, 64>;
  var depth = 0u;
  for (var pc = 3u + data_length; pc < n; pc += 4u) {
    let op = input_at(pc); let a = input_at(pc+1u); let b = input_at(pc+2u); let c = input_at(pc+3u);
    if (op > 18u || a >= 64u || (op >= 2u && op <= 11u && b >= 64u) || (op >= 3u && op <= 9u && c >= 64u)) { return false; }
    if (op == 14u || op == 16u) {
      if (depth == 64u) { return false; }
      stack[depth] = op; depth += 1u;
    } else if (op == 15u) {
      if (depth == 0u) { return false; } depth -= 1u;
      if (stack[depth] != 14u) { return false; }
    } else if (op == 17u) {
      if (depth == 0u || stack[depth-1u] != 16u) { return false; }
      stack[depth-1u] = 17u;
    } else if (op == 18u) {
      if (depth == 0u) { return false; } depth -= 1u;
      if (stack[depth] != 16u && stack[depth] != 17u) { return false; }
    }
  }
  return depth == 0u;
}
@compute @workgroup_size(1)
fn main() {
  if (!valid_input()) { output[0] = 0u; return; }
  var r: array<u32, 64>;
  r[0] = TARGET; r[1] = input_length();
r[2]=7u;
r[3]=7u;
r[4]=select(0u,1u,r[2]==r[3]);
if(r[4]!=0u){
r[5]=89u;
}else{
r[5]=78u;
}
put(r[5]);
r[5]=10u;
put(r[5]);
  output[0] = select(cursor - 1u, 0u, bad);
}
