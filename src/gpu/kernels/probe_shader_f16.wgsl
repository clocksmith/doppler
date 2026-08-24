enable f16;

@compute @workgroup_size(1)
fn main() {
  var value: f16 = 1.0h;
  _ = value;
}
