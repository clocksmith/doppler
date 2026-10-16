enable f16;
enable chromium_experimental_subgroup_matrix;
struct Uniforms { M:u32, N:u32, K:u32, alpha:f32, transpose_b:u32, p0:u32, p1:u32, p2:u32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> A:array<f32>;
@group(0) @binding(2) var<storage,read> B:array<f16>;
@group(0) @binding(3) var<storage,read_write> C:array<f32>;
var<workgroup> staged_b:array<f32,64>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group:vec3u, @builtin(local_invocation_index) tid:u32) {
  let row=group.x*8u;
  let col=group.y*8u;
  var acc=subgroup_matrix_result<f32,8,8>(0.0);
  for(var k=0u;k<u.K;k+=8u) {
    for(var i=tid;i<64u;i+=32u) {
      staged_b[i]=f32(B[(col+i/8u)*u.K+k+i%8u]);
    }
    workgroupBarrier();
    let a=subgroupMatrixLoad<subgroup_matrix_left<f32,8,8>>(&A,row*u.K+k,false,u.K);
    let b=subgroupMatrixLoad<subgroup_matrix_right<f32,8,8>>(&staged_b,0u,true,8u);
    acc=subgroupMatrixMultiplyAccumulate(a,b,acc);
    workgroupBarrier();
  }
  subgroupMatrixStore(&C,row*u.N+col,acc,false,u.N);
}
