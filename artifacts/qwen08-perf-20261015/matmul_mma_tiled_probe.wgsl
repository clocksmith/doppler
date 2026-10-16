enable f16;
enable chromium_experimental_subgroup_matrix;
struct Uniforms { M:u32, N:u32, K:u32, alpha:f32, transpose_b:u32, p0:u32, p1:u32, p2:u32 }
@group(0) @binding(0) var<uniform> u:Uniforms;
@group(0) @binding(1) var<storage,read> A:array<f32>;
@group(0) @binding(2) var<storage,read> B:array<f16>;
@group(0) @binding(3) var<storage,read_write> C:array<f32>;
var<workgroup> staged_a:array<f32,512>;
var<workgroup> staged_b:array<f32,512>;
@compute @workgroup_size(32)
fn main(@builtin(workgroup_id) group:vec3u, @builtin(local_invocation_index) tid:u32) {
  let row=group.x*16u;
  let col=group.y*16u;
  var acc00=subgroup_matrix_result<f32,8,8>(0.0);
  var acc01=subgroup_matrix_result<f32,8,8>(0.0);
  var acc10=subgroup_matrix_result<f32,8,8>(0.0);
  var acc11=subgroup_matrix_result<f32,8,8>(0.0);
  for(var k=0u;k<u.K;k+=32u) {
    for(var i=tid;i<512u;i+=32u) {
      staged_a[i]=A[(row+i/32u)*u.K+k+i%32u];
      staged_b[i]=f32(B[(col+i/32u)*u.K+k+i%32u]);
    }
    workgroupBarrier();
    for(var inner=0u;inner<32u;inner+=8u) {
      let a0=subgroupMatrixLoad<subgroup_matrix_left<f32,8,8>>(&staged_a,inner,false,32u);
      let a1=subgroupMatrixLoad<subgroup_matrix_left<f32,8,8>>(&staged_a,256u+inner,false,32u);
      let b0=subgroupMatrixLoad<subgroup_matrix_right<f32,8,8>>(&staged_b,inner,true,32u);
      let b1=subgroupMatrixLoad<subgroup_matrix_right<f32,8,8>>(&staged_b,256u+inner,true,32u);
      acc00=subgroupMatrixMultiplyAccumulate(a0,b0,acc00);
      acc01=subgroupMatrixMultiplyAccumulate(a0,b1,acc01);
      acc10=subgroupMatrixMultiplyAccumulate(a1,b0,acc10);
      acc11=subgroupMatrixMultiplyAccumulate(a1,b1,acc11);
    }
    workgroupBarrier();
  }
  subgroupMatrixStore(&C,row*u.N+col,acc00,false,u.N);
  subgroupMatrixStore(&C,row*u.N+col+8u,acc01,false,u.N);
  subgroupMatrixStore(&C,(row+8u)*u.N+col,acc10,false,u.N);
  subgroupMatrixStore(&C,(row+8u)*u.N+col+8u,acc11,false,u.N);
}
