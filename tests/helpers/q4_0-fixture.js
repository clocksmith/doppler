import { float16ToFloat32 } from '../../src/formats/float16.js';
import { dequantQ5_0Ref } from '../kernels/reference/dequant.js';
export function q40Fixture(rows, K) {
  const bytes = new Uint8Array(rows*K/32*18), view = new DataView(bytes.buffer);
  for (let b=0;b<rows*K/32;b++) {
    view.setUint16(b*18,[0x3800,0xb400,0x2e00,0x0010][b%4],true);
    for (let j=0;j<16;j++) bytes[b*18+2+j]=((b+j*3)%16)|(((b*5+j*7)%16)<<4);
  }
  return bytes;
}
export function decodeQ40(bytes, rows, K) {
  const out=new Float32Array(rows*K), view=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  for(let b=0;b<rows*K/32;b++) {
    const scale=float16ToFloat32(view.getUint16(b*18,true));
    for(let j=0;j<16;j++) {
      const q=bytes[b*18+2+j];out[b*32+j]=(q%16-8)*scale;out[b*32+16+j]=(Math.floor(q/16)-8)*scale;
    }
  }
  return out;
}
export function projectReference(input, weights, M, N, K) {
  const out=new Float32Array(M*N);
  for(let m=0;m<M;m++)for(let n=0;n<N;n++) {
    let sum=0;for(let k=0;k<K;k++)sum+=input[m*K+k]*weights[n*K+k];out[m*N+n]=sum;
  }
  return out;
}
export function packedFixture(dtype,rows,K) {
  if(dtype==='q4_0')return q40Fixture(rows,K);
  const size={q4_1:20,q5_0:22,q8_0:34,q6k:210}[dtype], values=dtype==='q6k'?256:32;
  const bytes=new Uint8Array(rows*K/values*size),view=new DataView(bytes.buffer);
  for(let b=0;b<rows*K/values;b++) {
    const base=b*size;
    if(dtype==='q6k') {
      for(let i=0;i<192;i++)bytes[base+i]=(i*17+b*31)%256;
      for(let i=0;i<16;i++)view.setInt8(base+192+i,[-128,127,-1,0,1,-17,63,-64][(i+b)%8]);
      view.setUint16(base+208,[0x2400,0xa000,0x0010,0][b%4],true);
    } else if(dtype==='q5_0') {
      view.setUint16(base,[0x2800,0xa400,0x3000][b%3],true);
      view.setUint32(base+2,(Math.imul(b+1,0x9e3779b1)^0x5a5a0f0f)>>>0,true);
      for(let i=6;i<22;i++)bytes[base+i]=(i*37+b*11)%256;
    } else {
      view.setUint16(base,[0x2800,0xa400,0x0010,0][b%4],true);
      if(dtype==='q4_1')view.setUint16(base+2,[0xb000,0x3000,0x0001,0][b%4],true);
      for(let i=dtype==='q4_1'?4:2;i<size;i++)bytes[base+i]=(i*29+b*13)%256;
    }
  }
  return bytes;
}
export function decodePacked(dtype,bytes,rows,K) {
  if(dtype==='q4_0')return decodeQ40(bytes,rows,K);
  if(dtype==='q5_0')return dequantQ5_0Ref(bytes,rows*K/32);
  const values=dtype==='q6k'?256:32,size={q4_1:20,q8_0:34,q6k:210}[dtype];
  const out=new Float32Array(rows*K),v=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength);
  for(let b=0;b<out.length/values;b++) {
    const base=b*size,start=b*values;
    if(dtype==='q6k') {
      const d=float16ToFloat32(v.getUint16(base+208,true));
      for(let half=0;half<2;half++)for(let lane=0;lane<32;lane++) {
        const lo0=bytes[base+half*64+lane],lo1=bytes[base+half*64+32+lane],hi=bytes[base+128+half*32+lane];
        const q=[(lo0&15)|((hi&3)<<4),(lo1&15)|(((hi>>2)&3)<<4),(lo0>>4)|(((hi>>4)&3)<<4),(lo1>>4)|(((hi>>6)&3)<<4)];
        for(let part=0;part<4;part++)out[start+half*128+part*32+lane]=d*v.getInt8(base+192+half*8+part*2+Math.floor(lane/16))*(q[part]-32);
      }
    } else if(dtype==='q8_0') {
      const d=float16ToFloat32(v.getUint16(base,true));for(let j=0;j<32;j++)out[start+j]=d*v.getInt8(base+2+j);
    } else {
      const d=float16ToFloat32(v.getUint16(base,true)),m=float16ToFloat32(v.getUint16(base+2,true));
      for(let j=0;j<16;j++){const q=bytes[base+4+j];out[start+j]=(q&15)*d+m;out[start+16+j]=(q>>4)*d+m;}
    }
  }
  return out;
}
