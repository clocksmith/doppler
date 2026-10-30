export function buildGGUFFixture(name,shape,dtypeId,payload) {
  const chunks=[],encoder=new TextEncoder();
  const u32=value=>{const b=new Uint8Array(4);new DataView(b.buffer).setUint32(0,value,true);chunks.push(b);};
  const u64=value=>{const b=new Uint8Array(8);new DataView(b.buffer).setBigUint64(0,BigInt(value),true);chunks.push(b);};
  const string=value=>{const b=encoder.encode(value);u64(b.length);chunks.push(b);};
  u32(0x46554747);u32(3);u64(1);u64(2);
  string('general.architecture');u32(8);string('llama');
  string('tokenizer.ggml.eos_token_id');u32(4);u32(1);
  string(name);u32(shape.length);for(const d of [...shape].reverse())u64(d);u32(dtypeId);u64(0);
  const length=chunks.reduce((n,b)=>n+b.length,0),offset=Math.ceil(length/32)*32;
  const result=new Uint8Array(offset+payload.length);let start=0;for(const b of chunks){result.set(b,start);start+=b.length;}result.set(payload,offset);return result;
}
