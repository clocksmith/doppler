import fs from "node:fs";
import {createHash} from "node:crypto";
const source=fs.readFileSync(process.argv[2] ?? "src/formats/sha256.js","utf8");
const replacement="let a = hashState[0], b = hashState[1], c = hashState[2], d = hashState[3];\n  let e = hashState[4], f = hashState[5], g = hashState[6], h = hashState[7];";
const candidate=source.replace("let [a, b, c, d, e, f, g, h] = hashState;",replacement);
if(candidate===source)throw Error("No exact substitution");
const implementations={baseline:await import("data:text/javascript;base64,"+Buffer.from(source).toString("base64")),directReads:await import("data:text/javascript;base64,"+Buffer.from(candidate).toString("base64"))};
const input=new Uint8Array(32*1024*1024).fill(19);
const expected=createHash("sha256").update(input).digest("hex");
const samples=[];
for(let run=0;run<5;run++)for(const name of run%2?["directReads","baseline"]:["baseline","directReads"]){
const start=performance.now();const actual=implementations[name].sha256BytesHex(input);const elapsedMs=performance.now()-start;
if(actual!==expected)throw Error("Digest mismatch");
samples.push({run,warmup:run===0,name,elapsedMs,MiBPerSecond:32000/elapsedMs});
}
console.log(JSON.stringify({hypothesis:"Direct state reads reduce per-block typed-array iteration overhead without changing the digest, block algorithm, or incremental workspace.",node:process.version,bytes:input.length,expected,sourceSha256:createHash("sha256").update(source).digest("hex"),candidateSha256:createHash("sha256").update(candidate).digest("hex"),samples,scope:"Alternating same-process local probe. Other acceptance processes active; no browser or startup performance claim."}));
