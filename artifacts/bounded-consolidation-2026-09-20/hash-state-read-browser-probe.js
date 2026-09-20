import fs from "node:fs"; import {chromium} from "playwright"; import {createHash} from "node:crypto";
const source=fs.readFileSync(process.argv[2] ?? "src/formats/sha256.js","utf8");
const candidate=source.replace("let [a, b, c, d, e, f, g, h] = hashState;","let a = hashState[0], b = hashState[1], c = hashState[2], d = hashState[3];\n  let e = hashState[4], f = hashState[5], g = hashState[6], h = hashState[7];");
const expected=createHash("sha256").update(new Uint8Array(32*1024*1024).fill(19)).digest("hex");
const browser=await chromium.launch({channel:"chrome",headless:true});
try{const page=await browser.newPage(); const samples=await page.evaluate(async({source,candidate,expected})=>{
const implementations={baseline:await import("data:text/javascript,"+encodeURIComponent(source)),directReads:await import("data:text/javascript,"+encodeURIComponent(candidate))};
const input=new Uint8Array(32*1024*1024).fill(19); const samples=[];
for(let run=0;run<5;run++)for(const name of run%2?["directReads","baseline"]:["baseline","directReads"]){
const start=performance.now(); const actual=implementations[name].sha256BytesHex(input);const elapsedMs=performance.now()-start;
if(actual!==expected)throw Error("Digest mismatch"); samples.push({run,warmup:run===0,name,elapsedMs,MiBPerSecond:32000/elapsedMs});}
return samples;},{source,candidate,expected});
console.log(JSON.stringify({browser:browser.version(),bytes:33554432,expected,sourceSha256:createHash("sha256").update(source).digest("hex"),candidateSha256:createHash("sha256").update(candidate).digest("hex"),samples,scope:"Same-browser alternating synthetic hash probe; not model startup or GPU execution."}));
}finally{await browser.close();}
