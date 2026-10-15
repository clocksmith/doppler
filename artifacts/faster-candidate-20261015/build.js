// Signed Capsule for the faster candidate: runtime archive 21d054ee (Doppler 1c9ec3db) with the combined
// paired kernels selected by the execution graph, prefill chunked at 16 tokens, bound to placement C.
// Adapted from ../chunked-prefill-capsule-20261014/build.js: same steps, previous capsule-input
// (model IR receipt, generation-report template, release manifest, release policy) and test-scope
// signer. Changes:
// - runtime archive 21d054ee... (this folder's doppler-gpu-0.6.28.tgz, npm pack of 1c9ec3db) and the
//   rig source root exported from 1c9ec3db, so the program bundle carries the new WGSL programs;
// - manifest 64d74719... (the Capsule manifest 6b2cb026... with the attention, norm, projection and
//   head programs replaced and the sliced state and parallel Q/K mechanism programs declared);
// - qualification reports and capture from Reploid artifacts/faster-candidate-20261015/placement-c,
//   placement C: mac-host browser stage 0, amd-host native stage 1, a second amd-host native stage 2
//   with the output head (one placement per Capsule, as before);
// - the release event is sequence 7, chained to renewal 6 (b19052d5..., expires 2026-10-15T19:34Z),
//   and is verified from checkpoints 6, 5 and 0 with the candidate runtime, as the renewal scripts do,
//   and from checkpoints 6 and 5 with the 2df37d88 runtime the final-9 hosts run.
// Generation 2 (attempt c2 and later): manifest c6220c6a..., which also declares the shipped rmsnorm_paired
// program (mechanism kernel paired_qk_head_norm) that the runtime runs for the Q/K head norms. The
// generation-1 Capsule (capsule-1/, semantic root e84a2024...) omitted it from its signed source closure,
// so the installed stage refused it during pipeline preparation. Generation 2 is sequence 8, chained to
// that Capsule's sequence 7 (2486b1e7...), which an installed host had already admitted.
// Generation 3 (attempt c4): the chosen recurrent-state variant, the rolled sliced state program (Doppler
// 77c5d9b2, archive 3b132ee7..., manifest 548e6559...; reploid artifacts/faster-candidate-20261015/variant-choice).
// Sequence 9, chained to generation 2's sequence 8 (0b40731c...), which installed hosts had admitted.
// The generation 1 and 2 archive is unrolled-1c9ec3db/doppler-gpu-0.6.28.tgz; the generation 3 archive is this
// folder's doppler-gpu-0.6.28.tgz.
// Usage: node build.js /var/tmp/qwen35-capsule-faster-20261015[-suffix] <placement C attempt dir, e.g. c|c2|c4>
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { writeProgramBundle } from '../../src/tooling/program-bundle.js';
import { rigModelCapsule } from '../../src/tooling/model-capsule-rig.js';

const repo=resolve(import.meta.dirname,'../..');
const [output,attempt]=process.argv.slice(2);
assert(output&&output.startsWith('/var/tmp/qwen35-capsule-faster-20261015'),'Usage: node build.js /var/tmp/qwen35-capsule-faster-20261015[-suffix] <attempt>');
assert(/^c[0-9]*$/.test(attempt||''),'Placement C attempt directory required');
const candidateGenerationOf=name=>name==='c'?1:['c4','c5','c6','c7','c8'].includes(name)?3:2;
const sourceRevision=candidateGenerationOf(attempt)===3?'77c5d9b2c68b138a67d8cec24a506fa5e022ef42':'1c9ec3dbc2a369fe9331d047887ef71e45f7f6ad';
const candidateGeneration=candidateGenerationOf(attempt);
const fixture='/var/tmp/reploid-faster-candidate-20261015'+({1:'',2:'-2',3:'-3'})[candidateGeneration];
const runtime=fixture+'/vendor/doppler/0.6.28';
const previousRuntime='/var/tmp/reploid-completion-candidate-20261013/vendor/doppler/0.6.28';
const previous='/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const priorCapsule='/var/tmp/qwen35-capsule-closure-20261013';
const evidence=resolve(repo,'../reploid/artifacts/faster-candidate-20261015/placement-c/'+attempt);
const reports=evidence+'/reports';
const archive=repo+'/artifacts/faster-candidate-20261015/'+(candidateGeneration===3?'':'unrolled-1c9ec3db/')+'doppler-gpu-0.6.28.tgz';
const archiveHash=candidateGeneration===3?'3b132ee7ff984c80ef48487d3010028ade46b64d39323304d674feebfde3bf78':'21d054eeb70212a7434c9eccfe467fe0accbb8685e9358f2b72341500978cd5f';
const modelDir='/var/tmp/doppler-faster-model-20261015'+({1:'',2:'-2',3:'-3'})[candidateGeneration];
const priorHistoryPath={1:repo+'/artifacts/chunked-prefill-capsule-20261014/release-renewal-6/release-events.json',
 2:repo+'/artifacts/faster-candidate-20261015/capsule-1/release-events.json',3:repo+'/artifacts/faster-candidate-20261015/capsule-2/release-events.json'}[candidateGeneration];
const SEQUENCE={1:7,2:8,3:9}[candidateGeneration];
const PRIOR_DIGEST={1:'sha256:b19052d587ee901ec00e1ae0256bdd1211a5257aafdf1db3ad86723c71ddd755',2:'sha256:2486b1e7aa76d38415e23d145d97265ce1faecc5eec1086c9e5b54d9b6c82cb0',
 3:'sha256:0b40731ce1a425c9805790c0bbc97d3837c78c4225813abc43637b623db06fdf'}[candidateGeneration];
const MANIFEST={1:'sha256:64d747193bd983feb3d25eea6227284c6e979ea35c7cb874a3b0a0a582c2acbd',2:'sha256:c6220c6aefef7006aa2b3bc3c89932d61788340597c9abf055d27e8cdd1767fd',
 3:'sha256:548e65597f85e190d6f0edf38c63569b872a61696c28ec95dce031f913e953db'}[candidateGeneration];
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const write=async(name,value)=>fs.writeFile(output+'/'+name,JSON.stringify(value,null,2)+'\n');
assert.equal(await hash(archive),archiveHash);
assert.equal(await hash(fixture+'/archive.tgz'),archiveHash);
const summary=await read(evidence+'/summary.json');
assert(summary.allNumericalMatches&&summary.prefixes===55&&summary.tolerance===0.001&&!summary.failure&&!summary.cleanupFailure,'Placement C did not pass');
await fs.mkdir(output); // Refuse to overwrite an earlier candidate.
await fs.mkdir(output+'/source');await fs.mkdir(output+'/source/capsule-input');
// Rig source root: an exact export of the candidate's source revision, independent of the working tree.
execFileSync('sh',['-c','git archive "$1" src | tar -x -C "$2"','export',sourceRevision,output+'/source'],{cwd:repo,stdio:'pipe'});
await fs.symlink(modelDir,output+'/source/capsule-input/model');
const capturePath='/var/tmp/faster-candidate-20261015-mac/placement-c/'+attempt+'/output/reference.json';
const captureHash=await hash(capturePath);
assert.equal('sha256:'+captureHash,summary.captureSha256);
const capture=await read(capturePath), run=capture.runs[0];
const sourcePath=repo+'/artifacts/qwen35-08b-capsule-intake-20261012/source-token-result.json';
const source=await read(sourcePath);
assert.equal(await hash(sourcePath),'b62fab0b64d300cbfa1c51fd233a8c73377cf5872da7d831c5d992eb7aa9d74a');
// The first request (19 tokens) is prefilled in chunks of 16 and 3; its tokens must still equal the source.
assert.deepEqual(run.numerical[0].inputTokenIds,source.inputIds);
assert.equal(source.inputIds.length,19);
assert.deepEqual(run.numerical.map(row=>row.tokenId),source.generatedTokenIds);
assert.deepEqual(run.generation,source.settings);
assert.equal(run.stopReason,'eos-token');assert.equal(source.stopReason,'eos');
assert.equal(run.text,source.outputText);assert.equal(run.steps,2);
const order=capture.descriptors.map(row=>row.platform);
const hostOf=platform=>platform==='mac.lan'?'mac-host':platform.startsWith('amd-host')?'amd-host':platform.startsWith('intel-host')?'intel-host':null;
assert.deepEqual(order.map(hostOf),['mac-host','amd-host','amd-host']);
const generation=await read(previous+'/generation-report.json');
generation.scope=`Observed first 19-token request (prefill chunked at 16) on the three-stage route ${order.map(hostOf).join(' / ')}; exact independent source-token parity only`;
generation.surface='node';generation.output=run.text;
// Hosts are named by alias; the inherited report carried a private machine address.
generation.deviceInfo=order.map((platform,index)=>({index,host:hostOf(platform)}));
generation.metrics.generationConfig={...run.generation,seed:null};
generation.provenance.rawGenerationConfig=run.generation;
generation.provenance.seedNormalization='Captured greedy request omits seed; transcript schema represents unspecified seed as explicit null.';
generation.metrics.referenceTranscript.surface='node-webgpu';
generation.metrics.referenceTranscript.generationConfig=generation.metrics.generationConfig;
generation.metrics.sourceParity.provenance.physicalCaptureSha256=captureHash;
generation.metrics.sourceParity.provenance.archiveSha256=archiveHash;
generation.metrics.sourceParity.generation.runtimeTokens=run.numerical.map(row=>row.tokenId);
generation.provenance.captureSha256=captureHash;
generation.provenance.archiveSha256=archiveHash;
generation.provenance.rawCapture=`reploid/artifacts/faster-candidate-20261015/placement-c/${attempt} capture, retained locally; hashes in summary.json`;
delete generation.provenance.paritySha256;
generation.provenance.sourceParityRecomputed=true;
await fs.writeFile(output+'/source/capsule-input/generation-report.json',JSON.stringify(generation,null,2)+'\n');
await fs.copyFile(sourcePath,output+'/source/capsule-input/source-token-result.json');
const paths=[0,1,2].map(p=>`${reports}/partition-${p}.json`);
const rows=await Promise.all(paths.map(read));
const initial=rows[0].initialExecutionIdentity;
for(const row of rows){assert.equal(row.provenance.archiveHash,'sha256:'+archiveHash);assert.deepEqual(row.initialExecutionIdentity,initial);assert.equal(row.comparison.steps.length,55);assert.equal(row.comparison.tolerance,0.001);assert.equal(row.status,'passed');}
assert.equal(initial.programLoadPolicy.runtimeConfig.inference.session.prefillTokenChunkSize,16);
assert.equal(initial.programLoadPolicy.runtimeConfig.inference.session.prefillChunkLayers,1);
assert.deepEqual(new Set(rows.map(row=>row.surface+':'+row.partitionIndex)),new Set(['browser-webgpu:0','node-webgpu:1','node-webgpu:2']));
await write('initial-identity.json',initial);
const oldPolicy=await read(previous+'/release-policy.json');
const manifestIdentity='sha256:'+await hash(modelDir+'/manifest.json');
assert.equal(manifestIdentity,MANIFEST);
const policy={...oldPolicy,modelIdentity:manifestIdentity,runtimeArchiveSha256:archiveHash,scope:'Local signed browser/Node opening qualification only; no public eligibility or application promotion'};
await write('release-policy.json',policy);
const release=await read(previous+'/release.json');
release.application.applicationRevision='sha256:'+await hash(fixture+'/snapshot-binding.json');
release.application.applicationRevisionDigest=release.application.applicationRevision;
release.revocation.policyDigest='sha256:'+await hash(output+'/release-policy.json');
const evidenceDigest='sha256:'+await hash(reports+'/hashes.json');
const route=order.map(hostOf).join(', ');
release.exclusions.known=[
 {code:'application-gate-failed',scope:'public-chat-release',reason:'Direct three-stage resident numerical comparison passed with the combined paired kernels and prefill chunked at 16 tokens; installed conversations under this Capsule are qualified separately. Local qualification only.',evidenceDigest},
 {code:'unsupported-device',scope:'unobserved-gpu-provider-tuples',reason:`Evidence covers the recorded providers in this partition order only: ${route}. Other providers and other role assignments are not qualified by these reports.`,evidenceDigest},
 {code:'application-gate-failed',scope:'cold-start-responsiveness',reason:'Qualification used retained driver caches on shared hosts; cold-start responsiveness is not established for this runtime.',evidenceDigest},
];
await write('release.json',release);
const programOptions={repoRoot:output+'/source',modelDir:output+'/source/capsule-input/model',manifestPath:output+'/source/capsule-input/model/manifest.json',referenceReportPath:output+'/source/capsule-input/generation-report.json',outputPath:output+'/artifacts/program-bundle.json',createdAtUtc:new Date().toISOString()};
await write('program-options.json',programOptions);
await writeProgramBundle(programOptions);
// Every pipeline a captured stage prepared before readiness must be inside the signed source closure.
// (Generation 1 omitted rmsnorm_paired.wgsl, which the Q/K head norms run outside the graph's steps; the
// installed stage then refused it during preparation.)
{
 const bundle=await read(programOptions.outputPath);
 const closure=new Set(bundle.wgslModules.map(module=>module.file));
 const prepared=capture.descriptors.flatMap(row=>row.descriptor?.pipelinePreparation?.pipelines??[]).map(row=>row.shaderFile);
 assert(prepared.length>0,'The capture records no prepared pipelines');
 const outside=[...new Set(prepared)].filter(file=>!closure.has(file));
 assert.deepEqual(outside,[],'Prepared pipelines outside the signed source closure: '+outside.join(', '));
}
execFileSync(process.execPath,[repo+'/tools/build-verified-piece-index.js',programOptions.modelDir,output+'/pieces.json','16777216'],{cwd:repo,stdio:'pipe'});
const priorPieces=await read(priorCapsule+'/pieces.json'), newPieces=await read(output+'/pieces.json');
assert.deepEqual(newPieces.files,priorPieces.files,'Weight and tokenizer pieces must equal the previous signed index');
assert.equal(newPieces.maxPieceBytes,priorPieces.maxPieceBytes);
assert.equal(newPieces.manifestIdentity,'sha256:'+await hash(programOptions.manifestPath));
const rigOptions={repoRoot:output+'/source',programBundlePath:programOptions.outputPath,manifestPath:programOptions.manifestPath,modelDir:programOptions.modelDir,modelIRReceiptPath:previous+'/model-ir-receipt.json',initialExecutionIdentityPath:output+'/initial-identity.json',qualificationReportPaths:paths,residentQualificationContractPath:reports+'/contract.json',pieceIndexPath:output+'/pieces.json',releaseManifestPath:output+'/release.json',signingPrivateKeyPath:previous+'/signing-private.jwk.json',signingPublicKeyPath:previous+'/signing-public.jwk.json',signingAuthority:policy.authority,outputPath:output+'/capsule.json'};
await write('rig-options.json',rigOptions);
const rig=await rigModelCapsule(rigOptions);await write('rig-result.json',rig);
const load=(root,name)=>import(pathToFileURL(root+'/src/config/'+name).href);
const {migrateCapsuleV2}=await load(runtime,'capsule-v3.js');
const {signCapsuleReleaseEvent}=await load(runtime,'capsule-release-events.js');
const {getCapsuleIdentity,verifyCapsule}=await load(runtime,'capsule.js');
const signer={authority:policy.authority,privateKeyJwk:await read(rigOptions.signingPrivateKeyPath),publicKeyJwk:await read(rigOptions.signingPublicKeyPath)};
const trustedSigners={[signer.authority]:signer.publicKeyJwk};
await write('trusted-signers.json',trustedSigners);
const migration=await migrateCapsuleV2(await read(output+'/capsule.json'),{trustedSigners,signer});
await write('capsule-v3.json',migration.capsule);
const identity=getCapsuleIdentity(migration.capsule);const now=new Date().toISOString();
const expires=new Date(Date.parse(now)+policy.offlineExpirySeconds*1000).toISOString();
// Continue the authority's existing history: sequence 7 after the sequence-6 renewal (generation 1), or
// sequence 8 after generation 1's sequence 7 (generation 2).
const priorHistory=await read(priorHistoryPath);
assert.equal(priorHistory.length,SEQUENCE-1);const prior=priorHistory[SEQUENCE-2];
assert.equal(prior.sequence,SEQUENCE-1);assert.equal(prior.signature.authority,signer.authority);
assert.equal(prior.digest,PRIOR_DIGEST);
assert(Date.parse(now)<Date.parse(prior.expiresAtUtc),'Previous sequence expired; renew through a new decision');
const event=await signCapsuleReleaseEvent({capsule:{schema:identity.schema,semanticRoot:identity.semanticRoot,envelopeDigest:identity.envelopeDigest},sequence:SEQUENCE,previousEventDigest:prior.digest,issuedAtUtc:now,expiresAtUtc:expires,action:'eligible',release:migration.release,migratedFrom:migration.migratedFrom,nextSigner:null},signer);
const history=[...priorHistory,event];
await write('release-events.json',history);
const artifactStore={readArtifact:artifact=>fs.readFile(output+'/'+artifact.path)};
const checkpoints=[...priorHistory.slice(4).reverse().map(row=>({sequence:row.sequence,digest:row.digest})),{sequence:0,digest:null}];
const verified=[];
// The candidate runtime must admit the history (fatal). The 2df37d88 runtime is recorded only: final-10
// hosts open the Capsule with the candidate runtime it delivers.
for(const [root,label,from,required] of [[runtime,'candidate '+archiveHash.slice(0,8),checkpoints,true],[previousRuntime,'final-9 runtime 2df37d88',checkpoints.slice(0,-1),false]]){
 const {verifyCapsule:verifyWith}=await load(root,'capsule.js');
 for(const checkpoint of from){
  try{
   const verification=await verifyWith(migration.capsule,{trustedSigners,releaseTrustedSigners:trustedSigners,releaseEvents:history,releasePolicy:{now,minimumSequence:SEQUENCE,checkpoint},artifactStore});
   assert.deepEqual(verification.lifecycle.checkpoint,{sequence:SEQUENCE,digest:event.digest});
   verified.push({runtime:label,required,from:checkpoint,to:verification.lifecycle.checkpoint,artifacts:verification.artifactReceipts.length});
  }catch(error){if(required)throw error;verified.push({runtime:label,required,from:checkpoint,rejected:error.message});}
 }
}
const verification=await verifyCapsule(migration.capsule,{trustedSigners,releaseTrustedSigners:trustedSigners,releaseEvents:history,releasePolicy:{now,minimumSequence:SEQUENCE,checkpoint:checkpoints[0]},artifactStore});
await write('exact-runtime-verification.json',{archiveSha256:archiveHash,runtime,identity,artifactReceipts:verification.artifactReceipts,lifecycle:verification.lifecycle,verifiedFromCheckpoints:verified,physicalOpening:false,publicPromotion:false});
for(const name of ['source/capsule-input/generation-report.json','artifacts/program-bundle.json','capsule.json','capsule-v3.json','release.json'])assert(!/[a-z0-9-]+\.ts\.net/i.test(await fs.readFile(output+'/'+name,'utf8')),'Private machine address in '+name);
await write('build-binding.json',{builderRevision:execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim(),builderSha256:await hash(import.meta.filename),archiveSha256:archiveHash,sourceRevision,sourceRoot:output+'/source',runtime,qualificationCampaign:'reploid/artifacts/faster-candidate-20261015/placement-c/'+attempt,partitionOrder:order.map(hostOf),qualificationReports:await Promise.all(paths.map(async p=>({path:p,sha256:await hash(p)}))),contractSha256:await hash(rigOptions.residentQualificationContractPath),modelIRReceiptSha256:await hash(rigOptions.modelIRReceiptPath),manifestSha256:await hash(rigOptions.manifestPath),pieceIndexSha256:await hash(rigOptions.pieceIndexPath),programBundleSha256:await hash(programOptions.outputPath),prefillTokenChunkSize:16,identity,issuedAtUtc:now,expiresAtUtc:expires,artifactCount:verification.artifactReceipts.length,releaseHistory:{sequence:SEQUENCE,previousEventDigest:prior.digest,continues:{1:'chunked-prefill-capsule-20261014/release-renewal-6 (sequence 6)',2:'faster-candidate-20261015/capsule-1 (sequence 7)',3:'faster-candidate-20261015/capsule-2 (sequence 8)'}[candidateGeneration],verifiedFromCheckpoints:verified},scope:`Test authority only; release history continues at sequence ${SEQUENCE}. No production or physical acceptance.`});
console.log(JSON.stringify({output,identity,partitionOrder:order.map(hostOf),artifacts:verification.artifactReceipts.length,releaseDigest:event.digest,expiresAtUtc:expires,verified:verified.length,physicalOpening:false}));
