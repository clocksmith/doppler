// Signed Capsule for candidate 2df37d88 with prefill chunked at 16 tokens.
// Adapted from ../kernel-completion-repair-20261013/signed-capsule/build.js: same steps, archive,
// manifest, pieces, test-scope signer and release policy. Changes:
// - qualification reports and capture come from Reploid campaign 3 (one placement; the rig binds a
//   single resident contract file, so one Capsule carries one placement's reports);
// - the initial execution identity (from those reports) declares prefillTokenChunkSize 16;
// - the release event is sequence 4, chained to the sequence-3 renewal (74ab985f...).
// Usage: node build.js /var/tmp/qwen35-capsule-chunk16-20261014[-suffix] <campaign-3 attempt dir, e.g. a2|b>
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
assert(output&&output.startsWith('/var/tmp/qwen35-capsule-chunk16-20261014'),'Usage: node build.js /var/tmp/qwen35-capsule-chunk16-20261014[-suffix] <attempt>');
assert(/^[ab][0-9]*$/.test(attempt||''),'Campaign 3 attempt directory required');
const sourceRevision='a5f3ca4c5de29bd018402e571f4b2c432ebfc0a5';
const fixture='/var/tmp/reploid-completion-candidate-20261013';
const runtime=fixture+'/vendor/doppler/0.6.28';
const previous='/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const priorCapsule='/var/tmp/qwen35-capsule-closure-20261013';
const evidence=resolve(repo,'../reploid/artifacts/completion-qualification-3-20261014/'+attempt);
const reports=evidence+'/reports';
const archive=repo+'/artifacts/kernel-completion-repair-20261013/doppler-gpu-0.6.28.tgz';
const archiveHash='2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8';
const modelDir='/var/tmp/doppler-cold-compile-model-20261013';
const priorHistoryPath=repo+'/artifacts/kernel-completion-repair-20261013/release-renewal-3/release-events.json';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const write=async(name,value)=>fs.writeFile(output+'/'+name,JSON.stringify(value,null,2)+'\n');
assert.equal(await hash(archive),archiveHash);
assert.equal(await hash(fixture+'/archive.tgz'),archiveHash);
const summary=await read(evidence+'/summary.json');
assert(summary.allNumericalMatches&&summary.prefixes===55&&summary.tolerance===0.001&&!summary.failure&&!summary.cleanupFailure,'Campaign 3 placement did not pass');
await fs.mkdir(output); // Refuse to overwrite an earlier candidate.
await fs.mkdir(output+'/source');await fs.mkdir(output+'/source/capsule-input');
// Rig source root: an exact export of the candidate's source revision, independent of the working tree.
execFileSync('sh',['-c','git archive "$1" src | tar -x -C "$2"','export',sourceRevision,output+'/source'],{cwd:repo,stdio:'pipe'});
await fs.symlink(modelDir,output+'/source/capsule-input/model');
const capturePath='/var/tmp/reploid-completion-qualification-3-20261014/'+attempt+'/output/reference.json';
const captureHash=await hash(capturePath);
assert.equal('sha256:'+captureHash,summary.captureSha256);
const capture=await read(capturePath), run=capture.runs[0];
const sourcePath=repo+'/artifacts/qwen35-08b-capsule-intake-20261012/source-token-result.json';
const source=await read(sourcePath);
assert.equal(await hash(sourcePath),'b62fab0b64d300cbfa1c51fd233a8c73377cf5872da7d831c5d992eb7aa9d74a');
// The first request (19 tokens) is now prefilled in chunks of 16 and 3; its tokens must still equal the source.
assert.deepEqual(run.numerical[0].inputTokenIds,source.inputIds);
assert.equal(source.inputIds.length,19);
assert.deepEqual(run.numerical.map(row=>row.tokenId),source.generatedTokenIds);
assert.deepEqual(run.generation,source.settings);
assert.equal(run.stopReason,'eos-token');assert.equal(source.stopReason,'eos');
assert.equal(run.text,source.outputText);assert.equal(run.steps,2);
const order=capture.descriptors.map(row=>row.platform);
const hostOf=platform=>platform==='mac.lan'?'mac-host':platform.startsWith('amd-host')?'amd-host':platform.startsWith('intel-host')?'intel-host':null;
assert(order.every(hostOf));
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
generation.provenance.rawCapture=`reploid/artifacts/completion-qualification-3-20261014/${attempt} capture, retained locally; hashes in summary.json`;
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
assert.equal(manifestIdentity,'sha256:6b2cb026e6cd17469ad18fd5253e5e413313fc42756955544181190796c9e934');
const policy={...oldPolicy,modelIdentity:manifestIdentity,runtimeArchiveSha256:archiveHash,scope:'Local signed browser/Node opening qualification only; no public eligibility or application promotion'};
await write('release-policy.json',policy);
const release=await read(previous+'/release.json');
release.application.applicationRevision='sha256:3f6d98a0dbf9f83a11e87f28b6483291618c9073756604b5e6e3c609736ecdf6';
release.application.applicationRevisionDigest=release.application.applicationRevision;
assert.equal(await hash(fixture+'/snapshot-binding.json'),release.application.applicationRevision.slice(7));
release.revocation.policyDigest='sha256:'+await hash(output+'/release-policy.json');
const evidenceDigest='sha256:'+await hash(reports+'/hashes.json');
const route=order.map(hostOf).join(', ');
release.exclusions.known=[
 {code:'application-gate-failed',scope:'public-chat-release',reason:'Direct three-stage resident numerical comparison passed with prefill chunked at 16 tokens; installed conversations under this Capsule remain unqualified. Local qualification only.',evidenceDigest},
 {code:'unsupported-device',scope:'unobserved-gpu-provider-tuples',reason:`Evidence covers the recorded providers in this partition order only: ${route}. Other providers and other role assignments are not qualified by these reports.`,evidenceDigest},
 {code:'application-gate-failed',scope:'cold-start-responsiveness',reason:'Qualification used retained driver caches on shared hosts; cold-start responsiveness is not established for this runtime.',evidenceDigest},
];
await write('release.json',release);
const programOptions={repoRoot:output+'/source',modelDir:output+'/source/capsule-input/model',manifestPath:output+'/source/capsule-input/model/manifest.json',referenceReportPath:output+'/source/capsule-input/generation-report.json',outputPath:output+'/artifacts/program-bundle.json',createdAtUtc:new Date().toISOString()};
await write('program-options.json',programOptions);
await writeProgramBundle(programOptions);
execFileSync(process.execPath,[repo+'/tools/build-verified-piece-index.js',programOptions.modelDir,output+'/pieces.json','16777216'],{cwd:repo,stdio:'pipe'});
const priorPieces=await read(priorCapsule+'/pieces.json'), newPieces=await read(output+'/pieces.json');
assert.deepEqual(newPieces.files,priorPieces.files,'Weight and tokenizer pieces must equal the previous signed index');
assert.equal(newPieces.maxPieceBytes,priorPieces.maxPieceBytes);
assert.equal(newPieces.manifestIdentity,'sha256:'+await hash(programOptions.manifestPath));
const rigOptions={repoRoot:output+'/source',programBundlePath:programOptions.outputPath,manifestPath:programOptions.manifestPath,modelDir:programOptions.modelDir,modelIRReceiptPath:previous+'/model-ir-receipt.json',initialExecutionIdentityPath:output+'/initial-identity.json',qualificationReportPaths:paths,residentQualificationContractPath:reports+'/contract.json',pieceIndexPath:output+'/pieces.json',releaseManifestPath:output+'/release.json',signingPrivateKeyPath:previous+'/signing-private.jwk.json',signingPublicKeyPath:previous+'/signing-public.jwk.json',signingAuthority:policy.authority,outputPath:output+'/capsule.json'};
await write('rig-options.json',rigOptions);
const rig=await rigModelCapsule(rigOptions);await write('rig-result.json',rig);
const load=name=>import(pathToFileURL(runtime+'/src/config/'+name).href);
const {migrateCapsuleV2}=await load('capsule-v3.js');
const {signCapsuleReleaseEvent}=await load('capsule-release-events.js');
const {getCapsuleIdentity,verifyCapsule}=await load('capsule.js');
const signer={authority:policy.authority,privateKeyJwk:await read(rigOptions.signingPrivateKeyPath),publicKeyJwk:await read(rigOptions.signingPublicKeyPath)};
const trustedSigners={[signer.authority]:signer.publicKeyJwk};
await write('trusted-signers.json',trustedSigners);
const migration=await migrateCapsuleV2(await read(output+'/capsule.json'),{trustedSigners,signer});
await write('capsule-v3.json',migration.capsule);
const identity=getCapsuleIdentity(migration.capsule);const now=new Date().toISOString();
const expires=new Date(Date.parse(now)+policy.offlineExpirySeconds*1000).toISOString();
// Continue the authority's existing history: sequence 4 after the sequence-3 renewal.
const priorHistory=await read(priorHistoryPath);
assert.equal(priorHistory.length,3);const prior=priorHistory[2];
assert.equal(prior.sequence,3);assert.equal(prior.signature.authority,signer.authority);
assert.equal(prior.digest,'sha256:74ab985fb86baa7fe0c283bd9dabfb536b4a077e7942e9419eeca661500dea6a');
assert(Date.parse(now)<Date.parse(prior.expiresAtUtc),'Sequence 3 expired');
const event=await signCapsuleReleaseEvent({capsule:{schema:identity.schema,semanticRoot:identity.semanticRoot,envelopeDigest:identity.envelopeDigest},sequence:4,previousEventDigest:prior.digest,issuedAtUtc:now,expiresAtUtc:expires,action:'eligible',release:migration.release,migratedFrom:migration.migratedFrom,nextSigner:null},signer);
const history=[...priorHistory,event];
await write('release-events.json',history);
const open=checkpoint=>verifyCapsule(migration.capsule,{trustedSigners,releaseTrustedSigners:trustedSigners,releaseEvents:history,releasePolicy:{now,minimumSequence:4,checkpoint},artifactStore:{readArtifact:artifact=>fs.readFile(output+'/'+artifact.path)}});
// A host at the sequence-3 checkpoint and a host with no history must both admit this one.
const verification=await open({sequence:3,digest:prior.digest});
const fresh=await open({sequence:0,digest:null});
assert.deepEqual(fresh.lifecycle.checkpoint,verification.lifecycle.checkpoint);
assert.equal(verification.lifecycle.checkpoint.sequence,4);
await write('exact-runtime-verification.json',{archiveSha256:archiveHash,runtime,identity,artifactReceipts:verification.artifactReceipts,lifecycle:verification.lifecycle,verifiedFromCheckpoints:[3,0],physicalOpening:false,publicPromotion:false});
for(const name of ['source/capsule-input/generation-report.json','artifacts/program-bundle.json','capsule.json','capsule-v3.json','release.json'])assert(!/[a-z0-9-]+\.ts\.net/i.test(await fs.readFile(output+'/'+name,'utf8')),'Private machine address in '+name);
await write('build-binding.json',{builderRevision:execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim(),builderSha256:await hash(import.meta.filename),archiveSha256:archiveHash,sourceRevision,sourceRoot:output+'/source',runtime,qualificationCampaign:'reploid/artifacts/completion-qualification-3-20261014/'+attempt,partitionOrder:order.map(hostOf),qualificationReports:await Promise.all(paths.map(async p=>({path:p,sha256:await hash(p)}))),contractSha256:await hash(rigOptions.residentQualificationContractPath),modelIRReceiptSha256:await hash(rigOptions.modelIRReceiptPath),manifestSha256:await hash(rigOptions.manifestPath),pieceIndexSha256:await hash(rigOptions.pieceIndexPath),programBundleSha256:await hash(programOptions.outputPath),prefillTokenChunkSize:16,identity,issuedAtUtc:now,expiresAtUtc:expires,artifactCount:verification.artifactReceipts.length,releaseHistory:{sequence:4,previousEventDigest:prior.digest,continues:'release-renewal-3 (sequence 3)'},scope:'Test authority only; release history continues at sequence 4. No production or physical acceptance.'});
console.log(JSON.stringify({output,identity,partitionOrder:order.map(hostOf),artifacts:verification.artifactReceipts.length,releaseDigest:event.digest,expiresAtUtc:expires,physicalOpening:false}));
