// Isolated signed Capsule for the three-stage qualified candidate 2df37d88.
// Adapted from qwen35-08b-capsule-intake-20261012/signed-closure-candidate/build.js:
// same steps and test-scope signer, new archive, manifest, capture and report set.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { writeProgramBundle } from '../../../src/tooling/program-bundle.js';
import { rigModelCapsule } from '../../../src/tooling/model-capsule-rig.js';

const repo=resolve(import.meta.dirname,'../../..');
const output=process.argv[2];
assert(output&&output.startsWith('/var/tmp/qwen35-capsule-completion-20261013'),'Usage: node build.js /var/tmp/qwen35-capsule-completion-20261013[-suffix]');
const sourceRevision='a5f3ca4c5de29bd018402e571f4b2c432ebfc0a5';
const fixture='/var/tmp/reploid-completion-candidate-20261013';
const runtime=fixture+'/vendor/doppler/0.6.28';
const previous='/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const priorCapsule='/var/tmp/qwen35-capsule-closure-20261013';
const evidence=resolve(repo,'../reploid/artifacts/completion-qualification-2-20261013');
const reports=evidence+'/reports';
const archive=repo+'/artifacts/kernel-completion-repair-20261013/doppler-gpu-0.6.28.tgz';
const archiveHash='2df37d88f592b523bb90c4c79840354fd301af9803a7d9009e1c8939561c71f8';
const modelDir='/var/tmp/doppler-cold-compile-model-20261013';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const write=async(name,value)=>fs.writeFile(output+'/'+name,JSON.stringify(value,null,2)+'\n');
assert.equal(await hash(archive),archiveHash);
assert.equal(await hash(fixture+'/archive.tgz'),archiveHash);
await fs.mkdir(output); // Refuse to overwrite an earlier candidate.
await fs.mkdir(output+'/source');await fs.mkdir(output+'/source/capsule-input');
// Rig source root: an exact export of the candidate's source revision, independent of the working tree.
execFileSync('sh',['-c','git archive "$1" src | tar -x -C "$2"','export',sourceRevision,output+'/source'],{cwd:repo,stdio:'pipe'});
await fs.symlink(modelDir,output+'/source/capsule-input/model');
const capturePath='/var/tmp/reploid-completion-qualification-2-20261013/output/reference.json';
const captureHash=await hash(capturePath);
assert.equal(captureHash,'3a117a2ba86f90f0434434f3791b8ff66f9c1d5b108f499aec94938b424d2293');
const capture=await read(capturePath), run=capture.runs[0];
const sourcePath=repo+'/artifacts/qwen35-08b-capsule-intake-20261012/source-token-result.json';
const source=await read(sourcePath);
assert.equal(await hash(sourcePath),'b62fab0b64d300cbfa1c51fd233a8c73377cf5872da7d831c5d992eb7aa9d74a');
assert.deepEqual(run.numerical[0].inputTokenIds,source.inputIds);
assert.deepEqual(run.numerical.map(row=>row.tokenId),source.generatedTokenIds);
assert.deepEqual(run.generation,source.settings);
assert.equal(run.stopReason,'eos-token');assert.equal(source.stopReason,'eos');
assert.equal(run.text,source.outputText);assert.equal(run.steps,2);
const generation=await read(previous+'/generation-report.json');
generation.scope='Observed first19-token request in the three-stage Mac browser / AMD Node / Intel Node route; exact independent source-token parity only';
generation.surface='node';generation.output=run.text;
// Hosts are named by alias; the inherited report carried a private machine address.
generation.deviceInfo=[{index:0,host:'mac-host'},{index:1,host:'amd-host'},{index:2,host:'intel-host'}];
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
generation.provenance.rawCapture='reploid/artifacts/completion-qualification-2-20261013 capture, retained locally; hashes in summary.json';
delete generation.provenance.paritySha256;
generation.provenance.sourceParityRecomputed=true;
await fs.writeFile(output+'/source/capsule-input/generation-report.json',JSON.stringify(generation,null,2)+'\n');
await fs.copyFile(sourcePath,output+'/source/capsule-input/source-token-result.json');
const paths=[0,1,2].map(p=>`${reports}/partition-${p}.json`);
const rows=await Promise.all(paths.map(read));
const initial=rows[0].initialExecutionIdentity;
for(const row of rows){assert.equal(row.provenance.archiveHash,'sha256:'+archiveHash);assert.deepEqual(row.initialExecutionIdentity,initial);assert.equal(row.comparison.steps.length,55);assert.equal(row.comparison.tolerance,0.001);assert.equal(row.status,'passed');}
assert.deepEqual(new Set(rows.map(row=>row.surface+':'+row.partitionIndex)),new Set(['browser-webgpu:0','node-webgpu:1','node-webgpu:2']));
await write('initial-identity.json',initial);
const oldPolicy=await read(previous+'/release-policy.json');
const manifestIdentity='sha256:'+await hash(modelDir+'/manifest.json');
assert.equal(manifestIdentity,'sha256:6b2cb026e6cd17469ad18fd5253e5e413313fc42756955544181190796c9e934');
// The inherited policy named an older manifest; bind this candidate's model identity.
const policy={...oldPolicy,modelIdentity:manifestIdentity,runtimeArchiveSha256:archiveHash,scope:'Local signed browser/Node opening qualification only; no public eligibility or application promotion'};
await write('release-policy.json',policy);
const release=await read(previous+'/release.json');
release.application.applicationRevision='sha256:3f6d98a0dbf9f83a11e87f28b6483291618c9073756604b5e6e3c609736ecdf6';
release.application.applicationRevisionDigest=release.application.applicationRevision;
assert.equal(await hash(fixture+'/snapshot-binding.json'),release.application.applicationRevision.slice(7));
release.revocation.policyDigest='sha256:'+await hash(output+'/release-policy.json');
const evidenceDigest='sha256:'+await hash(reports+'/hashes.json');
release.exclusions.known=[
 {code:'application-gate-failed',scope:'public-chat-release',reason:'Direct three-stage resident numerical comparison passed; signed opening, installed conversations and concurrency remain unqualified. Local qualification only.',evidenceDigest},
 {code:'unsupported-device',scope:'unobserved-gpu-provider-tuples',reason:'Evidence covers the recorded Mac browser, AMD Node and Intel Node providers in their recorded partition roles only. Other providers and other role assignments are not qualified by these reports.',evidenceDigest},
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
// Continue the authority's existing release history. Checkpoints are keyed by authority and application,
// and a host that accepted the earlier release holds sequence 1; a second sequence-1 history would be a fork.
const priorHistory=await read('/var/tmp/qwen35-capsule-0fad-20261012/release-events.json');
assert.equal(priorHistory.length,1);const prior=priorHistory[0];
assert.equal(prior.sequence,1);assert.equal(prior.signature.authority,signer.authority);
assert.equal(prior.digest,'sha256:d47ddb8d6f45b23230fbf2cf1731122ccc6160c51a968e616ff6cf366bf8c9d2');
const event=await signCapsuleReleaseEvent({capsule:{schema:identity.schema,semanticRoot:identity.semanticRoot,envelopeDigest:identity.envelopeDigest},sequence:2,previousEventDigest:prior.digest,issuedAtUtc:now,expiresAtUtc:expires,action:'eligible',release:migration.release,migratedFrom:migration.migratedFrom,nextSigner:null},signer);
const history=[prior,event];
await write('release-events.json',history);
const open=checkpoint=>verifyCapsule(migration.capsule,{trustedSigners,releaseTrustedSigners:trustedSigners,releaseEvents:history,releasePolicy:{now,minimumSequence:2,checkpoint},artifactStore:{readArtifact:artifact=>fs.readFile(output+'/'+artifact.path)}});
// A host that accepted the earlier release, and a host with no history, must both admit this one.
const verification=await open({sequence:1,digest:prior.digest});
const fresh=await open({sequence:0,digest:null});
assert.deepEqual(fresh.lifecycle.checkpoint,verification.lifecycle.checkpoint);
assert.equal(verification.lifecycle.checkpoint.sequence,2);
await write('exact-runtime-verification.json',{archiveSha256:archiveHash,runtime,identity,artifactReceipts:verification.artifactReceipts,lifecycle:verification.lifecycle,physicalOpening:false,publicPromotion:false});
for(const name of ['source/capsule-input/generation-report.json','artifacts/program-bundle.json','capsule.json','capsule-v3.json'])assert(!/[a-z0-9-]+\.ts\.net/i.test(await fs.readFile(output+'/'+name,'utf8')),'Private machine address in '+name);
await write('build-binding.json',{builderRevision:execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim(),builderSha256:await hash(import.meta.filename),archiveSha256:archiveHash,sourceRevision,sourceRoot:output+'/source',runtime,previousCandidatePreserved:true,qualificationReports:await Promise.all(paths.map(async p=>({path:p,sha256:await hash(p)}))),contractSha256:await hash(rigOptions.residentQualificationContractPath),modelIRReceiptSha256:await hash(rigOptions.modelIRReceiptPath),manifestSha256:await hash(rigOptions.manifestPath),pieceIndexSha256:await hash(rigOptions.pieceIndexPath),programBundleSha256:await hash(programOptions.outputPath),identity,issuedAtUtc:now,expiresAtUtc:expires,artifactCount:verification.artifactReceipts.length,releaseHistory:{sequence:2,previousEventDigest:prior.digest,continues:'qwen35-capsule-0fad-20261012 release event'},scope:'Release history continues the authority and application history at sequence 2; previous candidates preserved. No production or physical acceptance.'});
console.log(JSON.stringify({output,identity,artifacts:verification.artifactReceipts.length,expiresAtUtc:expires,physicalOpening:false}));
