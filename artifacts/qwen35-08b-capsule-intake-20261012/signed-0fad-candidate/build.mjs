import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import { execFileSync } from 'node:child_process';
import { writeProgramBundle } from '../../../src/tooling/program-bundle.js';
import { rigModelCapsule } from '../../../src/tooling/model-capsule-rig.js';

const repo=resolve(import.meta.dirname,'../../..');
const output='/var/tmp/qwen35-capsule-0fad-20261012';
const candidate='/var/tmp/doppler-resident-preparation-candidate-20261012';
const runtime=candidate+'/consumer/node_modules/doppler-gpu';
const previous='/var/tmp/doppler-0.6.28-source-20261012/capsule-input';
const reports=repo+'/artifacts/resident-pipeline-preparation-20261012/native-qualification';
const archive=candidate+'/combined-package/doppler-gpu-0.6.28.tgz';
const archiveHash='0fadac5113d2cc154f65d9d40de50722a16454995e927c7d22bedc5f1a4925cd';
const read=async p=>JSON.parse(await fs.readFile(p,'utf8'));
const hash=async p=>createHash('sha256').update(await fs.readFile(p)).digest('hex');
const write=async(name,value)=>fs.writeFile(output+'/'+name,JSON.stringify(value,null,2)+'\n');
assert.equal(await hash(archive),archiveHash);
await fs.mkdir(output); // Refuse to overwrite an earlier candidate.
await fs.mkdir(output+'/source');await fs.mkdir(output+'/source/capsule-input');
await fs.symlink(candidate+'/source/src',output+'/source/src');
await fs.symlink(previous+'/model',output+'/source/capsule-input/model');
await fs.copyFile(previous+'/generation-report.json',output+'/source/capsule-input/generation-report.json');
const paths=[0,1].flatMap(d=>[0,1].map(p=>`${reports}/direction-${d}/partition-${p}.json`));
const rows=await Promise.all(paths.map(read));
const initial=rows[0].initialExecutionIdentity;
for(const row of rows){assert.equal(row.provenance.archiveHash,'sha256:'+archiveHash);assert.deepEqual(row.initialExecutionIdentity,initial);assert.equal(row.comparison.steps.length,55);assert.equal(row.comparison.tolerance,0.001);assert.equal(row.status,'passed');}
assert.deepEqual(new Set(rows.map(row=>row.surface+':'+row.partitionIndex)),new Set(['browser-webgpu:0','node-webgpu:0','browser-webgpu:1','node-webgpu:1']));
await write('initial-identity.json',initial);
const oldPolicy=await read(previous+'/release-policy.json');
const policy={...oldPolicy,runtimeArchiveSha256:archiveHash,scope:'Local signed browser/Node opening qualification only; no public eligibility or application promotion'};
await write('release-policy.json',policy);
const release=await read(previous+'/release.json');
release.application.applicationRevision='sha256:397f4efbf47a8300660fa2e8b11761a3586c0d537044037d904aef255097ed3e';
release.application.applicationRevisionDigest=release.application.applicationRevision;
assert.equal(await hash('/var/tmp/reploid-readiness-numerical-20261012/snapshot-binding.json'),release.application.applicationRevision.slice(7));
release.revocation.policyDigest='sha256:'+await hash(output+'/release-policy.json');
const evidence='sha256:'+await hash(reports+'/evidence-binding.json');
release.exclusions.known=[
 {code:'application-gate-failed',scope:'public-chat-release',reason:'Direct browser/Node resident numerical comparison passed; signed opening, demanding concurrency and document usefulness remain unqualified. Local qualification only.',evidenceDigest:evidence},
 {code:'unsupported-device',scope:'unobserved-gpu-provider-tuples',reason:'Evidence covers the recorded Mac browser and AMD Node provider in both partition roles. Other providers, browser AMD on these exact bytes and Suby are not qualified by these reports.',evidenceDigest:evidence},
 {code:'application-gate-failed',scope:'cold-start-responsiveness',reason:'Cold AMD Node recurrent pipeline preparation took 743.124 seconds and exceeded the 300-second responsiveness bound; subsequent execution did not establish fast startup.',evidenceDigest:evidence},
];
await write('release.json',release);
const programOptions={repoRoot:output+'/source',modelDir:output+'/source/capsule-input/model',manifestPath:output+'/source/capsule-input/model/manifest.json',referenceReportPath:output+'/source/capsule-input/generation-report.json',outputPath:output+'/artifacts/program-bundle.json',createdAtUtc:new Date().toISOString()};
await write('program-options.json',programOptions);
await writeProgramBundle(programOptions);
const rigOptions={repoRoot:output+'/source',programBundlePath:programOptions.outputPath,manifestPath:programOptions.manifestPath,modelDir:programOptions.modelDir,modelIRReceiptPath:previous+'/model-ir-receipt.json',initialExecutionIdentityPath:output+'/initial-identity.json',qualificationReportPaths:paths,residentQualificationContractPath:reports+'/direction-0/contract.json',pieceIndexPath:previous+'/pieces.json',releaseManifestPath:output+'/release.json',signingPrivateKeyPath:previous+'/signing-private.jwk.json',signingPublicKeyPath:previous+'/signing-public.jwk.json',signingAuthority:policy.authority,outputPath:output+'/capsule.json'};
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
const event=await signCapsuleReleaseEvent({capsule:{schema:identity.schema,semanticRoot:identity.semanticRoot,envelopeDigest:identity.envelopeDigest},sequence:1,previousEventDigest:null,issuedAtUtc:now,expiresAtUtc:expires,action:'eligible',release:migration.release,migratedFrom:migration.migratedFrom,nextSigner:null},signer);
await write('release-events.json',[event]);
const verification=await verifyCapsule(migration.capsule,{trustedSigners,releaseTrustedSigners:trustedSigners,releaseEvents:[event],releasePolicy:{now,minimumSequence:1,checkpoint:{sequence:0,digest:null}},artifactStore:{readArtifact:artifact=>fs.readFile(output+'/'+artifact.path)}});
await write('exact-runtime-verification.json',{archiveSha256:archiveHash,runtime,identity,artifactReceipts:verification.artifactReceipts,lifecycle:verification.lifecycle,physicalOpening:false,publicPromotion:false});
await write('build-binding.json',{builderRevision:execFileSync('git',['rev-parse','HEAD'],{cwd:repo,encoding:'utf8'}).trim(),builderSha256:await hash(import.meta.filename),archiveSha256:archiveHash,sourceRoot:candidate+'/source',runtime,previousCandidatePreserved:true,qualificationReports:await Promise.all(paths.map(async p=>({path:p,sha256:await hash(p)}))),contractSha256:await hash(rigOptions.residentQualificationContractPath),modelIRReceiptSha256:await hash(rigOptions.modelIRReceiptPath),manifestSha256:await hash(rigOptions.manifestPath),pieceIndexSha256:await hash(rigOptions.pieceIndexPath),programBundleSha256:await hash(programOptions.outputPath),identity,issuedAtUtc:now,expiresAtUtc:expires,artifactCount:verification.artifactReceipts.length,scope:'Isolated fresh release history; previous candidate/history preserved. No production or physical acceptance.'});
console.log(JSON.stringify({output,identity,artifacts:verification.artifactReceipts.length,expiresAtUtc:expires,physicalOpening:false}));
