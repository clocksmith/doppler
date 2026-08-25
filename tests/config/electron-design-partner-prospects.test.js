import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

import {
  buildElectronDesignPartnerProspectsReport,
  validateElectronDesignPartnerProspects,
} from '../../tools/check-electron-design-partner-prospects.js';

const policy = JSON.parse(await fs.readFile(
  'tools/policies/electron-design-partner-prospects.json',
  'utf8'
));

function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

{
  const report = await buildElectronDesignPartnerProspectsReport();
  assert.equal(report.ok, true, report.errors.join('\n'));
  assert.equal(report.prospects.length, 5);
  assert.equal(report.primaryProspects, 3);
  assert.equal(report.qualifiedCustomers, 0);
  assert.deepEqual(report.prospects.map((prospect) => prospect.id), [
    'anythingllm',
    'joplin',
    'cherry-studio',
    'chatbox',
    'affine',
  ]);
  assert.ok(report.prospects.every((prospect) => prospect.claimAllowed === false));
  assert.ok(report.prospects.every((prospect) => (
    prospect.providerPolicy.doeProof === 'optional-separately-authorized-provider'
  )));
}

{
  const broken = clone(policy);
  broken.prospects[0].providerPolicy.doeProof = 'required';
  const report = validateElectronDesignPartnerProspects(broken);
  assert.ok(report.errors.includes(
    'anythingllm.providerPolicy.doeProof must remain optional and separately authorized'
  ));
}

{
  const broken = clone(policy);
  broken.prospects[1].claimAllowed = true;
  const report = validateElectronDesignPartnerProspects(broken);
  assert.ok(report.errors.includes(
    'joplin.claimAllowed must remain false in the prospect register'
  ));
}

{
  const broken = clone(policy);
  [broken.prospects[0], broken.prospects[1]] = [broken.prospects[1], broken.prospects[0]];
  const report = validateElectronDesignPartnerProspects(broken);
  assert.ok(report.errors.includes('prospects[0].id must be anythingllm'));
}

{
  const broken = clone(policy);
  broken.prospects[2].providerPolicy.selfPromotionAllowed = true;
  const report = validateElectronDesignPartnerProspects(broken);
  assert.ok(report.errors.includes(
    'cherry-studio.providerPolicy.selfPromotionAllowed must be false'
  ));
}

{
  const broken = clone(policy);
  broken.prospects[3].custodyPolicy.rawCustomerContent = 'shared-with-doe';
  const report = validateElectronDesignPartnerProspects(broken);
  assert.ok(report.errors.includes(
    'chatbox.custodyPolicy.rawCustomerContent must be never-cross-product-by-default'
  ));
}

console.log('electron-design-partner-prospects.test: ok');
