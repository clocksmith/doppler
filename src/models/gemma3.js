// Pointer module for the Gemma 3 family of RDRR artifacts. Entries derive
// from models/catalog.json through family.js; importing this module does not
// load manifests or shards.

import { createFamily } from './family.js';

const family = createFamily({ familyId: 'gemma3' });

export const FAMILY_ID = family.FAMILY_ID;
export const HF_REPO_ID = family.HF_REPO_ID;
export const KNOWN_MODELS = family.KNOWN_MODELS;
export const resolveModel = family.resolveModel;
export const resolveHfBaseUrl = family.resolveHfBaseUrl;
