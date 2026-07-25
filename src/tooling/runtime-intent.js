import { assertRuntimeProfileIntentCompatibility } from '../config/schema/runtime-profile.schema.js';

export function getLegacyRuntimeIntent(runtimeConfig) {
  return runtimeConfig?.shared?.tooling?.intent ?? null;
}

export function stripLegacyRuntimeIntent(runtimeConfig) {
  if (!runtimeConfig?.shared?.tooling || runtimeConfig.shared.tooling.intent === undefined) {
    return runtimeConfig;
  }
  const tooling = { ...runtimeConfig.shared.tooling };
  delete tooling.intent;
  return {
    ...runtimeConfig,
    shared: {
      ...runtimeConfig.shared,
      tooling,
    },
  };
}

export function assertRuntimeInputIntentCompatibility(requestIntent, documents) {
  for (const document of documents) {
    const legacyIntent = getLegacyRuntimeIntent(document.runtime);
    if (legacyIntent !== null && legacyIntent !== requestIntent) {
      throw new Error(
        `tooling command: ${document.kind} "${document.ref}" declares legacy ` +
        `runtime.shared.tooling.intent="${legacyIntent}", which conflicts with normalized ` +
        `request.intent="${requestIntent}".`
      );
    }
    if (document.kind === 'runtimeProfile') {
      assertRuntimeProfileIntentCompatibility(
        document.config,
        requestIntent,
        `runtime profile "${document.ref}"`
      );
    }
  }
  return true;
}
