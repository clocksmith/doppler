import assert from 'node:assert/strict';

const { doppler: nodeDoppler } = await import('../../src/client/doppler-api.js');
const { doppler: browserDoppler } = await import('../../src/client/doppler-api.browser.js');

{
  await assert.rejects(
    async () => {
      for await (const _token of nodeDoppler('hello', {
        model: 'gemma3-270m',
        runtimePreset: 'modes/debug',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );

  await assert.rejects(
    async () => {
      for await (const _token of browserDoppler('hello', {
        model: 'gemma3-270m',
        runtimePreset: 'modes/debug',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );

  await assert.rejects(
    () => nodeDoppler.text('hello', {
      model: 'gemma3-270m',
      runtimePreset: 'modes/debug',
    }),
    /does not accept load-affecting options/
  );

  await assert.rejects(
    () => browserDoppler.text('hello', {
      model: 'gemma3-270m',
      runtimePreset: 'modes/debug',
    }),
    /does not accept load-affecting options/
  );
}

{
  await assert.rejects(
    () => nodeDoppler.text('hello', {
      model: 'gemma3-270m',
      runtimeConfigUrl: '/runtime/debug.json',
    }),
    /does not accept load-affecting options/
  );

  await assert.rejects(
    () => browserDoppler.text('hello', {
      model: 'gemma3-270m',
      runtimeConfigUrl: '/runtime/debug.json',
    }),
    /does not accept load-affecting options/
  );

  await assert.rejects(
    async () => {
      for await (const _token of nodeDoppler.chat([{ role: 'user', content: 'hello' }], {
        model: 'gemma3-270m',
        runtimePreset: 'modes/debug',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );

  await assert.rejects(
    async () => {
      for await (const _token of browserDoppler.chat([{ role: 'user', content: 'hello' }], {
        model: 'gemma3-270m',
        runtimePreset: 'modes/debug',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );

  await assert.rejects(
    () => nodeDoppler.chatText([{ role: 'user', content: 'hello' }], {
      model: 'gemma3-270m',
      runtimeConfig: {
        inference: {
          prompt: 'bad',
        },
      },
    }),
    /does not accept load-affecting options/
  );

  await assert.rejects(
    () => browserDoppler.chatText([{ role: 'user', content: 'hello' }], {
      model: 'gemma3-270m',
      runtimeConfig: {
        inference: {
          prompt: 'bad',
        },
      },
    }),
    /does not accept load-affecting options/
  );

  await assert.rejects(
    async () => {
      for await (const _token of nodeDoppler.chat([{ role: 'user', content: 'hello' }], {
        model: 'gemma3-270m',
        runtimeConfigUrl: '/runtime/debug.json',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );

  await assert.rejects(
    async () => {
      for await (const _token of browserDoppler.chat([{ role: 'user', content: 'hello' }], {
        model: 'gemma3-270m',
        runtimeConfigUrl: '/runtime/debug.json',
      })) {
        break;
      }
    },
    /does not accept load-affecting options/
  );
}

{
  const [nodeModels, browserModels] = await Promise.all([
    nodeDoppler.listModels(),
    browserDoppler.listModels(),
  ]);
  assert.deepEqual(browserModels, nodeModels);
  assert.ok(browserModels.every((entry) => typeof entry === 'string' && entry.includes('-')));
}

console.log('doppler-browser-api-parity.test: ok');
