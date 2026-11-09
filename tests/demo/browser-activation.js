import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { validateRecording } from '../../demo/ui/activation-grid/data.js';

export async function checkLayerMap(page) {
  await page.selectOption('#activity-mode','difference');
  await page.locator('#activation-scrub').fill('22');
  await page.locator('.activity-layer-tile').first().waitFor();
  assert.equal(await page.locator('.activity-layer-tile').count(),24);
  assert.equal(await page.locator('#activation-canvas').isVisible(),false);
  const pixels = () => page.locator('.activity-layer-tile canvas').first().evaluate(c=>[...c.getContext('2d').getImageData(0,0,c.width,c.height).data]);
  const before=await pixels();
  await page.locator('#activation-scrub').fill('23');
  await page.waitForFunction(()=>document.querySelector('.activity-history canvas').getAttribute('aria-valuenow')==='23');
  assert.notDeepEqual(await pixels(),before,'Visible tile updates with the same replay token');
  await page.locator('.activity-history canvas').focus();
  await page.keyboard.press('ArrowLeft');
  await page.waitForFunction(()=>document.querySelector('.activity-history canvas').getAttribute('aria-valuenow')==='22');
  assert.equal(await page.locator('#output-text').textContent(),'The','History navigation also rewinds the matching answer');
  await page.locator('.activity-layer-tile').nth(3).click();
  await page.locator('.activity-layer-detail').waitFor({state:'visible'});
  assert.match(await page.locator('.activity-layer-detail').textContent(),/Full attention/);
  await page.locator('.activity-layer-detail > button').click();
  await page.locator('.activity-layer-detail').waitFor({state:'hidden'});
  await fs.mkdir('.cache/layer-map',{recursive:true});
  for (const width of [1280,390,320]) {
    await page.setViewportSize({width,height:950});
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
    const geometry = await page.locator('.activity-layer-tile canvas').evaluateAll(cs=>cs.map(c=>({w:c.width,h:c.height,box:c.getBoundingClientRect().width})));
    assert(geometry.every(c=>c.w===32 && c.h===32 && c.box>=32),'Every scalar remains represented at desktop and phone widths');
    await page.locator('#inspection-workspace').screenshot({path:`.cache/layer-map/light-${width}.png`});
  }
  const light=await pixels();
  await page.emulateMedia({colorScheme:'dark'});
  await page.evaluate(()=>new Promise(requestAnimationFrame));
  assert.notDeepEqual(await pixels(),light,'Tiles follow system theme');
  assert((await pixels()).filter((_,i)=>i%4===3).every(a=>a===255),'All pixels are opaque');
  await page.locator('#inspection-workspace').screenshot({path:'.cache/layer-map/dark-320.png'});
  await page.emulateMedia({colorScheme:'light'});
  await page.setViewportSize({width:1280,height:1000});
}

// Geometry protects the first-load gap and keeps secondary controls out of the main view.
export async function checkFirstLoadLayout(page, empty = false) {
  await fs.mkdir('.cache/demo-first-load', { recursive: true });
  for (const width of [1280, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => new Promise(requestAnimationFrame));
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
    if (empty) {
      const surface = await page.locator('.chat-surface').boundingBox();
      const composer = await page.locator('#input-area').boundingBox();
      assert(surface.height < 80, `Empty conversation must not reserve a transcript at ${width}px`);
      assert(composer.y - surface.y - surface.height < 40, `Composer follows empty conversation at ${width}px`);
      await page.locator('#dialogue-section').screenshot({ path: `.cache/demo-first-load/empty-${width}.png` });
    } else {
      assert.equal(await page.locator('.activity-preview').count(), 6);
      assert.equal(await page.locator('#activity-gallery').isVisible(), false, 'Preview shortcuts stay under Details');
      assert.equal(await page.locator('#activity-files').isVisible(), false, 'File actions stay under Details');
      assert.equal(await page.locator('#activity-selectors').isVisible(), false, 'Playback settings remain under Details');
      assert.equal(await page.locator('#activity-mode').isVisible(), true, 'The replay exposes its view selector');
      assert.equal(await page.locator('#activation-player').isVisible(), true);
      const example = await page.locator('#shuffle-btn').boundingBox();
      assert(example && example.height <= 44, 'The example action fits one line beside Send');
      await page.locator('#inspection-workspace').screenshot({ path: `.cache/demo-first-load/activity-${width}.png` });
    }
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
}

// Synthetic data exercises only the UI. Physical captures are a separate opt-in check.
export async function checkActivationPlayback(page) {
  const frames = [0, 1, 0, 1].map((layer, index) => ({
    index, forward: Math.floor(index / 2), atMs: index * 5,
    phase: index < 2 ? 'prefill' : 'decode', role: 'layer-output', layer,
    tokenStart: index < 2 ? 0 : 1, tokenIds: [10 + Math.floor(index / 2)], shape: [1, 3],
    dtype: 'f32', precision: null, status: 'captured', values: [index, index ? -index : 0, .5], reason: null,
  }));
  const recording = { schema: 'doppler.activation-recording/v1', demoBuild: 'synthetic-ui-test',
    display: { layout: 'layer-major-rgb/v1', palette: 'signed-tanh-rgb/v1', scale: 1 },
    trace: { schema: 'doppler.activation-trace/v1', identity: { modelId: 'synthetic-ui-test', manifestHash: 'a'.repeat(64), runtimeVersion: 'fixture' },
      geometry: { layers: 2, width: 3 }, capturePolicy: { maxBytes: 1024, maxFrameBytes: 128, maxFrames: 8, maxGridValues: 64, maxInFlight: 2 },
      status: 'complete', stoppedReason: null, recordedBytes: 96, frames } };
  assert.equal(await page.locator('#activation-player').isVisible(), false, 'An instruction replaces the empty grid');
  await page.setInputFiles('#import-file', { name: 'trace.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(recording)) });
  await page.locator('#activation-player').waitFor({ state: 'visible' });
  if (!await page.locator('#run-details').evaluate(n => n.open)) await page.locator('#run-details > summary').click();
  await page.selectOption('#activity-playback', 'captures');
  await page.waitForFunction(() => document.querySelector('#activation-scrub').max === '3');
  assert.equal(await page.locator('#activity-layer-map').isVisible(), true);
  const final = await page.locator('.activity-layer-tile canvas').first().evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  const scale = await page.inputValue('#activation-scale');
  await page.locator('#activation-scrub').fill('0');
  await page.waitForFunction(() => document.querySelector('#activation-position').value === '0');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  const first = await page.locator('.activity-layer-tile canvas').first().evaluate(c => [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data]);
  assert.notDeepEqual(first, final, 'Scrubbing changes measured samples');
  assert.equal(await page.inputValue('#activation-scale'), scale, 'Playback never rescales as token position changes');
  if (!await page.locator('#activation-speed').isVisible()) await page.locator('#run-details > summary').click();
  await page.selectOption('#activation-speed', '0.25');
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 0);
  await page.click('#activation-play');
  assert.equal(await page.locator('#activation-play').textContent(), 'Play');
  if (!await page.locator('#activation-export').isVisible()) await page.locator('#run-details > summary').click();
  const pending = page.waitForEvent('download'); await page.click('#activation-export');
  const download = await pending;
  const reopened = validateRecording(JSON.parse(await fs.readFile(await download.path(), 'utf8')));
  assert.deepEqual(reopened.trace, recording.trace, 'Offline export preserves every numerical value');
  assert.equal(reopened.display.scales[`${await page.inputValue('#activity-mode')}:overview`], Number(scale), 'Export preserves the fixed view scale');
  await page.setInputFiles('#import-file', { name: 'bad.json', mimeType: 'application/json', buffer: Buffer.from('{"schema":"doppler.activation-recording/v1"}') });
  await page.waitForFunction(() => document.querySelector('#activation-status').textContent.includes('Invalid'));
  assert.equal(await page.evaluate(async () => (await import(new URL('ui/activation-grid/index.js', document.querySelector('script[src$="/demo.js"]').src))).getActivationRecording().demoBuild), 'synthetic-ui-test', 'Invalid imports preserve the accepted recording');
  assert.equal(await page.evaluate(async () => (await import(new URL('report.js', document.querySelector('script[src$="/demo.js"]').src))).buildReport()?.activationRecording ?? null), null, 'A recording is never attached to an unrelated answer receipt');
}

export async function checkTokenGeometry(page) {
  // Cover a broad probability range independently of how long the sample answer is.
  await page.evaluate(async () => {
    const entry = document.querySelector('script[src$="/demo.js"]').src;
    const output = await import(new URL('output.js', entry));
    const tokens = [.05, .2, .4, .6, .8, .95].map((probability, index) => ({
      index, tokenId: index, text: ` word${index}`, probability,
      topCandidates: [{ tokenId: 20, text: ' other', probability: .3 }, { tokenId: 21, text: ' alternative', probability: .1 }],
    }));
    output.renderTokenInspection(tokens); output.showTokenInspectorView(true);
  });
  await page.locator('.token-chip').first().click();
  await fs.mkdir('.cache/activation-grid', { recursive: true });
  for (const width of [1200, 390, 320]) {
    await page.setViewportSize({ width, height: 950 });
    const bars = await page.locator('.token-candidate-bar-bg').evaluateAll(nodes => nodes.map(n => {
      const b = n.getBoundingClientRect(); return { left: b.left, right: b.right, width: b.width };
    }));
    assert(bars.length > 1 && bars.every(b => b.width > 0));
    assert(bars.every(b => Math.abs(b.left - bars[0].left) < 1 && Math.abs(b.right - bars[0].right) < 1), `Candidate bars align at ${width}px`);
    const gradients = await page.locator('.token-chip:not(.token-chip--unavailable)').evaluateAll(nodes => nodes.map(n => getComputedStyle(n).backgroundImage));
    assert(gradients.length && gradients.every(s => s.startsWith('linear-gradient')));
    assert(new Set(gradients).size > 3, 'Continuous probability gradients are not three categorical colors');
    await page.locator('#token-inspector-view').screenshot({ path: `.cache/activation-grid/token-inspector-${width}.png` });
    assert(await page.locator('#model-select').isVisible(), 'Model selector remains visible outside Settings');
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No page overflow at ${width}px`);
  }
  await page.setViewportSize({ width: 1280, height: 1000 });
  await page.evaluate(async () => {
    const entry = document.querySelector('script[src$="/demo.js"]').src;
    const output = await import(new URL('output.js', entry));
    const { state } = await import(new URL('ui/state.js', entry));
    output.renderTokenInspection(state.lastInspection.tokens);
  });
  await page.locator('.token-chip').first().click();
}

// Uses the published physical recording; does not run a model or fixture generator.
export async function checkRealActivityPlayback(page) {
  const sample = await page.evaluate(async () => (await import(new URL('data/sample-inspection.js', document.querySelector('script[src$="/demo.js"]').src))).SAMPLE_INSPECTION_RECEIPT);
  const promptPosition = String(sample.inputTokenIds.length - 1);
  // Reopen the initial gallery to exercise its actual entry point.
  await page.evaluate(() => { localStorage.setItem('doppler.demo.xray-enabled', 'true'); localStorage.setItem('doppler.demo.activity-example', 'true'); });
  await page.reload({ waitUntil: 'networkidle' });
  assert.equal(await page.locator('#run-details').evaluate(n => n.open), false, 'Saved timing preference must not reopen verbose details');
  assert.match(await page.locator('#activity-measurement-note').textContent(), /GPU timing on/);
  assert.equal(await page.locator('#activation-record').isChecked(), false, 'Recording opt-out survives reload');
  await page.locator('#activation-view').scrollIntoViewIfNeeded();
  await page.waitForFunction(text => document.querySelector('#output-text').textContent === text
    && document.querySelector('#activation-play').textContent === 'Play', sample.outputText, { timeout: 120000 });
  assert.equal(await page.locator('#chat-thread .chat-message-text').first().textContent(), 'Explain why the sky is blue in one sentence.');
  assert.match(await page.locator('#recorded-run-note').textContent(), /Recorded replay/);
  assert.match(await page.locator('#activation-layout').textContent(), /fully opaque/);
  assert.equal(await page.locator('#activity-explanation').isVisible(), false);
  assert.equal(await page.locator('#activation-live').isVisible(), false, 'Saved recordings have no live-follow action');
  assert.equal(await page.locator('#activity-paths-button').isVisible(), false, 'Unpinned paths need no follow-layer action');
  assert.equal(await page.inputValue('#activity-mode'), 'difference', 'Activation changes opens by default');
  await page.selectOption('#activity-mode', 'model-paths');
  await page.locator('#activity-paths').waitFor({state:'visible'});
  assert.equal(await page.locator('#activity-paths').isVisible(), true);
  assert.equal(await page.locator('.activity-path-layer').count(), 24);
  await page.getByRole('button', {name:'Inspect layer 4', exact:true}).click();
  await page.waitForFunction(() => document.querySelector('.activity-path-detail').textContent.includes('Full attention'));
  assert.match(await page.locator('.activity-path-detail').textContent(), /Attention resultnot captured/);
  await page.click('#activity-paths-button');
  await page.locator('#activation-scrub').fill('0');
  const before = await page.locator('#activity-paths').textContent();
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 2);
  await page.click('#activation-play');
  assert.notEqual(await page.locator('#activity-paths').textContent(), before, 'Playback changes captured path values');
  await page.locator('#activation-scrub').fill(await page.locator('#activation-scrub').getAttribute('max'));
  await page.selectOption('#activity-mode', 'contributions');
  await page.locator('#run-details > summary').click();
  await page.locator('#activation-position').fill(promptPosition);
  await page.locator('#run-details > summary').click();
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.match(await page.locator('#activation-caption').textContent(), new RegExp(`token ${promptPosition}`), await page.locator('#activity-availability').textContent() + ' ' + await page.locator('#activation-status').textContent());
  assert.equal(await page.inputValue('#activity-mode'), 'contributions', 'The selected preview opens its own interactive grid');
  assert.equal(await page.locator('#activity-gallery').isVisible(), false, 'One view selector replaces the redundant shortcut row');
  assert.match(await page.locator('#activity-source').textContent(), /Recorded example/);
  await page.locator('#run-details > summary').click();
  const modes = await page.locator('#activity-mode option').evaluateAll(options => options.map(o => ({ id: o.value, disabled: o.disabled })));
  const results = [];
  for (const mode of modes) {
    if (mode.disabled || mode.id === 'model-paths') { results.push({ ...mode }); continue; }
    await page.selectOption('#activity-mode', mode.id);
    await page.evaluate(() => new Promise(requestAnimationFrame));
    const state = await page.evaluate(() => {
      const c = document.querySelector('#activation-canvas');
      const layout = document.querySelector('#activation-layout').textContent;
      return { caption: document.querySelector('#activation-caption').textContent, layout, width: c.width, height: c.height,
        opaque: [...c.getContext('2d').getImageData(0, 0, c.width, c.height).data].filter((_,i) => i % 4 === 3).every(a => a === 255),
        pixelColors: new Set(c.getContext('2d').getImageData(0, 0, c.width, c.height).data).size,
        position: document.querySelector('#activation-position').value };
    });
    const snapshots = ['weights','quantization-error','attention-prefill'].includes(mode.id);
    assert.equal(await page.locator('#activation-timeline').isVisible(), !snapshots, `${mode.id} has truthful playback controls`);
    assert.equal(state.position, promptPosition, 'Switching grids preserves the shared token position');
    assert(state.opaque, `${mode.id} has no alpha-encoded dimension`);
    assert(state.width > 1 && state.pixelColors > 2, `${mode.id} renders recorded numerical values`);
    assert(!state.layout.startsWith('0/'), `${mode.id} has real capture coverage`);
    results.push({ id: mode.id, ...state });
  }
  assert.equal(modes.filter(m => !m.disabled).length, 19);
  assert.equal(modes.find(m => m.id === 'expert-routing').disabled, true, 'A dense model cannot pretend to route experts');
  await page.selectOption('#activity-mode', 'weights');
  await page.locator('#run-details > summary').click();
  await page.locator('#inspection-workspace').screenshot({ path: '.cache/activation-grid/weights-heatmap.png' });
  await page.locator('#run-details > summary').click();
  await page.selectOption('#activity-detail', 'raw');
  await page.selectOption('#activity-weight', 'lm_head');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.match(await page.locator('#activation-layout').textContent(), /Other connections require/);
  await page.selectOption('#activity-mode', 'recurrent-memory');
  await page.selectOption('#activity-detail', 'overview');
  await page.locator('#activity-weight').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#activity-weight').isVisible(), false);
  await page.selectOption('#activity-mode', 'layer-output');
  if (!await page.locator('#activation-position').isVisible()) await page.locator('#run-details > summary').click();
  await fs.mkdir('.cache/activation-grid/playback', { recursive: true });
  for (const width of [1200, 390, 320]) {
    await page.setViewportSize({ width, height: 900 });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `No overflow at ${width}px`);
    for (const id of ['activity-mode', 'activation-play', 'activity-layer-map']) {
      const rect = await page.locator(`#${id}`).boundingBox();
      assert(rect && rect.x >= 0 && rect.x + rect.width <= width + 1, `${id} fits at ${width}px`);
    }
    await page.locator('#inspection-workspace').screenshot({ path: `.cache/activation-grid/playback/activity-${width}.png` });
    await page.selectOption('#activity-mode', 'model-paths');
    await page.locator('#inspection-workspace').screenshot({ path: `.cache/activation-grid/playback/paths-${width}.png` });
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    await page.selectOption('#activity-mode', 'layer-output');
    await page.locator('#activity-layer-map').waitFor({state:'visible'});
  }
  await page.selectOption('#activity-playback', 'tokens');
  await page.locator('#activation-scrub').fill('0');
  await page.click('#activation-play');
  await page.waitForFunction(() => Number(document.querySelector('#activation-scrub').value) > 1);
  await page.click('#activation-play');
  const position = await page.inputValue('#activation-position');
  await page.selectOption('#activity-mode', 'kv-cache');
  assert.equal(await page.inputValue('#activation-position'), position);
  if (!await page.locator('#activation-export').isVisible()) await page.locator('#run-details > summary').click();
  const pending = page.waitForEvent('download');
  await page.click('#activation-export');
  const downloaded = await pending;
  const exported = await fs.readFile(await downloaded.path());
  // Reopen the saved numerical file with all networking blocked.
  await page.route('**/*', route => route.abort());
  await page.setInputFiles('#import-file', { name: 'saved-activity.json', mimeType: 'application/json', buffer: exported });
  await page.waitForFunction(() => document.querySelector('#activation-status').textContent.includes('0 missing captures'));
  assert.match(await page.locator('#activity-legend').textContent(), /stored keys and values/);
  assert.equal(await page.inputValue('#prompt-input'), 'Explain why the sky is blue in one sentence.', 'Offline activity exports retain their prompt');
  assert.equal(await page.locator('#output-text').textContent(), sample.outputText, 'Offline activity exports retain their answer');
  await page.unroute('**/*');
  await fs.writeFile('.cache/activation-grid/playback/receipt.json', JSON.stringify({ schema: 'doppler.activity-playback-check/v1', origin: page.url(),
    viewports: [1200,390,320], modes: results, offlineImport: true, exportBytes: exported.length, modelExecutionRequired: false }, null, 2));
  await page.evaluate(async () => {
    const entry = document.querySelector('script[src$="/demo.js"]').src;
    const { loadSampleInspection } = await import(new URL('core.js', entry));
    const { SAMPLE_INSPECTION_RECEIPT } = await import(new URL('data/sample-inspection.js', entry));
    loadSampleInspection(SAMPLE_INSPECTION_RECEIPT);
  });
  await page.locator('#activation-player').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#activation-player').isVisible(), false, 'An answer without a capture cannot inherit another run’s activity');
}

// The public event contract drives grids during an unfinished answer, not only at receipt time.
export async function checkLiveActivity(page) {
  await page.locator('#activation-record').check();
  await page.selectOption('#activity-mode', 'weights');
  await page.evaluate(() => { __demoContract.streamNext = true; });
  await page.fill('#prompt-input', 'Stream activity while answering.');
  await page.click('#run-btn');
  await page.waitForFunction(() => typeof __demoContract.completeStream === 'function');
  const emit = async (position, value) => page.evaluate(({position, value}) => {
    const frame = { index: position, forward: position, atMs: position * 10 + 1, phase: 'decode', role: 'layer-output', layer: 0,
      tokenStart: position, tokenIds: [1], shape: [1,3], dtype: 'f32', status: 'captured', values: [value, -value, .5], reason: null };
    const input = {...frame, fullShape:[1,3], offset:0, role:'ffn-down-input', metadata:{}, representation:'raw'};
    const weights = [{ id:'layer_0/down', layer:0, shape:[1,3], raw:[1,2,3], rawOffset:0, rms:[1,2,3],
      columnEnergy:[1,4,9], rowGroupSize:1, blockSize:1, resident:true, status:'captured', dtype:'f32' }];
    __demoContract.emitActivation({ type:'activation-frame', frame, trace:{ identity:{modelId:'live-fixture', adapters:[]},
      geometry:{layers:1,width:3}, status:'recording', recordedBytes:(position+1)*24, frames:[],
      activity:{frames:[input], weights, model:{layerTypes:['full_attention']}, unavailable:{}, recordedBytes:24} } });
    __demoContract.emitTokens([1]);
  }, {position,value});
  await emit(0,1);
  await page.waitForFunction(() => document.querySelector('#activation-caption').dataset.live === 'true');
  assert.equal(await page.inputValue('#activity-mode'), 'difference', 'A static snapshot switches to activation changes when generation begins');
  const pixels = () => page.locator('.activity-layer-tile canvas').first().evaluate(c => [...c.getContext('2d').getImageData(0,0,c.width,c.height).data]);
  const startPixels = await pixels();
  await emit(1,.2);
  await page.waitForFunction(() => document.querySelector('#activation-position').value === '1');
  assert.notDeepEqual(await pixels(), startPixels, 'The default difference view changes as tokens arrive');
  await page.locator('#run-details').evaluate(node => { node.open = false; });
  await fs.mkdir('.cache/demo-quiet-progress', { recursive: true });
  const viewport = page.viewportSize();
  for (const width of [1280, 390]) {
    await page.setViewportSize({ width, height: 900 });
    assert.equal(await page.locator('#run-indicator').isVisible(), true);
    assert.equal(await page.locator('#activity-source').isVisible(), false);
    assert.equal(await page.locator('#activity-coverage').isVisible(), false);
    assert.equal(await page.locator('#activity-legend').isVisible(), true);
    assert.equal(await page.locator('#activation-caption').textContent(), 'Decode · token 1');
    for (const heading of ['dialogue-heading', 'activity-heading', 'precision-heading']) {
      assert.equal(await page.locator('#' + heading).evaluate(node => getComputedStyle(node.parentElement).clipPath), 'inset(50%)');
    }
    await page.screenshot({ path: `.cache/demo-quiet-progress/live-${width}.png`, fullPage: true });
  }
  await page.setViewportSize(viewport);

  await page.selectOption('#activity-mode', 'contributions');
  await page.evaluate(() => new Promise(requestAnimationFrame));
  const image = () => page.locator('#activation-canvas').evaluate(c => [...c.getContext('2d').getImageData(0,0,c.width,c.height).data]);
  const first = await image();
  await emit(2,.4);
  await page.waitForFunction(() => document.querySelector('#activation-position').value === '2');
  assert.notDeepEqual(await image(), first, 'Weight/input map updates before answer completes');
  assert.equal(await page.locator('#activation-live').getAttribute('aria-pressed'), 'true');
  assert.equal(await page.locator('#activation-live').isVisible(), false, 'Following live needs no redundant action');
  await page.locator('#activation-scrub').fill('0');
  await emit(3,.6);
  await page.evaluate(() => new Promise(requestAnimationFrame));
  assert.equal(await page.inputValue('#activation-position'), '0', 'Scrubbing pauses live follow');
  await page.click('#activation-live');
  await page.waitForFunction(() => document.querySelector('#activation-position').value === '3');
  assert.match(await page.locator('#output-text').textContent(), /Contract/, 'Answer text streams alongside the grid');
  await page.click('#stop-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Stopped');
  await page.waitForFunction(() => document.querySelector('#activation-live').disabled);
  assert.equal(await page.locator('#activation-live').isDisabled(), true);
  assert.equal(await page.locator('#activation-live').isVisible(), false, 'Completed runs have no live-follow action');
  await page.locator('#activation-record').uncheck();
  await page.fill('#prompt-input', 'Answer without recording activity.');
  await page.click('#run-btn');
  await page.waitForFunction(() => document.querySelector('#output-phase').textContent === 'Complete');
  await page.locator('#activation-player').waitFor({ state: 'hidden' });
  assert.equal(await page.locator('#activation-player').isVisible(), false, 'Recording-off runs clear the previous answer’s grid');
}
