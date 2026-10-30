// Explicit shape routing and allocation geometry only; all tensor arithmetic
// stays in the selected ONNX/WebGPU session.
export class BucketedPrefill {
  constructor(entries, controlFactory) {
    this.entries = entries;
    this.controlFactory = controlFactory;
    if (!entries.length || entries.some((entry, index) => entry.bound % 64 || entry.bound < 64 ||
        (index && entry.bound <= entries[index - 1].bound))) throw new Error('Prefill bucket bounds must be ascending native chunk64 multiples');
    const first = entries[0], hidden = new Set(first.controls.map(port => port.name));
    this.inputNames = first.session.inputNames.filter(name => !hidden.has(name));
    this.inputMetadata = first.session.inputMetadata.filter(meta => !hidden.has(meta.name));
    this.outputNames = first.session.outputNames;
    this.outputMetadata = first.session.outputMetadata;
    this.config = first.session.config;
    for (const entry of entries) {
      const expected = [...this.inputNames, ...entry.controls.map(port => port.name)];
      if (expected.length !== entry.session.inputNames.length || expected.some(name => !entry.session.inputNames.includes(name)) ||
          JSON.stringify(entry.session.outputNames) !== JSON.stringify(this.outputNames)) {
        throw new Error('Prefill bucket model/control ports differ from the complete contract');
      }
    }
  }

  async run(feeds, fetches, options) {
    const length = feeds.inputs_embeds.dims[1], entry = this.entries.find(candidate => length <= candidate.bound);
    if (!entry || length < 1) throw new Error('Prompt length is outside the declared prefill buckets');
    const controls = [], prepared = { ...feeds }, geometry = [];
    try {
      for (const control of entry.controls) {
        const recipes = control.values ?? [control];
        const values = recipes.map(expression => expression.constant === undefined ?
          feeds[expression.sourceTensor]?.dims[expression.axis] * expression.multiplier + expression.addend : expression.constant);
        if (!values.every(Number.isSafeInteger)) throw new Error(`Invalid bucket allocation geometry: ${control.name}`);
        const tensor = this.controlFactory(control.type, values, control.shape);
        controls.push(tensor); prepared[control.name] = tensor;
        geometry.push({ name: control.name, recipes, values, type: control.type });
      }
      const result = await entry.session.run(prepared, fetches, options);
      this.selectedBound = entry.bound;
      this.geometry = geometry;
      return result;
    } finally { for (const tensor of controls) tensor.dispose(); }
  }

  async release() { await Promise.all(this.entries.map(entry => entry.session.release())); }
}
