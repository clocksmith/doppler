// Public ORT session facade. Model-visible ports are unchanged. On the first
// actual inference invocation, observe the declared initializer GPU outputs.
export class ObservedSession {
  constructor(session, ports, observe) {
    this.session = session;
    this.ports = ports;
    this.observe = observe;
    this.observed = false;
    this.inputNames = session.inputNames;
    this.inputMetadata = session.inputMetadata;
    const hidden = new Set(ports.map(port => port.output));
    this.outputNames = session.outputNames.filter(name => !hidden.has(name));
    this.outputMetadata = session.outputMetadata.filter(value => !hidden.has(value.name));
    this.config = session.config;
    if (session.outputNames.length !== this.outputNames.length + ports.length) throw new Error('Declared observation ports differ from graph');
  }

  async run(feeds, fetches, options) {
    if (fetches !== undefined) throw new Error('Initializer observation facade requires the public model default output request');
    const wanted = this.observed ? this.outputNames : [...this.outputNames, ...this.ports.map(port => port.output)];
    const result = await this.session.run(feeds, wanted, options);
    try {
      if (!this.observed) {
        for (const port of this.ports) {
          const tensor = result[port.output];
          if (!tensor || tensor.location !== 'gpu-buffer' || tensor.type !== port.type ||
              JSON.stringify(tensor.dims) !== JSON.stringify(port.shape)) {
            throw new Error(`Initializer observation must be an actual declared F32 GPU buffer: ${port.output}`);
          }
          await this.observe(port, tensor);
          tensor.dispose();
          delete result[port.output];
        }
        this.observed = true;
      }
      return result;
    } catch (error) {
      for (const tensor of Object.values(result)) tensor.dispose();
      throw error;
    }
  }

  async release() { await this.session.release(); }
}
