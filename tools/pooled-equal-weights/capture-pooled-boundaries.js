// Untimed diagnostic: split only the enclosing compute pass to insert observations.
export async function captureSingleTokenLayers(engine, tokenId) {
  const original = engine._encodeLayerR, pass = engine.layerFuse.pass, ahead = engine.encodeAhead;
  const copies = [], device = engine.device;
  engine.reset(); engine.layerFuse.pass = false; engine.encodeAhead = false;
  engine._fwdPre = null;
  engine._encodeLayerR = function (encoder, layer, position) {
    original.call(this, encoder, layer, position);
    if (position !== 0) throw new Error('Unexpected position in single-token boundary capture');
    const size = this.dims.dim * 4;
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(this.x, 0, buffer, 0, size);
    copies.push({ layer: this.layers.indexOf(layer), buffer, size });
  };
  try {
    const logits = Array.from(await engine.forwardToken(tokenId));
    const layers = [];
    for (const { layer, buffer, size } of copies) {
      await buffer.mapAsync(GPUMapMode.READ, 0, size);
      layers.push({ layer, values: Array.from(new Float32Array(buffer.getMappedRange(0, size))) });
      buffer.unmap();
    }
    return { logits, layers, scope: 'Single-token layer outputs; native math, enclosing compute pass separated for readback only' };
  } finally {
    engine._encodeLayerR = original; engine.layerFuse.pass = pass; engine.encodeAhead = ahead;
    engine._fwdPre = null;
    for (const { buffer } of copies) buffer.destroy();
  }
}
