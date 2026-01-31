/**
 * Diffusion VAE scaffold.
 *
 * @module inference/diffusion/vae
 */

export interface DecodeLatentsOptions {
  width: number;
  height: number;
  latentWidth: number;
  latentHeight: number;
  latentChannels: number;
  latentScale: number;
}

export declare function decodeLatents(
  latents: Float32Array,
  options: DecodeLatentsOptions
): Uint8ClampedArray;
