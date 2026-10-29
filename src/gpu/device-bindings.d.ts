export interface BindingValidationDevice extends GPUDevice { __dopplerBindGroupValidationWrapped?: boolean; }
/** Preserves native creation while rejecting malformed or cross-device owned buffers. */
export function wrapDeviceCreateBindGroup(device: BindingValidationDevice | null | undefined): BindingValidationDevice | null | undefined;
