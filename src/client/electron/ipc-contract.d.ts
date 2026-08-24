export declare const ELECTRON_RELEASE_IPC_CHANNEL: 'doppler:release:v1';
export declare function validateElectronReleaseIpcRequest(
  value: unknown
): Record<string, unknown> & { action: string };
export declare function createElectronReleaseIpcHandler(
  coordinator: Record<string, (...args: unknown[]) => unknown>
): (event: unknown, request: unknown) => Promise<unknown>;
