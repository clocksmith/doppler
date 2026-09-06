export declare const ELECTRON_RELEASE_IPC_CHANNEL: 'doppler:release:v1';
export declare function validateElectronReleaseIpcRequest(
  value: unknown
): Record<string, unknown> & { action: string };
export declare function createElectronReleaseIpcHandler(
  coordinator: ElectronReleaseStateCoordinator,
  options: { authorizeRequest(event: unknown, request: Record<string, unknown> & { action: string }): Promise<boolean> | boolean }
): (event: unknown, request: unknown) => Promise<unknown>;
import type { ElectronReleaseStateCoordinator } from './release-state.js';
