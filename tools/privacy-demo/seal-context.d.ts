import type { ClassHandle, MainModule, SEALContext } from 'node-seal';
export declare function openExperimentContext(): Promise<{seal: MainModule; context: SEALContext; own<T extends ClassHandle>(handle: T): T; close(): void}>;
