export type ResourceOwnership =
  | 'borrowed'
  | 'scopeOwned'
  | 'submitOwned'
  | 'transferred'
  | 'retained';

export const RESOURCE_OWNERSHIP: readonly ResourceOwnership[];

export interface ResourceEvent {
  readonly sequence: number;
  readonly action: string;
  readonly label: string;
  readonly ownership: ResourceOwnership | null;
  readonly detail: string | null;
}

export interface ResourceScope<T = object> {
  readonly mode: 'immediate' | 'recorded';
  register(resource: T | null, label: string, ownership?: ResourceOwnership): T | null;
  transfer(resource: T, ownership: ResourceOwnership, detail?: string | null): T;
  release(resource: T | null, label?: string | null): boolean;
  retain(resource: T | null, label?: string | null, detail?: string | null): T | null;
  close(outcome?: 'success' | 'failure'): ReadonlyArray<ResourceEvent>;
  getEvents(): ReadonlyArray<ResourceEvent>;
}

export function createImmediateResourceScope<T = object>(options: {
  release(resource: T): void;
}): ResourceScope<T>;
export function createRecordedResourceScope<T = object>(recorder: {
  trackTemporaryBuffer(resource: T): void;
}): ResourceScope<T>;
