import type { GenerationOptions } from '../../../../config/generation-contract.js';
import type { SelectedTokenResult } from './token-selection.js';
type Options = GenerationOptions & { signal?: AbortSignal | null; inputIds?: number[] };
type TokenContract = { padTokenId: number | null };
export declare function prefillWithToken(prompt: string, options: Options, tokenContract: TokenContract): Promise<SelectedTokenResult>;
export declare function decodeStepWithToken(currentIds: number[], options: Options, tokenContract: TokenContract): Promise<SelectedTokenResult>;
