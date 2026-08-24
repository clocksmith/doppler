export interface ParsedCliArguments {
  command: string | null;
  action: string | null;
  flags: Record<string, string | boolean>;
}

export function parseCliArguments(argv: string[]): ParsedCliArguments;
export function validateCommandFlags(parsed: ParsedCliArguments): void;
export function validateProgramBundleFlags(parsed: ParsedCliArguments): void;
export function validateIntakeFlags(parsed: ParsedCliArguments): void;
export function validateOnboardFlags(parsed: ParsedCliArguments): void;
export function validateBoundaryFlags(parsed: ParsedCliArguments): void;
export function validateBundleFlags(parsed: ParsedCliArguments): void;
export function validateProfilesFlags(parsed: ParsedCliArguments): void;
