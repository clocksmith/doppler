export interface ServeArgs {
  port: number;
  open: boolean;
  help: boolean;
}

export function parseArgs(argv: string[]): ServeArgs;
