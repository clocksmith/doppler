export interface StaticMount {
  urlPrefix: string;
  rootDir: string;
}

export interface StaticFileServerOptions {
  rootDir?: string;
  staticMounts?: StaticMount[];
  host?: string;
  port?: number;
}

export interface StaticFileServerHandle {
  baseUrl: string;
  close: () => Promise<void>;
}

export function createStaticFileServer(
  options?: StaticFileServerOptions
): Promise<StaticFileServerHandle>;
