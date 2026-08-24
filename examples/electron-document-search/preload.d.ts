export declare function exposeDocumentSearchReleaseBridge(
  contextBridge: { exposeInMainWorld(name: string, value: unknown): void },
  ipcRenderer: { invoke(channel: string, request: unknown): Promise<unknown> }
): void;
