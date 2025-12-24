/**
 * Chrome Extension API Type Declarations
 * Minimal declarations for DOPPLER bridge extension
 */

declare namespace chrome {
  namespace runtime {
    interface Port {
      name: string;
      disconnect(): void;
      postMessage(message: unknown): void;
      onMessage: {
        addListener(callback: (message: unknown) => void): void;
        removeListener(callback: (message: unknown) => void): void;
      };
      onDisconnect: {
        addListener(callback: () => void): void;
        removeListener(callback: () => void): void;
      };
    }

    const lastError: { message?: string } | undefined;

    function connectNative(application: string): Port;

    const onConnect: {
      addListener(callback: (port: Port) => void): void;
      removeListener(callback: (port: Port) => void): void;
    };

    const onConnectExternal: {
      addListener(callback: (port: Port) => void): void;
      removeListener(callback: (port: Port) => void): void;
    };
  }
}
