import type { P2PTransport, P2PTransportContext } from './shard-delivery.js';

export declare const P2P_WEBRTC_DATA_PLANE_CONTRACT_VERSION: 1;

export interface BrowserWebRTCDataPlanePeerSelection {
  peerId: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface BrowserWebRTCDataPlaneChannel {
  readyState?: string;
  send(data: string | ArrayBuffer | Uint8Array): void;
  addEventListener(type: 'message' | 'error' | 'close', listener: (event: any) => void): void;
  removeEventListener(type: 'message' | 'error' | 'close', listener: (event: any) => void): void;
}

export interface BrowserWebRTCDataPlaneGetChannelContext {
  peerId: string;
  shardIndex: number;
  context: P2PTransportContext;
  peerSelection: Record<string, unknown> | null;
}

export interface BrowserWebRTCDataPlaneConfig {
  enabled?: boolean;
  peerId?: string | null;
  requestTimeoutMs?: number;
  maxPayloadBytes?: number;
  selectPeer?: ((context: P2PTransportContext) => Promise<BrowserWebRTCDataPlanePeerSelection | string | null> | BrowserWebRTCDataPlanePeerSelection | string | null) | null;
  getDataChannel?: ((context: BrowserWebRTCDataPlaneGetChannelContext) => Promise<BrowserWebRTCDataPlaneChannel | null> | BrowserWebRTCDataPlaneChannel | null) | null;
}

export declare function isBrowserWebRTCAvailable(): boolean;

export declare function createBrowserWebRTCDataPlaneTransport(
  config?: BrowserWebRTCDataPlaneConfig
): P2PTransport | null;
