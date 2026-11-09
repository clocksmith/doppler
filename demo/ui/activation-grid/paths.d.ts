import type { ActivationRecording } from './data.js';
export interface PathNode {
  id:string; label:string; role:string; x:number; y:number; width:number;
  magnitude:number|null; count:number; method:string; latest:boolean;
}
export interface ModelPathView {
  layer:number; kind:string; layers:{layer:number; magnitude:number|null}[]; nodes:PathNode[];
  outputs:(number|string|undefined)[]; width:number;
}
export declare function modelPathView(recording:ActivationRecording, atMs:number, position:number, selectedLayer:number):ModelPathView;
export declare function renderModelPaths(container:HTMLElement, graph:ModelPathView, scale:number, activationScale:number, onLayer:(layer:number)=>void, dark?:boolean):void;
export interface LayerMapOptions {
  values:(number|string|undefined)[]; width:number; layers:number; types?: readonly string[];
  scale:number; history:import('./data.js').LayerHistory; historyScale:number; position:number;
  difference:boolean; baseline:boolean; selected:number; dark:boolean;
  onLayer:(layer:number)=>void; onPosition:(position:number)=>void; onInspect:(text:string|null)=>void;
}
export declare function renderLayerMap(container:HTMLElement, options:LayerMapOptions):void;
