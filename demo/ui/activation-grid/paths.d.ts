import type { ActivationRecording } from './data.js';
export interface PathNode {
  id:string; label:string; role:string; x:number; y:number; width:number;
  magnitude:number|null; count:number; method:string; latest:boolean;
}
export interface ModelPathView {
  layer:number; kind:string; layers:{layer:number; magnitude:number|null}[]; nodes:PathNode[];
}
export declare function modelPathView(recording:ActivationRecording, atMs:number, position:number, selectedLayer:number):ModelPathView;
export declare function renderModelPaths(container:HTMLElement, graph:ModelPathView, scale:number, onLayer:(layer:number)=>void):void;
