export declare function normalizeModelType(value: any): string | null;
export declare function isCompatibleModelType(modelType: string | null, mode: string): boolean;
export declare function isModeModelSelectable(mode: string): boolean;
export declare function getModeModelLabel(mode: string): string;
export declare function getModelTypeForId(modelId: string): Promise<string>;
