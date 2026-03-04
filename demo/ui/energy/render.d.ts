export function clearEnergyBoard(): void;
export function clearEnergyVector(): void;
export function clearEnergyIntensityBoard(): void;
export function renderEnergyBoard(state: unknown, shapeOrSize: unknown, threshold: number): void;
export function renderEnergyVector(state: unknown, rows: number, cols: number, threshold: number): void;
export function renderEnergyIntensityBoard(state: unknown, rows: number, cols: number): void;
export function drawEnergyChart(history?: number[]): void;
export function updateEnergyStats(result: Record<string, unknown>): void;
export function clearEnergyChart(): void;
