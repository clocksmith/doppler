/**
 * Network evolution helpers for multi-model topologies.
 *
 * @module inference/network-evolution
 */

export interface NetworkNodeGene {
  id: string;
  adapter?: string;
  temperature?: number;
}

export interface NetworkEdgeGene {
  from: string;
  to: string;
  weight: number;
}

export interface NetworkGenome {
  topology: {
    type: 'chain' | 'ring' | 'tree' | 'mesh' | 'dag';
    depth?: number;
    branchingFactor?: number;
    maxIterations?: number;  // For circular ring
  };
  nodes: NetworkNodeGene[];
  edges: NetworkEdgeGene[];
  combiner: {
    type: 'weighted' | 'voting' | 'llm-merge';
    weights?: number[];
    combinerExpertId?: string;
  };
}

export interface EvolutionConfig {
  populationSize?: number;
  generations?: number;
  eliteCount?: number;
  mutationRate?: number;
  evaluate: (genome: NetworkGenome) => Promise<number>;
  randomGenome: () => NetworkGenome;
}

export declare const mutateGenome: (genome: NetworkGenome, mutationRate?: number) => NetworkGenome;

export declare const crossoverGenome: (a: NetworkGenome, b: NetworkGenome) => NetworkGenome;

export declare function evolveNetwork(config: EvolutionConfig): Promise<NetworkGenome>;
