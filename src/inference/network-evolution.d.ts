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
    type: 'chain' | 'tree' | 'mesh' | 'dag';
    depth?: number;
    branchingFactor?: number;
  };
  nodes: NetworkNodeGene[];
  edges: NetworkEdgeGene[];
  combiner: {
    type: 'weighted' | 'voting';
    weights?: number[];
  };
}

export interface EvolutionConfig {
  populationSize?: number;
  generations?: number;
  eliteCount?: number;
  mutationRate?: number;
  random: () => number;
  evaluate: (genome: NetworkGenome) => Promise<number>;
  randomGenome: () => NetworkGenome;
}

export declare function evolveNetwork(config: EvolutionConfig): Promise<NetworkGenome>;
