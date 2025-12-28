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
    type: 'ring' | 'tree' | 'mesh' | 'dag';
    depth?: number;
    branchingFactor?: number;
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

export const mutateGenome = (genome: NetworkGenome, mutationRate: number = 0.1): NetworkGenome => {
  const mutated = JSON.parse(JSON.stringify(genome)) as NetworkGenome;
  if (Math.random() < mutationRate) {
    const types: NetworkGenome['topology']['type'][] = ['ring', 'tree', 'mesh', 'dag'];
    mutated.topology.type = types[Math.floor(Math.random() * types.length)];
  }

  for (const node of mutated.nodes) {
    if (Math.random() < mutationRate && typeof node.temperature === 'number') {
      node.temperature = Math.min(1, Math.max(0, node.temperature + (Math.random() - 0.5) * 0.2));
    }
  }

  for (const edge of mutated.edges) {
    if (Math.random() < mutationRate) {
      edge.weight = Math.min(1, Math.max(0, edge.weight + (Math.random() - 0.5) * 0.4));
    }
  }

  return mutated;
};

export const crossoverGenome = (a: NetworkGenome, b: NetworkGenome): NetworkGenome => {
  return Math.random() < 0.5 ? JSON.parse(JSON.stringify(a)) : JSON.parse(JSON.stringify(b));
};

export async function evolveNetwork(config: EvolutionConfig): Promise<NetworkGenome> {
  const {
    populationSize = 20,
    generations = 10,
    eliteCount = 2,
    mutationRate = 0.1,
    evaluate,
    randomGenome,
  } = config;

  let population = Array.from({ length: populationSize }, () => randomGenome());

  for (let gen = 0; gen < generations; gen++) {
    const scored = await Promise.all(
      population.map(async (genome) => ({ genome, score: await evaluate(genome) }))
    );
    scored.sort((a, b) => b.score - a.score);

    const elite = scored.slice(0, eliteCount).map((item) => item.genome);
    const offspring: NetworkGenome[] = [];

    while (offspring.length < populationSize - eliteCount) {
      const parentA = scored[Math.floor(Math.random() * scored.length)].genome;
      const parentB = scored[Math.floor(Math.random() * scored.length)].genome;
      const child = mutateGenome(crossoverGenome(parentA, parentB), mutationRate);
      offspring.push(child);
    }

    population = [...elite, ...offspring];
  }

  const finalScores = await Promise.all(
    population.map(async (genome) => ({ genome, score: await evaluate(genome) }))
  );
  finalScores.sort((a, b) => b.score - a.score);
  return finalScores[0].genome;
}
