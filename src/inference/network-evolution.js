/**
 * Network evolution helpers for multi-model topologies.
 *
 * @module inference/network-evolution
 */

/**
 * @typedef {Object} NetworkNodeGene
 * @property {string} id
 * @property {string} [adapter]
 * @property {number} [temperature]
 */

/**
 * @typedef {Object} NetworkEdgeGene
 * @property {string} from
 * @property {string} to
 * @property {number} weight
 */

/**
 * @typedef {Object} NetworkGenome
 * @property {{ type: 'chain' | 'ring' | 'tree' | 'mesh' | 'dag'; depth?: number; branchingFactor?: number; maxIterations?: number }} topology
 * @property {NetworkNodeGene[]} nodes
 * @property {NetworkEdgeGene[]} edges
 * @property {{ type: 'weighted' | 'voting' | 'llm-merge'; weights?: number[]; combinerExpertId?: string }} combiner
 */

/**
 * @typedef {Object} EvolutionConfig
 * @property {number} [populationSize]
 * @property {number} [generations]
 * @property {number} [eliteCount]
 * @property {number} [mutationRate]
 * @property {(genome: NetworkGenome) => Promise<number>} evaluate
 * @property {() => NetworkGenome} randomGenome
 */

/**
 * @param {NetworkGenome} genome
 * @param {number} [mutationRate=0.1]
 * @returns {NetworkGenome}
 */
export const mutateGenome = (genome, mutationRate = 0.1) => {
  /** @type {NetworkGenome} */
  const mutated = JSON.parse(JSON.stringify(genome));
  if (Math.random() < mutationRate) {
    /** @type {Array<'chain' | 'ring' | 'tree' | 'mesh' | 'dag'>} */
    const types = ['chain', 'ring', 'tree', 'mesh', 'dag'];
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

/**
 * @param {NetworkGenome} a
 * @param {NetworkGenome} b
 * @returns {NetworkGenome}
 */
export const crossoverGenome = (a, b) => {
  return Math.random() < 0.5 ? JSON.parse(JSON.stringify(a)) : JSON.parse(JSON.stringify(b));
};

/**
 * @param {EvolutionConfig} config
 * @returns {Promise<NetworkGenome>}
 */
export async function evolveNetwork(config) {
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
    /** @type {NetworkGenome[]} */
    const offspring = [];

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
