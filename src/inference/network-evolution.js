let fallbackRandomState = (Date.now() >>> 0) || 0x6d2b79f5;

function unseededRandom() {
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const values = new Uint32Array(1);
    crypto.getRandomValues(values);
    return values[0] / 4294967296;
  }
  fallbackRandomState = (fallbackRandomState + 0x6d2b79f5) >>> 0;
  return fallbackRandomState / 4294967296;
}

export const mutateGenome = (genome, mutationRate = 0.1) => {
  
  const mutated = JSON.parse(JSON.stringify(genome));
  if (unseededRandom() < mutationRate) {
    
    const types = ['chain', 'tree', 'mesh', 'dag'];
    mutated.topology.type = types[Math.floor(unseededRandom() * types.length)];
  }

  for (const node of mutated.nodes) {
    if (unseededRandom() < mutationRate && typeof node.temperature === 'number') {
      node.temperature = Math.min(1, Math.max(0, node.temperature + (unseededRandom() - 0.5) * 0.2));
    }
  }

  for (const edge of mutated.edges) {
    if (unseededRandom() < mutationRate) {
      edge.weight = Math.min(1, Math.max(0, edge.weight + (unseededRandom() - 0.5) * 0.4));
    }
  }

  return mutated;
};


export const crossoverGenome = (a, b) => {
  return unseededRandom() < 0.5 ? JSON.parse(JSON.stringify(a)) : JSON.parse(JSON.stringify(b));
};


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
    
    const offspring = [];

    while (offspring.length < populationSize - eliteCount) {
      const parentA = scored[Math.floor(unseededRandom() * scored.length)].genome;
      const parentB = scored[Math.floor(unseededRandom() * scored.length)].genome;
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
