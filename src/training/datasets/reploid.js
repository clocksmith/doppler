
function formatToolCall(tool, args, result) {
  const argText = args ? JSON.stringify(args) : '';
  const resultText = result ? JSON.stringify(result) : '';
  return `tool:${tool}\nargs:${argText}\nresult:${resultText}`;
}

function extractPair(event) {
  if (event.prompt && event.completion) {
    return { prompt: event.prompt, completion: event.completion };
  }
  if (event.query && event.response) {
    return { prompt: event.query, completion: event.response };
  }
  if (event.type === 'tool:execute' && event.payload) {
    const tool = event.payload.tool || 'tool';
    const args = event.payload.args || null;
    const result = event.payload.result || event.payload.output || null;
    return {
      prompt: event.payload.prompt || event.payload.query || `call ${tool}`,
      completion: formatToolCall(tool, args, result),
    };
  }
  return null;
}

export function reploidTracesToTextPairs(traces) {
  const pairs = [];
  for (const event of traces) {
    const pair = extractPair(event);
    if (pair) {
      pairs.push(pair);
    }
  }
  return pairs;
}
