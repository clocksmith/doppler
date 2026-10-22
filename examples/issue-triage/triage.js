// Issue triage policy. The application owns routes, prompt text, truncation and
// the review threshold; the scoring engine only returns raw yes/no logits.
// A suggestion is never applied here: the maintainer accepts or overrides it.

export const TRIAGE_POLICY = Object.freeze({
  schema: 'doppler-example.issue-triage-policy/v1',
  instruct: 'Given a GitHub issue, decide whether it is the described type of issue',
  routes: Object.freeze([
    Object.freeze({
      id: 'bug',
      query: 'A bug report: something is broken, crashes, raises an error, or behaves differently from what is expected or documented.',
    }),
    Object.freeze({
      id: 'feature',
      query: 'A feature request: asks for new functionality, an enhancement, or a change to how something currently works.',
    }),
    Object.freeze({
      id: 'question',
      query: 'A question: asks for help, support, an explanation, or how to use something, without reporting a defect or requesting a change.',
    }),
  ]),
  // Qwen3-Reranker's published prompt format, identical to the qualified fixture.
  prefix: '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n',
  suffix: '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',
  choices: Object.freeze([
    Object.freeze({ id: 'yes', label: 'yes' }),
    Object.freeze({ id: 'no', label: 'no' }),
  ]),
  maxSeqLen: 1024,
  bodyCharacterBudgets: Object.freeze([1600, 800, 300]),
  // Suggestions whose top-two route margin is below this are flagged for review.
  // Chosen on the training split only; see README.
  reviewMargin: null,
});

export function documentText(issue, bodyBudget) {
  const body = String(issue.body ?? '').replace(/\r\n/g, '\n').trim();
  const clipped = body.length > bodyBudget ? `${body.slice(0, bodyBudget)}…` : body;
  return `Title: ${String(issue.title ?? '').trim()}\n\n${clipped}`;
}

export function buildRouteRequest(policy, route, issue, bodyBudget) {
  return {
    prompt: `${policy.prefix}<Instruct>: ${policy.instruct}\n<Query>: ${route.query}\n<Document>: ${documentText(issue, bodyBudget)}${policy.suffix}`,
    choices: policy.choices.map((choice) => ({ ...choice })),
    maxSeqLen: policy.maxSeqLen,
  };
}

function isOversized(error) {
  return /exceeds maxSeqLen/.test(String(error?.message ?? error));
}

// scoreChoices(request, { signal }) -> { choices: [{ id, logit }] }
export async function suggestRoute(scoreChoices, issue, { policy = TRIAGE_POLICY, signal } = {}) {
  const routes = [];
  let bodyBudget = null;
  for (const budget of policy.bodyCharacterBudgets) {
    try {
      routes.length = 0;
      for (const route of policy.routes) {
        signal?.throwIfAborted();
        const result = await scoreChoices(buildRouteRequest(policy, route, issue, budget), { signal });
        const logit = (id) => result.choices.find((choice) => choice.id === id).logit;
        routes.push({ id: route.id, yes: logit('yes'), no: logit('no'), score: logit('yes') - logit('no') });
      }
      bodyBudget = budget;
      break;
    } catch (error) {
      if (!isOversized(error) || budget === policy.bodyCharacterBudgets.at(-1)) throw error;
    }
  }
  const ranked = [...routes].sort((a, b) => b.score - a.score);
  const margin = ranked[0].score - ranked[1].score;
  return {
    suggestedRoute: ranked[0].id,
    margin,
    needsReview: policy.reviewMargin == null ? true : margin < policy.reviewMargin,
    routes,
    bodyCharacterBudget: bodyBudget,
  };
}

// The maintainer's decision is the only thing that becomes a label.
export function recordDecision(suggestion, { issueId, decidedRoute, decidedBy, decidedAt = new Date().toISOString() }) {
  if (!decidedRoute || !decidedBy) throw new Error('A triage decision needs a route and a named human decider.');
  return {
    issueId,
    suggestedRoute: suggestion.suggestedRoute,
    decidedRoute,
    overridden: decidedRoute !== suggestion.suggestedRoute,
    decidedBy,
    decidedAt,
  };
}
