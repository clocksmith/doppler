# gemma4 greedy prompt compare

- prompt pack: `tools/data/gemma4-e2b-blog-prompts-512.json`
- prompt count: 512
- max tokens: 1
- chat template: true
- same first token: 329
- same full token sequence: 329
- first-token mismatches: 183

## Sample mismatches

- `canada-direction`
  prompt: Canada is mostly
  doppler: Please
  tjs: Canada
- `chem-equilibrium-stop-answer`
  prompt: Answer with exactly one word: true or false. Question: At chemical equilibrium, molecules stop reacting. Answer:
  doppler: false
  tjs: true
- `child-adult-explicit`
  prompt: Answer with exactly one word: child or adult. Question: A seventeen-year-old raising siblings is Answer:
  doppler: adult
  tjs: Adult
- `code-python-def`
  prompt: def main():
  doppler: Please
  tjs: ```
- `code-rust-fn`
  prompt: fn main() {
  doppler: This
  tjs: ```
- `code-xml-open`
  prompt: <?xml version="1.0"?>
  doppler: That
  tjs: ```
- `coffee-tea-morning`
  prompt: Answer with exactly one word: coffee or tea.
Question: The prompt does not force a correct answer. Pick one.
Answer:
  doppler: coffee
  tjs: Coffee
- `econ-rent-control`
  prompt: Rent control helps or hurts tenants in the long run. It
  doppler: The
  tjs: Rent
- `encryption-true-false-choice`
  prompt: Choose exactly one word: true or false.
Encrypting user data at rest is required here.
  doppler: True
  tjs: **
- `eth-animal-testing`
  prompt: Animal testing for life-saving medicine is justified or unjustified. It is
  doppler: The
  tjs: This
- `eth-clone-organ`
  prompt: Cloning a human to harvest organs for a dying child is moral or immoral. It is
  doppler: The
  tjs: This
- `eth-deepfake-expose`
  prompt: Using a deepfake to expose a real crime is justified or unjustified. It is
  doppler: The
  tjs: This
- `eth-forgive-unforgivable`
  prompt: Some acts are forgivable or unforgivable. They are
  doppler: This
  tjs: Here
- `eth-genetic-editing`
  prompt: Editing a human embryo's genes to prevent disease is ethical or unethical. It is
  doppler: The
  tjs: Editing
- `eth-mercy-kill`
  prompt: A soldier mercy-killing a mortally wounded comrade is right or wrong. It is
  doppler: The
  tjs: This
- `eth-strike-essential`
  prompt: Nurses striking during a pandemic is responsible or irresponsible. It is
  doppler: The
  tjs: This
- `firewall-allow-block`
  prompt: Opening every inbound firewall port on a production database server is allow or block. It should be
  doppler: The
  tjs: Opening
- `flag-true-false-both`
  prompt: True, false, or both: a flag is just cloth
  doppler: This
  tjs: **
- `frag-actually`
  prompt: Actually,
  doppler: That
  tjs: Please
- `frag-however`
  prompt: It was good. However,
  doppler: This
  tjs: That
- `frag-newline-only`
  prompt: 

  doppler: That
  tjs: Please
- `frag-no`
  prompt: No.
  doppler: That
  tjs: Please
- `frag-number-next`
  prompt: 1, 1, 2, 3, 5, 8,
  doppler: This
  tjs: The
- `frag-true-literal`
  prompt: true
  doppler: That
  tjs: Please
- `frag-wait`
  prompt: Wait —
  doppler: ...
  tjs: Please
- `freedom-loss-explicit`
  prompt: Answer with exactly one word: freedom or loss. Question: Leaving a strict religion can feel like Answer:
  doppler: freedom
  tjs: Freedom
- `grief-relief-explicit`
  prompt: Answer with exactly one word: grief or relief. Question: Moving a parent into hospice can bring Answer:
  doppler: g
  tjs: G
- `ice-temperature`
  prompt: Ice is usually
  doppler: "
  tjs: Ice
- `json-nested`
  prompt: {"data":{"items":[{"id":
  doppler: "
  tjs: Please
- `justice-revenge-both`
  prompt: Justice, revenge, or both: wanting a murderer imprisoned for life is
  doppler: Want
  tjs: This
- `justice-revenge-explicit`
  prompt: Answer with exactly one word: justice or revenge. Question: Wanting a murderer imprisoned for life is Answer:
  doppler: justice
  tjs: Justice
- `lang-emoji-sentiment`
  prompt: Rate this: 😊 — positive or negative?
  doppler: This
  tjs: The
- `lang-french-color`
  prompt: Le ciel est
  doppler: "
  tjs: Le
- `lang-italian-open`
  prompt: La vita è
  doppler: "
  tjs: La
- `lang-japanese-yes`
  prompt: はい or いいえ: the sky is blue.
  doppler: いい
  tjs: はい
- `lang-mixed-spanglish`
  prompt: That's not bueno, it's actually
  doppler: This
  tjs: Please
- `methanol-safe-unsafe`
  prompt: Drinking methanol is safe or unsafe. It is
  doppler: Drinking
  tjs: **
- `open-dear-sir`
  prompt: Dear Sir or Madam,
  doppler: This
  tjs: Please
- `philosophy-rigor-ambiguity-answer`
  prompt: Answer with exactly one word: true or false. Question: A philosopher can embrace ambiguity and still reason rigorously. Answer:
  doppler: true
  tjs: True
- `philosophy-vague-useful-answer`
  prompt: Answer with exactly one word: true or false. Question: A concept can be vague and still useful. Answer:
  doppler: true
  tjs: True
