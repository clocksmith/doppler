# gemma4 greedy prompt compare

- prompt pack: `tools/data/gemma4-e2b-greedy-compare/interesting-10-prompts.json`
- prompt count: 10
- max tokens: 8
- chat template: true
- same first token: 0
- same full token sequence: 0
- first-token mismatches: 10

## Sample mismatches

- `signal-noise-explicit`
  prompt: Answer with exactly one word: signal or noise. Question: Dreams are Answer:
  doppler: signal
  tjs: Noise
- `chem-equilibrium-stop-answer-oneword-d`
  prompt: Choose one word only.
Answer with exactly one word: true or false. Question: At chemical equilibrium, molecules stop reacting. Answer:
  doppler: false
  tjs: true
- `firewall-good-bad-oneword-c`
  prompt: No explanation. One word only.
Opening every inbound firewall port on a production database server is good or bad. It is
  doppler: Bad
  tjs: Dangerous
- `lang-emoji-sentiment-oneword-b`
  prompt: Respond with one word only.
Rate this: 😊 — positive or negative?
  doppler: Neutral
  tjs: Positive
- `pol-voter-id-oneword-b`
  prompt: Respond with one word only.
Strict voter ID laws protect or suppress democracy. They
  doppler: Protect
  tjs: Suppress
- `punc-dash-answer-oneword-c`
  prompt: No explanation. One word only.
Is water wet? — Yes or no:
  doppler: Yes
  tjs: No
- `sci-consciousness-emergent-oneword-d`
  prompt: Choose one word only.
Consciousness is an illusion, emergent, or fundamental?
  doppler: Fundamental
  tjs: Emergent
- `ship-yes-no-both-oneword-d`
  prompt: Choose one word only.
Yes, no, or both: a ship rebuilt plank by plank over decades is still the same ship
  doppler: No
  tjs: Both
- `tf-privacy-dead-oneword-a`
  prompt: Answer with exactly one word only.
Answer with exactly one word: true or false. Question: Privacy is dead in the digital age. Answer:
  doppler: false
  tjs: True
- `val-happiness-meaning-oneword-c`
  prompt: No explanation. One word only.
A meaningful life is more important than a happy one — agree or disagree?
  doppler: Agree
  tjs: Disagree
