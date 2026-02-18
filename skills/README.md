# DOPPLER Shared Skills

`skills/` is the canonical skill registry for this repository.

Required aliases:
- `.claude/skills -> ../skills`
- `.gemini/skills -> ../skills`

Primary skills:
- `skills/doppler-debug/SKILL.md`
- `skills/doppler-bench/SKILL.md`
- `skills/doppler-perf-squeeze/SKILL.md`
- `skills/doppler-convert/SKILL.md`
- `skills/doppler-kernel-reviewer/SKILL.md`

Verify instruction + skill parity:

```bash
npm run agents:verify
```
