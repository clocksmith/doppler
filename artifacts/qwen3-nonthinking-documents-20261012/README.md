# Official nonthinking document policy

The pinned original Qwen3-1.7B model card explicitly recommends `enable_thinking=false`, temperature 0.7, top-p 0.8, top-k 20 and min-p 0 for nonthinking use. The locally retained README hash matches the earlier source receipt; the recommendation appears at lines 151 and 276. The [freeze](freeze.json) records that immutable primary source.

The [candidate](suite.json) tests this documented mode on exactly the same eight short-document and four original contractor cases, with the same seeds and budgets. It is derived from the original thinking baseline, without the failed additional citation instruction. Only template mode and the documented temperature/top-p settings change. The official nonthinking template supplies the empty reasoning block in the prompt; generated output must contain no thinking markers and is reviewed directly, without text rewriting. Factual and citation requirements remain unchanged.

This is an exposed paired hypothesis, not a fresh holdout. It is prepared but not yet executed; execution follows settlement of the explicit-citation run. No runtime, package, model, reference or application policy changes. Source acceptance would still require converted-weight and physical runtime qualification before adoption.
