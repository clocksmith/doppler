# Debug Investigation Template

Use this template for model failures, incoherent output, or conversion-vs-runtime investigations that span more than one run.

## Failure

- model/artifact:
- command:
- surface:
- runtime preset/config:
- observed output:
- classification:

## Trusted reference

- source runtime:
- exact prompt text:
- token IDs:
- deterministic sampling tuple:
- early activation slice:
- output/logits slice:

## Boundary diff

| Boundary | Reference | Doppler | Verdict |
| --- | --- | --- | --- |
| embeddings |  |  |  |
| post input norm |  |  |  |
| Q/K/V pre-RoPE |  |  |  |
| Q/K post-RoPE |  |  |  |
| attention output |  |  |  |
| FFN output |  |  |  |
| final logits |  |  |  |

## First divergent boundary

- boundary:
- likely owner:
- evidence:

## Binary split

- next control experiment:
- expected branch A:
- expected branch B:

## Conversion status

- process exit:
- manifest present:
- expected shard set present:
- conversion report present:

## Conclusion

- current best diagnosis:
- next smallest permanent probe or fix:
