# UL Training Schema Changelog

## v1

1. Introduced `training.ul.schemaVersion=1`.
2. Added stage controls:
- `stage` (`stage1_joint` | `stage2_base`)
- `stage1Artifact`
- `stage1ArtifactHash`
- `artifactDir`

3. Added objective controls:
- `lambda0`
- `noiseSchedule`
- `priorAlignment`
- `decoderSigmoidWeight`
- `lossWeights`
- `freeze`

4. Added command-threaded fields for training flows:
- `trainingSchemaVersion`
- `trainingBenchSteps`
