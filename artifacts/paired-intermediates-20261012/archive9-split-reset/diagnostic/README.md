# Original184-token split failure localization

Archive9 remains unchanged. Frozen fixture7d60540a adds observations and selects only original request2, all51 retained prefixes. The run produced zero steps: Mac input completed layers0–11 and activation readback, while AMD lost its device in the first layer12 submission. The same AMD gfx-ring timeout/reset is confirmed in the bounded kernel journal, naming owned Chrome GPU PID1017420.

The final AMD model encoder contains14 dispatches and uses184 rows. It includes normalization, projections, convolution/recurrent attention, and FFN. Only the containing submission is localized; the trace does not establish which dispatch completed. The following readback was aborted after device loss. No layer13 model encoder was submitted. This rejects the explanation that the failed resident submission contains all12 layers, and records the actual untrimmed row count.

Both residents resolved prefillChunkLayers1 and prefillTokenChunkSize256. Actual unsplit queue grouping still needs comparison; matching configuration alone does not prove matching submissions. No arithmetic, reference, tolerance, input, or runtime package changed.

The original failure is preserved separately from cleanup observations. Mac close and queue barrier were observed; AMD resource settlement remains unknown after device loss. The owned browser launcher was subsequently stopped with SIGTERM and exited naturally. Public processes were not restarted. A prior controller setup failed because its owned browser stdin closed; that failure is retained separately.

This is shared-machine correctness diagnosis, not timing evidence. Archive9 unsplit55-prefix passes remain valid within their separate scope. No signed qualification report or promotion is justified by this failed split.
