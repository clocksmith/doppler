# Target-only decoder ownership

RD-CANON/1 assigns mutable sampling/decoder state only to B. The initial source checkpoint recomputed A’s transition using another decoder. This correction replaces that with the existing contract owner’s logical cursor mirror: A validates exact tokens, token range and history-derived successor, then preserves B’s delta/output digest. Reploid must authenticate the B commitment before invoking A commit; the runtime does not invent that authentication.

`before.log` fails because A creates a decoder. The same test now throws from either A decoder constructor and passes three actual resident transitions, EOS, text-stop and context-limit cursor checks. Changed tokens reject/retire; in-flight cancellation and legacy behavior remain checked. Target-state and signed-opening controls pass. Types, style, and architecture pass. No GPU or package change; earlier receipts retain their historical implementation scope.
