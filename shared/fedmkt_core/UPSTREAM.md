# FedMKT upstream record

The optional machine-learning modules in this directory were extracted and adapted from:

- Repository: `FederatedAI/FATE-LLM`
- Commit: `0c63377e468f0f62a9bdf5fb32424688b9478553`
- Package: `fate_llm.algo.fedmkt`
- License: Apache License 2.0

The original FATE and FuseAI/FuseLLM notices remain in the adapted source files.
LegalFedLLM changes imports from the FATE package namespace to this local package,
uses `AutoTokenizer` instead of the FATE tokenizer factory, and makes logit extraction
device-neutral. It also replaces the upstream `editdistance.eval` calls with
`rapidfuzz.distance.Levenshtein.distance` so Python 3.14 installations can use a
prebuilt wheel. Unused tokenizer-class entries are removed from the extracted
special-token lookup. The logit-generation adaptation disables the model cache,
stores
selected logits as float32, and normalizes CE over supervised non-padding labels
rather than every attended token. The latter preserves upstream whole-sequence CE
when all labels are supervised and correctly supports LegalFedLLM's signed
`chat_sft_answer_only_v1` label format. Operational teacher selection retains every
sample and chooses the first minimum-CE source from the Host followed by trusted
Clients in signed manifest order. The imported trainer applies distillation only
at causal positions whose next-token label is supervised. FATE Context,
Guest/Host/Arbiter channels, FATE-Flow and aggregation wrappers are not included.

The protocol-first selection and safety modules remain dependency-free. Install
`requirements.txt` before importing the upstream-derived `shared.fedmkt_core.ml`
modules.

The reviewed behavior contract and approved LegalFedLLM adaptations are recorded
in [`PARITY.md`](PARITY.md). Fixed golden tests preserve the pinned upstream
DTW/MinED path, mappings, cost matrix and per-step logit-transformation behavior.
LegalFedLLM removes the unused `greedy_dp` alternative and explicitly rejects
empty token sequences, for which upstream has no defined result.

The upstream eager, plain-JSON full-vocabulary mapper remains available only as
a parity reference. LegalFedLLM's operational vocabulary-mapping path uses the
demand-driven mapper in `shared/vocabulary_mapping.py`: cache identities bind
both exact tokenizer artifacts, direction, rules and demanded source-ID set;
equal-distance candidates deterministically choose the lowest target token ID.
The DTW transformation accepts the exact validated profile boundary markers
directly while retaining the upstream class registry as a parity fallback.

The upstream dense `DataCollatorForFedMKT` also remains available only as a
small-fixture target-construction oracle. LegalFedLLM's operational sparse-target
primitive stores token IDs, probabilities and validity flags at
`[batch, sequence, top-k]`, computes CE or KL from gathered model
log-probabilities, and applies the existing answer-only causal mask without
creating a `[batch, sequence, vocabulary]` target tensor. Duplicate suppression,
temperature softmax, empty-row base fallback, one-hot alignment fallback and
padding targets preserve the pinned upstream semantics.

The operational integration layer aligns all eligible Client candidates before
the inherited minimum-CE selection rule is applied. LegalFedLLM treats
`accepted and trust_score >= 0.5` as a binary eligibility gate: the score is
audited but never multiplies the selected distribution or loss. It builds one
demand-driven mapping for each signed Client alignment profile, requires all
profiles in a batch to terminate at one exact Host endpoint, rejects a Client
package as a unit on Client-originated mapping or alignment failure, rechecks
quorum, and aborts on Host/profile/tokenizer/cache failures. These protocol and
failure-policy adaptations are outside the extracted FedMKT implementation.
