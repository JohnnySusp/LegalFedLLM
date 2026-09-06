# LegalFedLLM–FedMKT parity contract

This contract records the intended behavior after reviewing LegalFedLLM commit
`78ff927cbd738fee9b216250452ad36c7940d8b9` against the extracted FedMKT source
from FATE-LLM commit `0c63377e468f0f62a9bdf5fb32424688b9478553`.

## Approved behavior

| Concern | LegalFedLLM contract |
| --- | --- |
| Teacher selection | Select one minimum-CE teacher per sample from the Host followed by trusted Clients in signed `selected_client_ids` order. |
| Sample retention | Retain every sample, including samples for which the Host is selected. |
| Host–Client CE tie | The Host wins because it is the first candidate. |
| Client–Client CE tie | The first tied Client in signed `selected_client_ids` order wins. |
| Selection loss | Mean causal cross-entropy over true answer-token labels only. |
| Distillation mask | Answer-only causal positions selected by `chat_sft_answer_only_v1`. |

These are intentional LegalFedLLM adaptations around the imported machine-learning
core. They must not be treated as accidental drift during later upstream reviews.

## Pinned alignment profiles

The proof of concept supports four exact Client-to-Host pairs:

| Client | Validation Host | Alignment identifier |
| --- | --- | --- |
| Qwen 3 1.7B | Granite 3.3 2B | `dtw:qwen3-1.7b--granite3.3-2b-v1` |
| Granite 3.3 2B | Granite 3.3 2B | `dtw:granite3.3-2b-client--granite3.3-2b-host-v1` |
| Qwen 3 1.7B | Mistral Nemo 12B | `dtw:qwen3-1.7b--mistral-nemo-instruct-2407-v1` |
| Granite 3.3 2B | Mistral Nemo 12B | `dtw:granite3.3-2b-client--mistral-nemo-instruct-2407-v1` |

The Qwen pair exercises heterogeneous DTW mapping. The Granite pair uses separate
Client and Host protocol identities over the same pinned tokenizer and therefore
produces an exact vocabulary mapping while traversing the same DTW path. Mistral
Nemo is the selected remote Host; Granite remains a compatibility Host. The live
round manifest currently carries one alignment identity, so different Client
profiles are not mixed in one live round yet. A future Host requires new signed
Client-to-Host profiles and a new acceptance run.

Client-to-Host alignment is owned by the Coordinator; Host-to-Client alignment is
owned by the Client flow. Both directions use the same shared pure alignment
component and exact profile identities.

Unknown alignment identifiers, moving revisions, tokenizer substitutions and
role-reversed or otherwise mismatched pairs fail closed. LegalFedLLM deliberately
does not advertise alignment support for the additional model/tokenizer families
present in upstream FedMKT.

## DTW/MinED parity boundary

For valid non-empty inputs, the shared alignment component preserves the pinned
FedMKT DTW path construction, diagonal/up/left tie precedence, bidirectional
mappings, cumulative cost matrix and `transform_step_logits()` behavior.
Fixed golden tests record the expected outputs from upstream commit `0c63377`.

`dtw` is the sole real strategy. The unused `greedy_dp` implementation and public
protocol option are removed. Empty token sequences raise `ValueError` because
upstream defines no meaningful result for them. `mock_identity` remains available
only for the deterministic mock federation path.

## Vocabulary mapping and tokenizer validation

Vocabulary mapping is demand-driven in both approved directions. A cache entry
contains only the sorted unique source token IDs requested for that identity;
its path and embedded hashes bind the alignment profile, direction, mapping
rules, exact tokenizer endpoints and requested set. Writes are atomic, reuse
verifies both hashes, and invalid or stale entries fail closed without repair.

The mapping rule preserves upstream marker replacement, exact-match preference
and Levenshtein distance. LegalFedLLM makes upstream's order-dependent equal-
distance behavior reproducible by ordering candidates by target token ID, so the
lowest target ID wins a tie. This is an intentional determinism adaptation.

The pinned runtime tokenizer facts are:

| Role | `tokenizer.json` SHA-256 | Base / addressable vocabulary | Highest addressable ID | Boundary marker |
| --- | --- | --- | --- | --- |
| Qwen Client | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` | 151643 / 151669 | 151668 | `Ġ` |
| Granite Client / validation Host | `91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e` | 49152 / 49159 | 49158 | `Ġ` |
| Mistral Nemo Host | `e11c71726323d33da7b8d6f6f269f1988931c0a52b7122bcdd8c05042974e0db` | 131072 / 131072 | 131071 | `Ġ` |

Validation also checks the exact runtime class, dense addressable ID range,
special-token state, model maximum length, padding side and existing chat-
template hash. Granite intentionally uses `<|end_of_text|>` as its BOS, EOS,
padding and unknown token at ID 0; The tokenizer-validation contract preserves that 
pinned state rather than inventing or replacing special tokens. Qwen model-output 
IDs 151669 through 151935 have no tokenizer entry and are rejected explicitly if 
demanded rather than silently discarded.

The inherited class-to-marker table remains only as a parity fallback. The
approved operational interface supplies the validated profile markers directly,
which supports the exact Qwen and Granite runtime classes without broadening the
profile family list.

The reusable integration interface can now execute validated Client-to-Host DTW
alignment. The live Coordinator HTTP route remains gated so the deterministic
mock federation continues to use `mock_identity` until the operational deployment
and full-dataset validation are complete.

## Sparse-target construction and loss

The inherited dense target collator remains unchanged as a small-fixture parity
oracle. Operational target construction uses fixed-width sparse tensors for
target token IDs, probabilities and validity flags with shape
`[batch, sequence, top-k]`; it never allocates a target tensor over the complete
model vocabulary.

Sparse construction applies the inherited temperature-scaled softmax only over
the retained logits. Mapping collisions keep the first occurrence in incoming
top-k order before softmax, matching the duplicate suppression in the pinned
alignment transform. An empty aligned row uses its supplied base-model row, and
the inherited one-token alignment fallback remains a probability-one target.
Padding positions retain the upstream one-hot padding target so dense and sparse
fixtures agree before the answer-only causal mask is applied.

Sparse CE and KL use gathered model log-probabilities and are mathematically
equivalent to the corresponding dense FedMKT losses. Target IDs, shapes,
probabilities, masks, row sums and duplicate state fail explicitly when invalid.
Dense materialization is guarded by a fixture-size limit and is not an
operational interface.

## Integrated operational path

`integration.py` connects validated packages and pinned tokenizers to one shared
Client-to-Host execution path. Each Client package selects its signed approved
profile. The integration resolves one deterministic mapping per profile over the
sorted union of that profile's demanded source and top-k IDs, requires every
profile to terminate at the same Host endpoint, aligns every eligible Client
before selection, selects the first minimum-CE teacher, and emits CPU float32
sparse targets plus answer-only labels and attention masks. The original
single-profile call remains supported as a compatibility form.

Eligibility is a gate, not a weight:

```text
eligible = all hard protocol checks passed and trust_score >= 0.15
```

Changing an eligible score does not change candidate CE or target probabilities.
A Client package with invalid demanded IDs or an alignment failure is rejected as
a whole, the signed accepted order is recomputed, and quorum is checked again.
Host/profile/tokenizer or persistent-cache failures abort explicitly instead of
silently substituting Host rows. Empty rows produced by an otherwise valid
alignment retain the approved per-position Host fallback.

The integration audit hashes the ordered per-profile directions, Client groups,
mapping identities and payloads, source package hashes, accepted and rejected
Client order, teacher decisions, fallback count, trainer inputs, temperature and
loss type. Cache-hit state, cache paths, elapsed time and hardware measurements
are deliberately kept outside this deterministic audit.
