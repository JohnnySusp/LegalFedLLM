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

## Step 4.1 pinned alignment profile

The proof of concept supports exactly one real heterogeneous tokenizer pair:

| Role | Profile | Model and tokenizer | Immutable revision |
| --- | --- | --- | --- |
| Client | `qwen3-1.7b-lora-v1` | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` |
| Host | `granite-3.3-2b-instruct-host-lora-v1` | `ibm-granite/granite-3.3-2b-instruct` | `652c333dc5066f2a1764854a1bcd0ce67163d74f` |

The signed alignment identifier is
`dtw:qwen3-1.7b--granite3.3-2b-v1`. Client-to-Host alignment is owned by the
Coordinator; Host-to-Client alignment is owned by the Client flow. Both directions
must use the same shared pure alignment component and profile identity.

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

## Steps 4.3 and 4.4 mapping/tokenizer boundary

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
| Granite Host | `91168e938f05796aa6dcca7e485e4b30ab52785320c7a6391ecef86e6c84681e` | 49152 / 49159 | 49158 | `Ġ` |

Validation also checks the exact runtime class, dense addressable ID range,
special-token state, model maximum length, padding side and existing chat-
template hash. Granite intentionally uses `<|end_of_text|>` as its BOS, EOS,
padding and unknown token at ID 0; Step 4.4 preserves that pinned state rather
than inventing or replacing special tokens. Qwen model-output IDs 151669 through
151935 have no tokenizer entry and are rejected explicitly if demanded rather
than silently discarded.

The inherited class-to-marker table remains only as a parity fallback. The
approved operational interface supplies the validated profile markers directly,
which supports the exact Qwen and Granite runtime classes without broadening the
profile family list.

Real Coordinator DTW execution remains blocked while the operational sparse-
target path and integrated execution are incomplete. Steps 4.3 and 4.4 add no
live-round service behavior.
