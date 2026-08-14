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

Real Coordinator DTW execution remains blocked while persistent vocabulary
mapping, Qwen–Granite tokenizer validation and the operational sparse-target path
are incomplete.
