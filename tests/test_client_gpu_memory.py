from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from client.gpu_memory import ClientGpuMemory, MIB, client_memory_budget
from client.model_profiles import QWEN_PROFILE_ID, pinned_client_profile
from client.peft_backend import TransformersPeftTrainingBackend, _answer_only_trainer_class
from client.reverse_training import (
    TransformersPeftReverseBackend, _selective_client_trainer_class,
    selective_client_hidden_state_loss, selective_client_loss,
)
from client.training import TrainingExecutionProfile, memory_efficient_execution_profile
from shared.answer_only import AnswerOnlyCollator, bounded_answer_only_loss


class ClientMemoryPolicyTests(unittest.TestCase):
    def test_target_gpu_budgets_leave_headroom_and_respect_other_processes(self):
        for total, windows, expected in ((6113, True, 5089), (8192, False, 7168)):
            with self.subTest(total=total):
                budget = client_memory_budget(
                    total_bytes=total * MIB, free_bytes=total * MIB,
                    reserved_bytes=0, windows=windows,
                )
                self.assertEqual(budget.limit_bytes, expected * MIB)
                busy = client_memory_budget(
                    total_bytes=total * MIB, free_bytes=2000 * MIB,
                    reserved_bytes=1000 * MIB, windows=windows,
                )
                self.assertEqual(busy.limit_bytes, 2744 * MIB)

    def test_busy_gpu_is_rejected_without_an_allocation(self):
        with self.assertRaisesRegex(RuntimeError, "insufficient free VRAM"):
            client_memory_budget(total_bytes=6113 * MIB, free_bytes=200 * MIB,
                                 reserved_bytes=0, windows=True)

    def test_cuda_profile_is_idempotent_and_preserves_effective_batch(self):
        original = TrainingExecutionProfile(
            backend="transformers", device="cuda", precision="bfloat16",
            micro_batch_size=2, gradient_accumulation_steps=4,
        )
        effective = memory_efficient_execution_profile(original)
        self.assertEqual(effective.micro_batch_size, 1)
        self.assertEqual(effective.gradient_accumulation_steps, 8)
        self.assertTrue(effective.gradient_checkpointing)
        self.assertEqual(memory_efficient_execution_profile(effective), effective)
        self.assertFalse(original.gradient_checkpointing)
        self.assertNotEqual(original.profile_hash(), effective.profile_hash())

    def test_cpu_profile_and_old_serialized_profile_are_not_rewritten(self):
        original = TrainingExecutionProfile(
            backend="transformers", device="cpu", precision="float32",
        )
        self.assertIs(memory_efficient_execution_profile(original), original)
        self.assertEqual(TrainingExecutionProfile.model_validate_json(
            original.model_dump_json()).profile_hash(), original.profile_hash())

    def test_cuda_reference_streaming_cannot_be_disabled_by_stale_environment(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            "CLIENT_KNOWLEDGE_BATCH_SIZE": "4", "CLIENT_KNOWLEDGE_SEQUENCE_CHUNK_SIZE": "0",
        }):
            backend = TransformersPeftTrainingBackend(
                data_dir=directory, model_profile=pinned_client_profile(QWEN_PROFILE_ID),
                execution_profile=TrainingExecutionProfile(
                    backend="transformers", device="cuda", precision="bfloat16",
                ),
            )
            self.assertEqual(backend.knowledge_batch_size, 1)
            self.assertEqual(backend.knowledge_sequence_chunk_size, 64)

    def test_model_lower_bound_is_checked_and_diagnostics_survive(self):
        cuda = mock.Mock()
        cuda.mem_get_info.return_value = (6 * 1024 * MIB, 6 * 1024 * MIB)
        cuda.memory_reserved.return_value = 0
        cuda.memory_allocated.return_value = 0
        cuda.max_memory_reserved.return_value = 0
        cuda.max_memory_allocated.return_value = 0
        parameter = mock.Mock()
        parameter.numel.return_value = 3 * 1024 * MIB
        parameter.element_size.return_value = 2
        model = mock.Mock()
        model.parameters.return_value = [parameter]
        model.buffers.return_value = []
        with tempfile.TemporaryDirectory() as directory:
            memory = ClientGpuMemory(SimpleNamespace(cuda=cuda), Path(directory))
            memory.prepare()
            cuda.set_per_process_memory_fraction.assert_called_once()
            with self.assertRaisesRegex(RuntimeError, "before model transfer"):
                memory.admit_model(model)
            model.to.assert_not_called()
            events = [json.loads(line) for line in memory.path.read_text().splitlines()]
            self.assertEqual([e["phase"] for e in events], ["admission", "model_cpu_loaded"])
            cuda.mem_get_info.return_value = (100 * MIB, 6 * 1024 * MIB)
            with self.assertRaisesRegex(RuntimeError, "less than 256 MiB"):
                memory.check_headroom()


def tiny_peft_model(kind):
    import peft
    import torch
    import transformers

    config_class, model_class = {
        "qwen3": (transformers.Qwen3Config, transformers.Qwen3ForCausalLM),
        "granite": (transformers.GraniteConfig, transformers.GraniteForCausalLM),
    }[kind]
    config = config_class(
        vocab_size=37, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, pad_token_id=0,
        attention_dropout=0.0, logits_scaling=2.0,
    )
    config._attn_implementation = "sdpa"
    model = peft.get_peft_model(model_class(config), peft.LoraConfig(
        r=2, lora_alpha=4, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(std=0.02)
    return model


def training_inputs():
    import torch
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [1, 3, 5, 7, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
        "labels": torch.tensor([[-100, -100, 3, 4, 5, 6], [-100, -100, 5, 7, -100, -100]]),
    }


class BoundedRealClassTests(unittest.TestCase):
    def setUp(self):
        from accelerate.state import AcceleratorState
        environment = mock.patch.dict(os.environ, {"ACCELERATE_MIXED_PRECISION": "no"})
        environment.start()
        self.addCleanup(environment.stop)
        AcceleratorState._reset_state(reset_partial_state=True)

    def tearDown(self):
        from accelerate.state import AcceleratorState
        AcceleratorState._reset_state(reset_partial_state=True)

    def test_saved_activation_context_with_non_reentrant_checkpointing(self):
        import torch
        from shared.answer_only import client_saved_activation_context
        with mock.patch.object(torch.autograd.graph, "save_on_cpu") as offload:
            client_saved_activation_context(torch, torch.device("cuda"))
            offload.assert_called_once_with(pin_memory=True)
        torch.manual_seed(31)
        oracle = tiny_peft_model("granite").train()
        model = copy.deepcopy(oracle)
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        inputs = training_inputs()
        expected = bounded_answer_only_loss(torch, oracle, inputs)
        expected.backward()
        # Exercise the actual hooks with checkpointing on CPU too; GPU transfer
        # and peak VRAM acceptance remain checks on the physical Clients.
        with mock.patch("shared.answer_only.client_saved_activation_context",
                        side_effect=lambda torch, device: torch.autograd.graph.save_on_cpu()):
            actual = bounded_answer_only_loss(torch, model, inputs)
        actual.backward()
        torch.testing.assert_close(actual, expected)
        for (name, a), (_, b) in zip(oracle.named_parameters(), model.named_parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_actual_trainer_optimizer_and_non_reentrant_checkpointing(self):
        import torch
        import transformers
        from client.peft_backend import _attach_memory_callback
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(41)
                model = tiny_peft_model("granite").train()
                original = {n: p.detach().clone() for n, p in model.named_parameters()}
                values = training_inputs()
                if reverse:
                    from client.reverse_training import _TrimmedClientReverseTrainingCollator
                    values.update(
                        sparse_target_token_ids=torch.tensor([[[2, 3]] * 6] * 2),
                        sparse_target_probabilities=torch.tensor([[[0.7, 0.3]] * 6] * 2),
                        sparse_target_valid_mask=torch.ones(2, 6, 2, dtype=torch.bool))
                    collator = _TrimmedClientReverseTrainingCollator()
                    dataset = [{k: v[i] for k, v in values.items()} for i in range(2)]
                else:
                    collator = AnswerOnlyCollator(torch, 0)
                    dataset = [{k: v[i].tolist() for k, v in values.items()} for i in range(2)]
                cls = (_selective_client_trainer_class if reverse else _answer_only_trainer_class)(
                    transformers, torch)
                trainer = cls(model=model, train_dataset=dataset, data_collator=collator,
                              args=transformers.TrainingArguments(
                                  output_dir=directory, use_cpu=True, report_to=[], bf16=True,
                                  per_device_train_batch_size=1, gradient_accumulation_steps=2,
                                  num_train_epochs=1, learning_rate=1e-3,
                                  gradient_checkpointing=True,
                                  gradient_checkpointing_kwargs={"use_reentrant": False},
                                  save_strategy="no", remove_unused_columns=False,
                                  disable_tqdm=True))
                memory = mock.Mock()
                _attach_memory_callback(trainer, transformers, memory)
                result = trainer.train()
                phases = [call.args[0] for call in memory.phase.call_args_list]
                self.assertIn("optimizer_begin", phases)
                self.assertIn("optimizer_complete", phases)
                self.assertIn("backward_complete", phases)
                self.assertEqual(trainer.state.global_step, 1)
                self.assertTrue(torch.isfinite(torch.tensor(result.training_loss)))
                self.assertTrue(model.is_gradient_checkpointing)
                changed = []
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        changed.append(not torch.equal(parameter, original[name]))
                    else:
                        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
                self.assertTrue(any(changed))

    def test_actual_qwen_granite_peft_losses_and_gradients_with_checkpointing(self):
        import torch
        for kind in ("qwen3", "granite"):
            for dtype in (torch.float32, torch.bfloat16):
                for reverse in (False, True):
                    with self.subTest(kind=kind, dtype=dtype, reverse=reverse):
                        torch.manual_seed(17)
                        oracle = tiny_peft_model(kind).to(dtype=dtype).train()
                        bounded = copy.deepcopy(oracle)
                        bounded.gradient_checkpointing_enable(
                            gradient_checkpointing_kwargs={"use_reentrant": False})
                        inputs = training_inputs()
                        inputs.update(
                            sparse_target_token_ids=torch.tensor([[[2, 3]] * 6] * 2),
                            sparse_target_probabilities=torch.tensor([[[0.7, 0.3]] * 6] * 2),
                            sparse_target_valid_mask=torch.ones(2, 6, 2, dtype=torch.bool),
                        )
                        with torch.autocast("cpu", dtype=torch.bfloat16,
                                            enabled=dtype == torch.bfloat16):
                            logits = oracle(input_ids=inputs["input_ids"],
                                            attention_mask=inputs["attention_mask"],
                                            use_cache=False).logits.float()
                            if reverse:
                                expected = selective_client_loss(torch, logits, inputs)[0]
                            else:
                                expected = torch.nn.functional.cross_entropy(
                                    logits[:, :-1].reshape(-1, 37),
                                    inputs["labels"][:, 1:].reshape(-1), ignore_index=-100)
                        expected.backward()
                        widths = []
                        head = bounded.get_base_model().get_output_embeddings()
                        hook = head.register_forward_pre_hook(
                            lambda module, args: widths.append(args[0].shape[1]))
                        with torch.autocast("cpu", dtype=torch.bfloat16,
                                            enabled=dtype == torch.bfloat16):
                            if reverse:
                                actual = selective_client_hidden_state_loss(
                                    torch, bounded, inputs, sequence_chunk_size=2)[0]
                            else:
                                actual = bounded_answer_only_loss(
                                    torch, bounded, inputs, sequence_chunk_size=2)
                        actual.backward()
                        hook.remove()
                        tolerance = 0.004 if dtype == torch.bfloat16 else 2e-6
                        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
                        self.assertEqual(actual.dtype, torch.float32)
                        self.assertLessEqual(max(widths), 2)
                        for (name, a), (_, b) in zip(oracle.named_parameters(), bounded.named_parameters()):
                            if a.requires_grad:
                                self.assertIsNotNone(b.grad, name)
                                torch.testing.assert_close(a.grad, b.grad, atol=tolerance, rtol=tolerance)
                            else:
                                self.assertIsNone(b.grad, name)

    def test_trainer_accumulation_matches_manual_gradients(self):
        import torch
        import transformers
        for reverse in (False, True):
            with self.subTest(reverse=reverse), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(23)
                oracle = tiny_peft_model("granite").train()
                model = copy.deepcopy(oracle)
                inputs = training_inputs()
                inputs.update(
                    sparse_target_token_ids=torch.tensor([[[2, 3]] * 6] * 2),
                    sparse_target_probabilities=torch.tensor([[[0.7, 0.3]] * 6] * 2),
                    sparse_target_valid_mask=torch.ones(2, 6, 2, dtype=torch.bool),
                )
                batches = [{k: v[i:i+1] for k, v in inputs.items()} for i in range(2)]
                denominator = inputs["labels"][:, 1:].ne(-100).sum()
                for batch in batches:
                    logits = oracle(input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"], use_cache=False).logits.float()
                    if reverse:
                        loss = selective_client_loss(torch, logits, batch)[0] / 2
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            logits[:, :-1].reshape(-1, 37), batch["labels"][:, 1:].reshape(-1),
                            ignore_index=-100, reduction="sum") / denominator
                    loss.backward()
                cls = (_selective_client_trainer_class if reverse else _answer_only_trainer_class)(
                    transformers, torch)
                trainer = cls(model=model, args=transformers.TrainingArguments(
                    output_dir=directory, use_cpu=True, report_to=[],
                    gradient_accumulation_steps=2, remove_unused_columns=False,
                ))
                trainer.current_gradient_accumulation_steps = 2
                count = trainer._get_num_items_in_batch(batches, torch.device("cpu"))
                for batch in batches:
                    trainer.training_step(model, batch, num_items_in_batch=count)
                for (name, a), (_, b) in zip(oracle.named_parameters(), model.named_parameters()):
                    if a.requires_grad:
                        torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-5)

    def test_reverse_validation_passes_streaming_to_real_inference(self):
        import torch
        from tests.test_client_reverse_training import reverse_fixture
        from shared.fedmkt_core.ml.logit_generation import generate_pub_data_logits
        from shared.reference_dataset import ReferenceSample
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime, _, job = reverse_fixture(root)
            backend = TransformersPeftReverseBackend(
                data_dir=root, model_profile=runtime.model_profile,
                execution_profile=TrainingExecutionProfile(
                    backend="transformers", device="cpu", precision="float32"),
            )
            backend.knowledge_sequence_chunk_size = 2
            model = tiny_peft_model("qwen3").eval()
            for p in model.parameters():
                p.requires_grad = False
            sample_id = job.public_data_partition.validation_sample_ids[0]
            sample = ReferenceSample(schema_version=1, dataset_id="reverse-reference",
                                     dataset_version="v1", sample_id=sample_id,
                                     chapter="Contracts", section="1", question="Q", gold_answer="A")
            tokenizer = SimpleNamespace(pad_token_id=0, decode=lambda ids, **kw: str(ids))
            item = SimpleNamespace(sample_id=sample_id, input_ids=[1, 2, 3, 4],
                                   labels=[-100, -100, 3, 4], attention_mask=[1, 1, 1, 1])
            with mock.patch.object(backend, "_load_base_model", return_value=model), \
                 mock.patch("client.reverse_training.resolve_alignment_profile",
                            return_value=SimpleNamespace(client=backend.model_profile)), \
                 mock.patch("client.reverse_training.load_pinned_tokenizer",
                            return_value=SimpleNamespace(tokenizer=tokenizer)), \
                 mock.patch("client.reverse_training.encode_reference_samples", return_value=[item]), \
                 mock.patch("shared.fedmkt_core.ml.logit_generation.generate_pub_data_logits",
                            wraps=generate_pub_data_logits) as generate:
                record = backend._evaluate_adapter(
                    job=job, validation_samples=[sample], adapter_role="parent",
                    adapter_version=0, checkpoint_hash="a" * 64, checkpoint_path=None)
            self.assertEqual(generate.call_args.kwargs["sequence_chunk_size"], 2)
            self.assertEqual(record.sample_count, 1)


if __name__ == "__main__":
    unittest.main()
