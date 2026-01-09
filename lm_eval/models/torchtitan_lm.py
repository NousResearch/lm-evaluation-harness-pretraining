"""
TorchTitan model integration for lm-evaluation-harness.

This module provides direct integration with torchtitan's modeling,
bypassing HuggingFace transformers for inference.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import Collator

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

eval_logger = logging.getLogger(__name__)


@register_model("torchtitan", "tt")
class TorchTitanLM(TemplateLM):
    """
    LM class for running evaluations directly on torchtitan models.

    Supports llama3 architecture models loaded from HuggingFace safetensors
    or standard PyTorch checkpoints.
    """

    def __init__(
        self,
        pretrained: str,
        tokenizer_path: str | None = None,
        model_name: str = "llama3",
        model_flavor: str = "8B",
        device: str = "cuda",
        dtype: str = "bfloat16",
        batch_size: int = 1,
        max_seq_len: int = 4096,
        torchtitan_path: str | None = None,
        **kwargs,
    ) -> None:
        """
        Initialize TorchTitan model for evaluation.

        Args:
            pretrained: Path to model checkpoint (HF safetensors dir or .pt file)
            tokenizer_path: Path to tokenizer directory. If None, uses pretrained path.
            model_name: Model architecture name (e.g., "llama3", "qwen3")
            model_flavor: Model size flavor (e.g., "8B", "70B")
            device: Device to run on ("cuda", "cpu", etc.)
            dtype: Data type ("bfloat16", "float16", "float32")
            batch_size: Batch size for inference
            max_seq_len: Maximum sequence length
            torchtitan_path: Path to torchtitan installation. Auto-detected if None.
        """
        super().__init__()

        # Setup torchtitan path
        if torchtitan_path is None:
            # Try to find torchtitan relative to this file or in common locations
            possible_paths = [
                "/home/phuc/workspace/moe/online_evals/torchtitan",
                os.path.join(os.path.dirname(__file__), "../../../../torchtitan"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    torchtitan_path = os.path.abspath(path)
                    break

        if torchtitan_path and torchtitan_path not in sys.path:
            sys.path.insert(0, torchtitan_path)

        # Import torchtitan components
        self._import_torchtitan(model_name)

        # Setup device and dtype
        self._device = torch.device(device)
        self._dtype = getattr(torch, dtype)
        self._batch_size = int(batch_size)
        self._max_seq_len = max_seq_len

        # Build model
        eval_logger.info(f"Building {model_name} model with flavor {model_flavor}")
        self.model_args = self._get_model_args(model_name, model_flavor)
        self.model_args.max_seq_len = max_seq_len
        self.model = self._build_model(model_name)
        self.model.eval()

        # Load checkpoint
        eval_logger.info(f"Loading checkpoint from {pretrained}")
        self._load_checkpoint(pretrained, model_name)

        # Move to device
        self.model = self.model.to(device=self._device, dtype=self._dtype)

        # Load tokenizer
        tokenizer_path = tokenizer_path or pretrained
        eval_logger.info(f"Loading tokenizer from {tokenizer_path}")
        self._load_tokenizer(tokenizer_path)

        self.backend = "causal"
        self._rank = 0
        self._world_size = 1

    def _import_torchtitan(self, model_name: str):
        """Import torchtitan modules based on model name."""
        if model_name == "llama3":
            from torchtitan.models.llama3 import Transformer, TransformerModelArgs
            from torchtitan.models.llama3 import llama3_args
            self._transformer_cls = Transformer
            self._model_args_cls = TransformerModelArgs
            self._model_configs = llama3_args
            self._state_dict_adapter_cls = None
            try:
                from torchtitan.models.llama3 import Llama3StateDictAdapter
                self._state_dict_adapter_cls = Llama3StateDictAdapter
            except ImportError:
                pass
        elif model_name == "qwen3":
            from torchtitan.models.qwen3 import Qwen3Model, Qwen3ModelArgs
            from torchtitan.models.qwen3 import qwen3_args
            self._transformer_cls = Qwen3Model
            self._model_args_cls = Qwen3ModelArgs
            self._model_configs = qwen3_args
            self._state_dict_adapter_cls = None
            try:
                from torchtitan.models.qwen3 import Qwen3StateDictAdapter
                self._state_dict_adapter_cls = Qwen3StateDictAdapter
            except ImportError:
                pass
        else:
            raise ValueError(f"Unsupported model: {model_name}. Supported: 'llama3', 'qwen3'")

    def _get_model_args(self, model_name: str, model_flavor: str):
        """Get model args for the specified flavor."""
        if model_flavor not in self._model_configs:
            available = list(self._model_configs.keys())
            raise ValueError(f"Unknown model flavor '{model_flavor}'. Available: {available}")
        return self._model_configs[model_flavor]

    def _build_model(self, model_name: str):
        """Build the model from args."""
        return self._transformer_cls(self.model_args)

    def _load_checkpoint(self, checkpoint_path: str, model_name: str):
        """Load checkpoint from file or directory."""
        if os.path.isdir(checkpoint_path):
            # Check if it's a DCP (Distributed Checkpoint) format
            if self._is_dcp_checkpoint(checkpoint_path):
                self._load_dcp_checkpoint(checkpoint_path)
            else:
                # HuggingFace safetensors directory
                self._load_hf_checkpoint(checkpoint_path, model_name)
        else:
            # Standard PyTorch checkpoint
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)

    def _is_dcp_checkpoint(self, checkpoint_dir: str) -> bool:
        """Check if the directory contains a DCP (Distributed Checkpoint) format checkpoint."""
        # DCP checkpoints have a .metadata file and .distcp files
        has_metadata = os.path.exists(os.path.join(checkpoint_dir, ".metadata"))
        has_distcp = any(f.endswith(".distcp") for f in os.listdir(checkpoint_dir))
        # Also check it's not an HF checkpoint (no safetensors)
        has_safetensors = any(f.endswith(".safetensors") for f in os.listdir(checkpoint_dir))
        return has_metadata and has_distcp and not has_safetensors

    def _load_dcp_checkpoint(self, checkpoint_dir: str):
        """Load a torchtitan DCP (Distributed Checkpoint) format checkpoint."""
        import torch.distributed.checkpoint as dcp

        eval_logger.info(f"Loading DCP checkpoint from {checkpoint_dir}")

        # torchtitan saves model weights without "model." prefix directly
        # e.g., tok_embeddings.weight, layers.0.attention.wq.weight, etc.
        # So we load directly into the model's state dict
        state_dict = self.model.state_dict()

        try:
            # Try loading with the newer API
            dcp.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
            )
        except TypeError:
            # Fallback for older PyTorch versions
            dcp.load_state_dict(
                state_dict=state_dict,
                storage_reader=dcp.FileSystemReader(checkpoint_dir),
            )

        # Load the model state dict
        self.model.load_state_dict(state_dict)
        eval_logger.info(f"Successfully loaded DCP checkpoint")

    def _load_hf_checkpoint(self, checkpoint_dir: str, model_name: str):
        """Load HuggingFace safetensors checkpoint with state dict conversion."""
        from safetensors import safe_open

        # Find safetensor files
        safetensor_files = [
            f for f in os.listdir(checkpoint_dir)
            if f.endswith(".safetensors")
        ]

        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")

        # Load all safetensor files
        eval_logger.info(f"Loading {len(safetensor_files)} safetensor files...")
        hf_state_dict = {}
        for filename in safetensor_files:
            filepath = os.path.join(checkpoint_dir, filename)
            with safe_open(filepath, framework="pt", device="cpu") as f:
                for key in f.keys():
                    hf_state_dict[key] = f.get_tensor(key)

        eval_logger.info(f"Loaded {len(hf_state_dict)} tensors from checkpoint")

        # Convert HF state dict to torchtitan format using the adapter
        if self._state_dict_adapter_cls is not None:
            eval_logger.info(f"Converting state dict using {self._state_dict_adapter_cls.__name__}")
            adapter = self._state_dict_adapter_cls(self.model_args, checkpoint_dir)
            state_dict = adapter.from_hf(hf_state_dict)
        else:
            eval_logger.warning("No state dict adapter available, using HF state dict directly")
            state_dict = hf_state_dict

        self.model.load_state_dict(state_dict)

    def _load_tokenizer(self, tokenizer_path: str):
        """Load tokenizer from path."""
        from torchtitan.components.tokenizer import HuggingFaceTokenizer
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path)

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def prefix_token_id(self) -> int:
        if self.tokenizer.bos_id is not None:
            return self.tokenizer.bos_id
        return self.tokenizer.eos_id

    @property
    def max_length(self) -> int:
        return self._max_seq_len

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self) -> int:
        return self._world_size

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        **kwargs,
    ) -> list[int]:
        """Tokenize a string."""
        # torchtitan tokenizer handles BOS/EOS based on config
        return self.tokenizer.encode(string)

    def tok_decode(self, tokens: list[int], skip_special_tokens: bool = True) -> str:
        """Decode tokens to string."""
        return self.tokenizer.decode(tokens)

    def _model_call(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            input_ids: [batch, seq_len] token ids

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        with torch.no_grad():
            with torch.autocast(device_type=self._device.type, dtype=self._dtype):
                logits = self.model(input_ids)  # [batch, seq, vocab]
        return logits

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str], list[int], list[int]]],
        disable_tqdm: bool = False,
        override_bs: int | None = None,
    ) -> list[tuple[float, bool]]:
        """
        Compute log-likelihood for continuation tokens.

        Args:
            requests: List of ((context_str, continuation_str), context_tokens, continuation_tokens)

        Returns:
            List of (log_prob, is_greedy) tuples
        """
        res = []

        def _collate(req):
            toks = req[1] + req[2]
            return -len(toks), tuple(toks)

        re_ord = Collator(requests, sort_fn=_collate, group_by=None)
        chunks = re_ord.get_batched(n=override_bs or self._batch_size)

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self.rank != 0),
            desc="Running loglikelihood requests",
        )

        for chunk in chunks:
            # Prepare batch
            inps = []
            cont_toks_list = []
            inplens = []

            for _, context_enc, continuation_enc in chunk:
                # Truncate from left if too long
                inp = (context_enc + continuation_enc)[-(self._max_seq_len + 1):-1]
                inps.append(inp)
                cont_toks_list.append(continuation_enc)
                inplens.append(len(inp))

            # Pad to same length (right padding)
            max_len = max(len(inp) for inp in inps)
            padded_inps = []
            for inp in inps:
                padded = inp + [self.eot_token_id] * (max_len - len(inp))
                padded_inps.append(padded)

            # Convert to tensor
            input_ids = torch.tensor(padded_inps, dtype=torch.long, device=self._device)

            # Forward pass
            logits = self._model_call(input_ids)  # [batch, seq, vocab]
            log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq, vocab]

            # Extract results for each item in batch
            for idx, ((_, ctx_tokens, _), cont_toks, inplen) in enumerate(
                zip(chunk, cont_toks_list, inplens)
            ):
                contlen = len(cont_toks)
                ctx_len = len(ctx_tokens)

                # Get continuation logits (shifted by 1 for next-token prediction)
                # Context ends at ctx_len, continuation spans ctx_len to ctx_len+contlen
                # But we predict token at position i from logits at position i-1
                cont_start = ctx_len - 1
                cont_end = ctx_len - 1 + contlen

                cont_log_probs = log_probs[idx, cont_start:cont_end, :]  # [contlen, vocab]
                cont_tokens_tensor = torch.tensor(cont_toks, device=self._device)

                # Gather log probs for actual continuation tokens
                token_log_probs = cont_log_probs[
                    torch.arange(contlen, device=self._device), cont_tokens_tensor
                ]
                total_log_prob = token_log_probs.sum().item()

                # Check if greedy
                greedy_tokens = cont_log_probs.argmax(dim=-1)
                is_greedy = torch.all(greedy_tokens == cont_tokens_tensor).item()

                res.append((total_log_prob, is_greedy))
                pbar.update(1)

        pbar.close()
        return re_ord.get_original(res)

    def loglikelihood_rolling(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        """Compute rolling log-likelihood for perplexity."""
        loglikelihoods = []

        for (string,) in tqdm(
            [req.args for req in requests],
            disable=disable_tqdm or (self.rank != 0),
            desc="Running rolling loglikelihood",
        ):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            # Process windows
            windows = [(None,) + x for x in rolling_token_windows]
            results = self._loglikelihood_tokens(windows, disable_tqdm=True)

            # Sum up log likelihoods
            total_ll = sum(r[0] for r in results)
            loglikelihoods.append(total_ll)

            self.cache_hook.add_partial("loglikelihood_rolling", (string,), total_ll)

        return loglikelihoods

    def generate_until(
        self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        """Generate text until stop sequences."""
        res = []

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm or (self.rank != 0),
            desc="Running generate_until requests",
        )

        for req in requests:
            context, gen_kwargs = req.args

            # Get generation parameters
            until = gen_kwargs.get("until", [])
            if isinstance(until, str):
                until = [until]
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)

            # Encode context
            context_enc = self.tok_encode(context)

            # Truncate if needed
            max_ctx_len = self.max_length - max_gen_toks
            if len(context_enc) > max_ctx_len:
                context_enc = context_enc[-max_ctx_len:]

            input_ids = torch.tensor([context_enc], dtype=torch.long, device=self._device)

            # Simple greedy/sampling generation
            generated = self._generate(
                input_ids,
                max_new_tokens=max_gen_toks,
                temperature=temperature,
            )

            # Decode only new tokens
            new_tokens = generated[0, len(context_enc):].tolist()
            text = self.tok_decode(new_tokens)

            # Handle stop sequences
            for stop_seq in until:
                if stop_seq in text:
                    text = text[:text.index(stop_seq)]

            res.append(text)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text)
            pbar.update(1)

        pbar.close()
        return res

    @torch.no_grad()
    def _generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Simple token-by-token generation.

        Args:
            input_ids: [batch, seq_len] input token ids
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            [batch, seq_len + max_new_tokens] generated token ids
        """
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Check if we've exceeded max length
            if generated.shape[1] >= self._max_seq_len:
                break

            # Forward pass
            logits = self._model_call(generated)
            next_token_logits = logits[:, -1, :]  # [batch, vocab]

            if temperature <= 0:
                # Greedy decoding
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Sample with temperature
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if EOS token generated
            if next_token.item() == self.eot_token_id:
                break

        return generated
