import time
from dataclasses import dataclass

# Injected per-batch stall in the broken baseline collate_fn.
# The fast tokenizers library makes raw tokenization ~2-5ms on M4 Pro,
# which is insufficient to produce a visible DataLoader bottleneck against
# a ~300-600ms GPU step. This constant ensures load_frac ≥ 30% at baseline
# so the profiler can attribute the stall and demonstrate a measurable fix.
_BASELINE_COLLATE_SLEEP_S = 0.1  # 100ms per batch → ~10% of a 1s step at baseline

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

from forge.metrics import compute_mfu


@dataclass
class HarnessConfig:
    num_workers: int = 0
    pin_memory: bool = False
    dtype: str = "fp32"           # "fp32" or "bfloat16"
    batch_size: int = 8
    tokenize_offline: bool = False
    n_steps: int = 100
    seq_len: int = 128
    model_name: str = "gpt2"


BASELINE_CONFIG = HarnessConfig(
    num_workers=0,
    pin_memory=False,
    dtype="fp32",
    batch_size=8,
    tokenize_offline=False,
    n_steps=100,
)


@dataclass
class StepProfile:
    step_idx: int
    load_time: float       # seconds: time waiting for next batch from DataLoader
    transfer_time: float   # seconds: CPU→MPS memory copy
    forward_time: float    # seconds: model forward pass
    backward_time: float   # seconds: loss.backward()
    optimizer_time: float  # seconds: optimizer.step()
    total_time: float      # seconds: wall clock for full step
    tokens: int            # batch_size × seq_len


def _build_loader(config: HarnessConfig, tokenizer: GPT2Tokenizer) -> DataLoader:
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    raw = raw.filter(lambda x: len(x["text"].strip()) > config.seq_len)

    if config.tokenize_offline:
        def _tokenize(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=config.seq_len,
                padding="max_length",
            )

        dataset = raw.map(_tokenize, batched=True, remove_columns=["text"])
        dataset.set_format("torch", columns=["input_ids"])

        def _collate_offline(batch):
            ids = torch.stack([b["input_ids"] for b in batch])
            return {"input_ids": ids, "labels": ids.clone()}

        collate_fn = _collate_offline

    else:
        dataset = raw

        def _collate_online(batch):
            # Intentionally slow: tokenize from raw text on every batch fetch
            # and sleep to guarantee a measurable DataLoader stall.
            # _BASELINE_COLLATE_SLEEP_S makes load_time dominate step_time so
            # the GPU idle % is clearly attributable in profiler output.
            texts = [b["text"] for b in batch]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=config.seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            time.sleep(_BASELINE_COLLATE_SLEEP_S)
            input_ids: torch.Tensor = enc["input_ids"]  # type: ignore[assignment]
            return {"input_ids": input_ids, "labels": input_ids.clone()}

        collate_fn = _collate_online

    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )


def run_training(config: HarnessConfig = BASELINE_CONFIG) -> list[StepProfile]:
    """Run GPT-2 Small for config.n_steps steps. Returns per-step wall-clock profiles.

    Each phase boundary is fenced with torch.mps.synchronize() to ensure the GPU
    has drained before we record the timestamp. Without synchronize(), MPS command
    submission is asynchronous and timing would measure submission, not execution.
    """
    device = torch.device("mps")
    tokenizer = GPT2Tokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(config.model_name)
    model.to(device)  # type: ignore[arg-type]
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    loader = _build_loader(config, tokenizer)
    data_iter = iter(loader)
    profiles: list[StepProfile] = []

    for step in range(config.n_steps):
        # --- load ---
        t0 = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        t1 = time.perf_counter()

        # --- transfer ---
        input_ids = batch["input_ids"].to(device, non_blocking=config.pin_memory)
        labels = batch["labels"].to(device, non_blocking=config.pin_memory)
        torch.mps.synchronize()
        t2 = time.perf_counter()

        # --- forward ---
        optimizer.zero_grad()
        if config.dtype == "bfloat16":
            with torch.autocast("mps", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss
        torch.mps.synchronize()
        t3 = time.perf_counter()

        # --- backward ---
        loss.backward()
        torch.mps.synchronize()
        t4 = time.perf_counter()

        # --- optimizer ---
        optimizer.step()
        torch.mps.synchronize()
        t5 = time.perf_counter()

        profiles.append(StepProfile(
            step_idx=step,
            load_time=t1 - t0,
            transfer_time=t2 - t1,
            forward_time=t3 - t2,
            backward_time=t4 - t3,
            optimizer_time=t5 - t4,
            total_time=t5 - t0,
            tokens=config.batch_size * config.seq_len,
        ))

        # Live MFU display — real-time, not post-hoc
        if (step + 1) % 10 == 0 or step == 0:
            window = profiles[-10:]
            rolling_tps = sum(s.tokens for s in window) / sum(s.total_time for s in window)
            mfu = compute_mfu(rolling_tps)
            print(
                f"  step {step+1:4d}/{config.n_steps} | "
                f"{rolling_tps:6.0f} tok/s | MFU {mfu*100:.1f}% | "
                f"load {profiles[-1].load_time*1000:.0f}ms",
                end="\r",
                flush=True,
            )

    print()  # clear the \r line
    return profiles
