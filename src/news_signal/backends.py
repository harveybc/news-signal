"""Pinned SDK integration; fixture is explicitly not an ML measurement."""

from importlib import metadata
import hashlib
import json
import os
from pathlib import Path

from .core import QUESTIONS, Refusal, canonical, digest

SDK_COMMIT = "1e28ac20c0896b1c37a744cd11f740eb98f8b178"


def same_gpu(observed, declared):
    """CUDA_VISIBLE_DEVICES needs the `GPU-` form; `torch.cuda.get_device_properties().uuid` prints the bare form. Comparing
    the two literally can never match on real hardware, so the prefix is normalised away and the rest must be identical."""
    def bare(value):
        text = str(value).strip().lower()
        return text[4:] if text.startswith("gpu-") else text
    return bool(observed) and bool(declared) and bare(observed) == bare(declared)


def file_digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def seal_checkpoint(directory):
    root = Path(directory).resolve(strict=True)
    if not root.is_dir():
        raise Refusal("CHECKPOINT_DIRECTORY_REQUIRED")
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise Refusal("CHECKPOINT_SYMLINK: materialize the snapshot first")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": file_digest(path)}
    if not {"model.safetensors", "rl_agent_config.json", "encoder/config.json"} <= set(files) or not any(p.startswith("tokenizer/") for p in files):
        raise Refusal("CHECKPOINT_INCOMPLETE")
    result = {"schema": "news_checkpoint.v1", "files": files}
    result["sha256"] = digest(result)
    return result


def verify_checkpoint(directory, manifest):
    if manifest != seal_checkpoint(directory):
        raise Refusal("CHECKPOINT_CHANGED")
    return manifest


class FixtureBackend:
    identity = {"kind": "NON_MODEL_FIXTURE", "name": "always-unclear-v1"}

    def predict(self, state, questions=None):
        questions = QUESTIONS if questions is None else questions
        return {"answers": {k: {"type": "choice", "choice": "unclear", "probabilities": {
            label: float(label == "unclear") for label in q["criteria"]
        }} for k, q in questions.items()}}


class LayaBackend:
    @classmethod
    def from_agent(cls, agent, identity, expected_device):
        if str(agent.device) != expected_device:
            raise Refusal("DEVICE_FALLBACK_FORBIDDEN")
        obj = cls.__new__(cls)
        obj.agent, obj.identity = agent, identity
        obj.expected_device = expected_device
        return obj

    def __init__(self, directory, manifest, device="cpu", gpu_uuid=None):
        if device not in {"cpu", "cuda:0"}:
            raise Refusal("EXPLICIT_CPU_OR_SINGLE_CUDA_DEVICE_REQUIRED")
        if device == "cuda:0" and (not gpu_uuid or not gpu_uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid):
            raise Refusal("PHYSICAL_GPU_UUID_MASK_REQUIRED")
        verify_checkpoint(directory, manifest)
        dist = metadata.distribution("laya")
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("vcs_info", {}).get("commit_id") != SDK_COMMIT:
            raise Refusal("PINNED_SDK_INSTALL_REQUIRED")
        # Libraries may cache these flags at import time.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from laya import Agent
        torch.set_num_threads(2)
        if device == "cuda:0":
            if not torch.cuda.is_available():
                raise Refusal("GPU_UUID_NOT_OBSERVED: no CUDA device is visible to this process")
            observed = str(torch.cuda.get_device_properties(0).uuid)
            if not same_gpu(observed, gpu_uuid):
                raise Refusal(f"GPU_UUID_NOT_OBSERVED: the visible device is {observed}, not {gpu_uuid}")
        # Offline snapshot includes encoder/config and tokenizer; never resolve a moving Hub ID.
        self.agent = Agent(str(Path(directory).resolve()), device=device, fast=False, compile=False)
        self.expected_device = device
        if str(self.agent.device) != device:
            raise Refusal("DEVICE_FALLBACK_FORBIDDEN")
        verify_checkpoint(directory, manifest)
        self.identity = {
            "kind": "LAYA_LOCAL_CHECKPOINT", "sdk_version": dist.version,
            "sdk_commit": SDK_COMMIT, "checkpoint_sha256": manifest["sha256"],
            "device": str(self.agent.device), "gpu_uuid": gpu_uuid if device != "cpu" else None,
            "gpu_uuid_observed": (str(torch.cuda.get_device_properties(0).uuid) if device == "cuda:0" else None),
            "gpu_uuid_attribution": ("MEASURED" if device == "cuda:0" else "NOT_APPLICABLE"),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "adapter_sha256": file_digest(Path(__file__)),
        }

    def predict(self, state, questions=None):
        # The question set is the model's input, not a filter applied afterwards: Laya encodes question, options and text
        # together. A caller that names no task gets the set this adapter shipped with.
        questions = QUESTIONS if questions is None else questions
        # Conservative cap below the pinned 512/192 state budget; never silently truncate news.
        tokens = self.agent.tok.encode(state, add_special_tokens=False)
        if len(tokens) > 256:
            raise Refusal("TOKEN_BUDGET_EXCEEDED: select a documented short news field, do not silently truncate")
        result = self.agent.system_one(state, questions, max_len=512, head_max_len=192)
        if str(self.agent.device) != self.expected_device:
            raise Refusal("DEVICE_FALLBACK_FORBIDDEN")
        return result
