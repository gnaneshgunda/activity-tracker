"""Small Language Model wrapper for edge deployment -- used by B8 (router)
and B9 (exemplar narration).

Model choice: Phi-3 Mini 3.8B (Microsoft, MIT licence).
  - 3.8B params, ~2.3 GB in Q4_K_M quantisation
  - Runs on CPU via llama-cpp-python; no GPU required
  - Instruction-tuned: follows the structured prompts in router.py and
    exemplars.py without fine-tuning
  - Fits comfortably on a Raspberry Pi 5 (8 GB) or Jetson Orin Nano

Alternative: Gemma-2 2B (google/gemma-2-2b-it-GGUF) -- smaller, slightly
lower quality on structured output.  Swap MODEL_REPO / MODEL_FILE below.

Download (one-time, ~2.3 GB):
    python3 -m models.slm --download

Inference:
    from models.slm import get_slm
    slm = get_slm()
    print(slm("What activity is this?"))
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

__all__ = [
    "SLMConfig",
    "SLM",
    "get_slm",
    "download_model",
    "MODEL_DIR",
    "DEFAULT_MODEL_FILE",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------

#: Directory where GGUF model files are stored.
MODEL_DIR = Path(os.environ.get("ACTIVITY_TRACKER_MODEL_DIR", "models/weights"))

#: Hugging Face repo and filename for Phi-3 Mini Q4_K_M.
MODEL_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
DEFAULT_MODEL_FILE = "Phi-3-mini-4k-instruct-q4.gguf"

#: Fallback: Gemma-2 2B (smaller, swap if Phi-3 is too large for target device)
_GEMMA_REPO = "bartowski/gemma-2-2b-it-GGUF"
_GEMMA_FILE = "gemma-2-2b-it-Q4_K_M.gguf"


class SLMConfig:
    """Runtime configuration for the SLM.

    Attributes
    ----------
    model_path:
        Absolute path to the GGUF file.  Defaults to
        ``MODEL_DIR / DEFAULT_MODEL_FILE``.
    n_ctx:
        Context window in tokens.  2048 is enough for the router prompt
        (~200 tokens) and the exemplar narration prompt (~800 tokens).
    n_threads:
        CPU threads for inference.  ``None`` = llama.cpp auto-detect.
    max_tokens:
        Maximum tokens to generate per call.  Router needs ~5; narration
        needs ~150.  Set conservatively to keep latency low at edge.
    temperature:
        Sampling temperature.  0.0 = greedy (deterministic), best for the
        router's one-word classification task.  0.3 for narration.
    verbose:
        Pass ``True`` to see llama.cpp progress bars (useful during first
        load; noisy in production).
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        n_ctx: int = 2048,
        n_threads: Optional[int] = None,
        max_tokens: int = 200,
        temperature: float = 0.1,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path) if model_path else MODEL_DIR / DEFAULT_MODEL_FILE
        self.n_ctx = n_ctx
        self.n_threads = n_threads or max(1, (os.cpu_count() or 4) - 1)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.verbose = verbose


class SLM:
    """Thin wrapper around ``llama_cpp.Llama`` that exposes a ``(str) -> str``
    callable interface compatible with ``RouterConfig.slm_fn`` and
    ``exemplars.explain()``.

    The model is loaded lazily on first call so import time stays fast.
    """

    def __init__(self, config: Optional[SLMConfig] = None) -> None:
        self._cfg = config or SLMConfig()
        self._llm = None  # lazy load

    def _load(self) -> None:
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for the SLM. Install with:\n"
                "  pip install llama-cpp-python\n"
                "For CPU-only edge builds:\n"
                "  CMAKE_ARGS='-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS' "
                "pip install llama-cpp-python"
            ) from exc

        model_path = self._cfg.model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Download it with: python3 -m models.slm --download"
            )

        log.info("Loading SLM from %s (n_ctx=%d, threads=%d)",
                 model_path, self._cfg.n_ctx, self._cfg.n_threads)
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=self._cfg.n_ctx,
            n_threads=self._cfg.n_threads,
            verbose=self._cfg.verbose,
        )
        log.info("SLM loaded.")

    def __call__(self, prompt: str) -> str:
        """Run inference. Returns the generated text string."""
        self._load()
        out = self._llm(
            prompt,
            max_tokens=self._cfg.max_tokens,
            temperature=self._cfg.temperature,
            stop=["</s>", "<|end|>", "<|endoftext|>"],
            echo=False,
        )
        return out["choices"][0]["text"].strip()

    def for_routing(self) -> "SLM":
        """Return a copy configured for the router: greedy, short output."""
        cfg = SLMConfig(
            model_path=self._cfg.model_path,
            n_ctx=self._cfg.n_ctx,
            n_threads=self._cfg.n_threads,
            max_tokens=10,       # router only needs one word
            temperature=0.0,     # deterministic
            verbose=self._cfg.verbose,
        )
        slm = SLM(cfg)
        slm._llm = self._llm    # share the loaded model
        return slm

    def for_narration(self) -> "SLM":
        """Return a copy configured for exemplar narration: slightly creative."""
        cfg = SLMConfig(
            model_path=self._cfg.model_path,
            n_ctx=self._cfg.n_ctx,
            n_threads=self._cfg.n_threads,
            max_tokens=200,
            temperature=0.3,
            verbose=self._cfg.verbose,
        )
        slm = SLM(cfg)
        slm._llm = self._llm    # share the loaded model
        return slm


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_singleton: Optional[SLM] = None


def get_slm(config: Optional[SLMConfig] = None) -> SLM:
    """Return the module-level SLM singleton, creating it if needed.

    The model file is loaded lazily on first ``__call__``, so importing
    this module is always fast even if the weights are not present yet.
    """
    global _singleton
    if _singleton is None:
        _singleton = SLM(config)
    return _singleton


# ---------------------------------------------------------------------------
# One-time model download
# ---------------------------------------------------------------------------


def download_model(
    *,
    repo: str = MODEL_REPO,
    filename: str = DEFAULT_MODEL_FILE,
    dest_dir: Path = MODEL_DIR,
) -> Path:
    """Download the GGUF model file from Hugging Face Hub.

    Requires ``huggingface_hub``:
        pip install huggingface-hub

    Parameters
    ----------
    repo:
        HF repo id, e.g. ``"microsoft/Phi-3-mini-4k-instruct-gguf"``.
    filename:
        GGUF filename inside the repo.
    dest_dir:
        Local directory to save the file.  Created if absent.

    Returns
    -------
    Path
        Local path to the downloaded file.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface-hub is required to download the model:\n"
            "  pip install huggingface-hub"
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / filename
    if dest.exists():
        print(f"Model already present: {dest}")
        return dest

    print(f"Downloading {filename} from {repo} → {dest_dir} ...")
    path = hf_hub_download(
        repo_id=repo,
        filename=filename,
        local_dir=str(dest_dir),
    )
    print(f"Downloaded: {path}")
    return Path(path)


# ---------------------------------------------------------------------------
# CLI entry point: python3 -m models.slm --download
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description="SLM model management")
    ap.add_argument("--download", action="store_true", help="Download the default model")
    ap.add_argument("--repo", default=MODEL_REPO)
    ap.add_argument("--file", default=DEFAULT_MODEL_FILE)
    ap.add_argument("--dest", default=str(MODEL_DIR))
    ap.add_argument("--test", action="store_true", help="Run a quick inference test")
    args = ap.parse_args()

    if args.download:
        download_model(repo=args.repo, filename=args.file, dest_dir=Path(args.dest))

    if args.test:
        slm = get_slm()
        resp = slm("Reply with exactly one word: hello")
        print(f"SLM response: {repr(resp)}")
