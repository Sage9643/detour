"""Text embedding models.

The rest of the system depends only on the `Embedder` protocol: L2-normalized float32
vectors, so cosine similarity is a dot product.

Default model: WordLlama `l2_supercat` (256-d). It is a pretrained *static* embedding
model: token embeddings distilled from large LLM embedding tables and trained
contrastively for sentence similarity; a sentence vector is the pooled token vectors.
Chosen because (D-010):
- its weights and tokenizer ship inside the PyPI wheel (16 MB), so no model download is
  needed at build or run time (Hugging Face was unreachable from the development
  environment, and Render's free tier has tight memory);
- it needs only numpy + tokenizers (no PyTorch / ONNX runtime);
- ~1-2 ms per batch of short texts on CPU.
Tradeoff: static embeddings ignore word order and are weaker than transformer sentence
encoders. Any model implementing `Embedder` can replace it; cached vectors are keyed by
`model_name`, so models never mix.
"""

from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

Vectors = npt.NDArray[np.float32]


class Embedder(Protocol):
    model_name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Return an (n, dim) float32 array of L2-normalized vectors."""
        ...


def l2_normalize(x: npt.NDArray[np.floating]) -> Vectors:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    out: Vectors = (x / norms).astype(np.float32)
    return out


class WordLlamaEmbedder:
    model_name = "wordllama-l2_supercat-256"
    dim = 256

    def __init__(self) -> None:
        import wordllama
        from safetensors import safe_open
        from tokenizers import Tokenizer
        from wordllama.inference import WordLlamaInference

        # Load the files bundled in the wheel directly. WordLlama.load() would try to
        # fetch the tokenizer from Hugging Face even though it is shipped locally.
        pkg = Path(wordllama.__file__).parent
        tokenizer = Tokenizer.from_file(
            str(pkg / "tokenizers" / "l2_supercat_tokenizer_config.json")
        )
        with safe_open(str(pkg / "weights" / "l2_supercat_256.safetensors"), framework="np") as f:
            table = f.get_tensor("embedding.weight")
        if table.shape[1] != self.dim:
            raise RuntimeError(f"unexpected WordLlama dimension {table.shape}")
        self._model = WordLlamaInference(table, tokenizer)

    def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        raw = self._model.embed(list(texts), norm=False, return_np=True)
        return l2_normalize(np.asarray(raw))


@lru_cache(maxsize=1)
def default_embedder() -> WordLlamaEmbedder:
    """Process-wide singleton (loading takes ~0.1 s)."""
    return WordLlamaEmbedder()
