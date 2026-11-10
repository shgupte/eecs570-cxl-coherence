"""Simulator parameters.

Embedding width is not stored in the Meta access files. The defaults match the
row size used by the earlier CXL traces in ``eecs570-cxl-coherence``: 128
elements of 4 bytes. They are assumptions, not measurements.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SimConfig:
    """Parameters shared by the row-transfer and TorchRec pooled models.

    Ownership is chosen per run. The default command uses contiguous row blocks.
    """

    num_gpus: int = 8
    embedding_dim: int = 128
    bytes_per_element: int = 4
    request_bytes: int = 8
    track_versions: bool = False
    collect_rows: bool = False

    def __post_init__(self) -> None:
        if self.num_gpus < 1:
            raise ValueError("num_gpus must be positive")
        if self.embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        if self.bytes_per_element < 1:
            raise ValueError("bytes_per_element must be positive")
        if self.request_bytes < 0:
            raise ValueError("request_bytes must be non-negative")

    @property
    def embedding_bytes(self) -> int:
        return self.embedding_dim * self.bytes_per_element

    @property
    def gradient_bytes(self) -> int:
        return self.embedding_bytes

    def to_dict(self) -> dict:
        data = asdict(self)
        data["embedding_bytes"] = self.embedding_bytes
        data["gradient_bytes"] = self.gradient_bytes
        return data
