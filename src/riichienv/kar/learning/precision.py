"""Read-only contracts for the arithmetic used by frozen FP32 inference.

Metadata parsing deliberately does not import Torch. Runtime checks read its
public backend APIs and never change process-wide backend preferences.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, cast

FROZEN_BACKEND_PRECISION_VERSION = "frozen_fp32_backend_v1"
_SDPA_NAMES = {"cpu": ("flash", "math"), "cuda": ("flash", "mem_efficient", "math", "cudnn")}


@dataclass(frozen=True)
class FrozenBackendPrecision:
    device_type: str
    torch_version: str
    matmul_precision: str
    sdpa_backends: tuple[str, ...]
    deterministic_algorithms: bool
    mkldnn_enabled: bool | None
    blas_library: str | None

    def __post_init__(self) -> None:
        if self.device_type not in _SDPA_NAMES or self.matmul_precision != "ieee":
            raise ValueError("Frozen inference requires a CPU/CUDA IEEE FP32 backend contract")
        if not isinstance(self.torch_version, str) or not self.torch_version:
            raise ValueError("Frozen backend contract needs its explicit Torch version")
        order = _SDPA_NAMES[self.device_type]
        if not self.sdpa_backends or self.sdpa_backends != tuple(name for name in order if name in self.sdpa_backends):
            raise ValueError("Frozen backend contract needs distinct, ordered enabled SDPA backends")
        if type(self.deterministic_algorithms) is not bool:
            raise ValueError("Frozen backend contract needs an explicit deterministic-algorithm setting")
        if self.device_type == "cpu":
            if type(self.mkldnn_enabled) is not bool or self.blas_library is not None:
                raise ValueError("CPU frozen backend contract needs its MKLDNN setting")
        elif self.mkldnn_enabled is not None or self.blas_library not in {"default", "cublas", "cublaslt", "ck"}:
            raise ValueError("CUDA frozen backend contract needs its BLAS preference")

    def to_record(self) -> dict[str, Any]:
        return {
            "version": FROZEN_BACKEND_PRECISION_VERSION,
            **asdict(self),
            "sdpa_backends": list(self.sdpa_backends),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> FrozenBackendPrecision:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"version", *cls.__dataclass_fields__}
            or record.get("version") != FROZEN_BACKEND_PRECISION_VERSION
            or not isinstance(record.get("sdpa_backends"), (list, tuple))
        ):
            raise ValueError("Missing or incompatible frozen IEEE FP32 backend contract")
        values = {key: record[key] for key in cls.__dataclass_fields__}
        values["sdpa_backends"] = tuple(values["sdpa_backends"])
        return cls(**values)


def capture_frozen_backend_precision(device_type: str) -> FrozenBackendPrecision:
    import torch  # noqa: PLC0415 -- metadata-only users need no Torch installation.

    if device_type not in _SDPA_NAMES:
        raise ValueError("Frozen IEEE FP32 inference supports CPU and CUDA only")
    setting = f"torch.backends.{'mkldnn' if device_type == 'cpu' else 'cuda'}.matmul.fp32_precision"
    try:
        # Torch exposes these public properties dynamically through PropModule;
        # the pinned Python typing stubs omit the MKLDNN matmul/BLAS enum fields.
        backend = cast(Any, torch.backends.mkldnn).matmul if device_type == "cpu" else torch.backends.cuda.matmul
        # The public getter resolves the generic/backend/operator hierarchy and
        # legacy set_float32_matmul_precision(). 'none' is the untouched IEEE
        # default in the pinned Torch 2.15 build; canonicalize it for artifacts.
        mode = backend.fp32_precision
        if mode not in {"none", "ieee"}:
            raise ValueError(
                f"Frozen inference requires IEEE FP32: {setting} is {mode!r}; "
                "configure an IEEE backend before calibration or loading the frozen policy"
            )
        return FrozenBackendPrecision(
            device_type=device_type,
            torch_version=str(torch.__version__),
            matmul_precision="ieee",
            sdpa_backends=tuple(
                name for name in _SDPA_NAMES[device_type] if getattr(torch.backends.cuda, name + "_sdp_enabled")()
            ),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            mkldnn_enabled=bool(torch.backends.mkldnn.enabled) if device_type == "cpu" else None,
            blas_library=cast(Any, torch.backends.cuda.preferred_blas_library()).name.lower()
            if device_type == "cuda"
            else None,
        )
    except (AttributeError, RuntimeError) as error:
        raise ValueError(
            f"Cannot verify the frozen IEEE FP32 backend via {setting}; use the pinned supported Torch backend"
        ) from error


def require_frozen_backend_precision(expected: FrozenBackendPrecision) -> None:
    if capture_frozen_backend_precision(expected.device_type) != expected:
        raise ValueError("Frozen IEEE FP32 backend settings changed; restore the bound backend or recalibrate")


@contextmanager
def frozen_fp32(expected: FrozenBackendPrecision) -> Iterator[None]:
    import torch  # noqa: PLC0415

    require_frozen_backend_precision(expected)
    with torch.autocast(device_type=expected.device_type, enabled=False):
        yield
    # Reject changed settings before a result can be archived or returned.
    require_frozen_backend_precision(expected)
