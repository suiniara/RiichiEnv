from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..calibration import CalibrationArtifact, calibrated_policy_identity
from ..contracts import content_hash
from ..policy import PolicyOptions
from ..release import RELEASE_VERSION, implementation_release_identity
from ..runtime import RuntimePolicy
from ..runtime_contracts import RuntimeBinding
from ..selection_quality import TAU_SELECTION_VERSION, validate_finite_tau_selection
from ..support import EventSupport
from ..versioning import artifact_contract, validate_artifact_contract
from .config import ModelConfig
from .inference import FrozenModelPredictor, model_weights_sha256
from .model import KARTransformer
from .precision import FrozenBackendPrecision


def frozen_export_plan(
    checkpoint: Path,
    *,
    device: str,
    policy: dict[str, Any],
    calibration_job: dict[str, Any] | None = None,
    tau_selection_job: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pointer = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    record = {
        "version": "frozen_release_export_plan_v1",
        "checkpoint_sha256": pointer["checkpoint_sha256"],
        "release_contract_sha256": pointer["metadata"]["release_contract"]["release_contract_sha256"],
        "device": device,
        "policy": policy,
        "calibration_job_sha256": (calibration_job or {}).get("job_sha256"),
        "tau_selection_job_sha256": (tau_selection_job or {}).get("selection_job_sha256"),
    }
    for document, key in ((calibration_job, "job_sha256"), (tau_selection_job, "selection_job_sha256")):
        if document is not None:
            unsigned = {name: value for name, value in document.items() if name != key}
            if document.get(key) != content_hash(unsigned):
                raise ValueError("Export input job artifact was modified")
    return {**record, "plan_sha256": content_hash(record)}


def validate_frozen_policy(
    options: PolicyOptions,
    binding: RuntimeBinding,
    support: EventSupport | None,
    calibration: CalibrationArtifact | None,
    selection: dict[str, Any] | None,
    *,
    inference_device_type: str,
) -> None:
    if calibration is not None:
        if support is None:
            raise ValueError("A calibrated policy requires its support artifact")
        calibration.validate(binding, support)
        if calibration.inference_device_type != inference_device_type:
            raise ValueError("Calibration originated on another backend; explicit recalibration is required")
    if options.mode == "prior_only" or math.isinf(options.tau):
        return
    if options.diagnostic_argmax:
        raise ValueError("A diagnostic argmax policy is not a validation-selected stochastic deployment")
    if calibration is None or support is None or not calibration.gate_valid or not calibration.empirical_check_valid:
        raise ValueError("Finite deployment requires passed independent calibration-quality checks")
    if not any(support.value_supported(calibration.value_branch_statuses)):
        raise ValueError("Finite deployment requires at least one train-supported, quality-qualified response branch")
    expected = calibrated_policy_identity(
        binding, support, calibration, mode=options.mode, max_prior_amplification=options.max_prior_amplification
    )
    if (
        selection is None
        or selection.get("version") != TAU_SELECTION_VERSION
        or selection.get("diagnostic_fixture") is not False
    ):
        raise ValueError("Missing production validation-policy selection")
    validate_finite_tau_selection(selection, expected_game_ids=calibration.split.fit_games)
    unsigned = {key: value for key, value in selection.items() if key != "tau_selection_sha256"}
    if (
        selection.get("tau_selection_sha256") != content_hash(unsigned)
        or selection.get("binding_sha256") != binding.digest
        or selection.get("policy_identity") != expected
    ):
        raise ValueError("Tau selection belongs to different calibration, support, backend or sampling options")
    if (
        float(selection["selected_tau"]) != options.tau
        or selection.get("max_prior_amplification") != options.max_prior_amplification
    ):
        raise ValueError("Frozen tau or amplification limit changed after validation")
    if (
        selection.get("validation_manifest_sha256") != calibration.split.validation_manifest_sha256
        or selection.get("calibration_split_salt") != calibration.split.salt
    ):
        raise ValueError("Tau selection uses another validation partition")


def save_frozen_release(
    directory: Path,
    model: KARTransformer,
    *,
    train_manifest_sha256: str,
    use_rule_features: bool,
    training_contract: dict[str, Any],
    support: EventSupport | None = None,
    calibration: CalibrationArtifact | None = None,
    options: PolicyOptions | None = None,
    tau_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_artifact_contract(training_contract, context="frozen training contract")
    predictor = FrozenModelPredictor(model, train_manifest_sha256, use_rule_features=use_rule_features)
    options = options or PolicyOptions()
    validate_frozen_policy(
        options,
        predictor.binding,
        support,
        calibration,
        tau_selection,
        inference_device_type=next(model.parameters()).device.type,
    )
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "weights.pt"
    torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, weights)
    policy = asdict(options)
    policy["tau"] = "infinity" if math.isinf(options.tau) else options.tau
    record = {
        **artifact_contract(),
        "release_version": RELEASE_VERSION,
        "model_config": asdict(model.config),
        "binding": predictor.binding.to_record(),
        "model_config_hash": model.config.contract_hash,
        "inference_device_type": next(model.parameters()).device.type,
        "backend_precision": predictor.backend_precision.to_record(),
        "use_rule_features": use_rule_features,
        "implementation": implementation_release_identity(),
        "weights_file_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "training_contract": training_contract,
        "support": support.to_record() if support is not None else None,
        "calibration": calibration.to_record() if calibration is not None else None,
        "policy": policy,
        "tau_min": 3 / math.log(options.max_prior_amplification)
        if options.max_prior_amplification is not None
        else None,
        "tau_selection": tau_selection,
        "evidence_status": "empirical_only" if options.mode == "empirical_only" else "certificate_unavailable",
        "note": "No optimizer; calibration metrics and 500-game counts are not error-bound certificates",
    }
    record["release_sha256"] = content_hash(record)
    (directory / "release.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return record


def load_training_model(checkpoint: Path, *, device: str = "cpu") -> tuple[KARTransformer, dict[str, Any]]:
    from .production_checkpoint import read_production_checkpoint  # noqa: PLC0415

    pointer = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf-8"))
    metadata = pointer["metadata"]
    contract = metadata["release_contract"]
    current = implementation_release_identity()
    for key in ("source_sha256", "native_extension_sha256"):
        if contract["implementation"][key] != current[key]:
            raise ValueError("Inference export needs the trained source/native build or an explicit migration")
    payload = read_production_checkpoint(checkpoint, contract=contract, runtime=metadata["runtime"])
    model = KARTransformer(ModelConfig(**metadata["model_config"])).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, metadata


class FrozenRelease:
    def __init__(self, directory: Path, *, device: str = "cpu"):
        self.directory = directory
        record = json.loads((directory / "release.json").read_text(encoding="utf-8"))
        unsigned = {key: value for key, value in record.items() if key != "release_sha256"}
        if record.get("release_version") != RELEASE_VERSION or record.get("release_sha256") != content_hash(unsigned):
            raise ValueError("Frozen release identity differs")
        validate_artifact_contract(record, context="frozen release")
        validate_artifact_contract(record.get("training_contract", {}), context="frozen training contract")
        current = implementation_release_identity()
        if any(record["implementation"][key] != current[key] for key in ("source_sha256", "native_extension_sha256")):
            raise ValueError("Frozen policy source/native build changed; explicit migration is required")
        path = directory / "weights.pt"
        if (
            not path.is_file()
            or path.stat().st_size > 2 * 1024**3
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["weights_file_sha256"]
        ):
            raise ValueError("Frozen weights are missing/modified/oversized")
        if record.get("calibration") is not None and torch.device(device).type != record["inference_device_type"]:
            raise ValueError(
                "A calibrated release cannot silently change CPU/CUDA backend; recalibrate for the target backend"
            )
        self.model = KARTransformer(ModelConfig(**record["model_config"])).to(device).eval()
        self.model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
        probe = FrozenModelPredictor(
            self.model, record["binding"]["train_manifest_sha256"], use_rule_features=record["use_rule_features"]
        )
        if (
            probe.binding.to_record() != record["binding"]
            or self.model.config.contract_hash != record["model_config_hash"]
            or probe.backend_precision != FrozenBackendPrecision.from_record(record.get("backend_precision", {}))
        ):
            raise ValueError("Loaded weights/configuration/features/backend do not match the frozen policy")
        self.support = (
            EventSupport.from_record(record["support"], probe.binding) if record["support"] is not None else None
        )
        self.calibration = None
        if record["calibration"] is not None:
            if self.support is None:
                raise ValueError("Missing event support")
            self.calibration = CalibrationArtifact.from_record(record["calibration"], probe.binding, self.support)
        policy = dict(record["policy"])
        policy["tau"] = math.inf if policy["tau"] == "infinity" else policy["tau"]
        self.options = PolicyOptions(**policy)
        expected_min = (
            3 / math.log(self.options.max_prior_amplification)
            if self.options.max_prior_amplification is not None
            else None
        )
        if record.get("tau_min") != expected_min:
            raise ValueError("R_max and recorded tau_min disagree")
        validate_frozen_policy(
            self.options,
            probe.binding,
            self.support,
            self.calibration,
            record.get("tau_selection"),
            inference_device_type=next(self.model.parameters()).device.type,
        )
        self.record = record
        self.initial_weight_hash = model_weights_sha256(self.model)

    def policy(self, *, seed: int, baseline: bool = False) -> RuntimePolicy:
        options = (
            PolicyOptions(mode="prior_only", tau=math.inf, max_prior_amplification=self.options.max_prior_amplification)
            if baseline
            else self.options
        )
        predictor = FrozenModelPredictor(
            self.model,
            self.record["binding"]["train_manifest_sha256"],
            use_rule_features=self.record["use_rule_features"],
        )
        if predictor.binding.to_record() != self.record["binding"]:
            raise ValueError("A cached release model changed")
        return RuntimePolicy(predictor, options=options, support=self.support, calibration=self.calibration, seed=seed)
