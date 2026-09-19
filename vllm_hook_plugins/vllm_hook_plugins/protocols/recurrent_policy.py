"""Model-neutral adaptive-depth policy and capability contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, FrozenSet, Literal, Mapping, Optional, Sequence


class Readout(str, Enum):
    """Signals that a stopping policy may consume."""

    CONTRACTION = "contraction"
    DISPLACEMENT = "displacement"
    COSINE = "cosine"
    ACCELERATION = "acceleration"
    ENTROPY = "entropy"
    ENTROPY_DELTA = "entropy_delta"
    CONFIDENCE = "confidence"
    MARGIN = "margin"
    TOP1_STABILITY = "top1_stability"
    PREDICTIVE_KL = "predictive_kl"
    NATIVE_GATE = "native_gate"


PREDICTION_READOUTS = frozenset(
    {
        Readout.ENTROPY,
        Readout.ENTROPY_DELTA,
        Readout.CONFIDENCE,
        Readout.MARGIN,
        Readout.TOP1_STABILITY,
        Readout.PREDICTIVE_KL,
    }
)

# Readout -> (metric name in ConvergenceState.metrics, comparison against threshold).
# token in pass with a metric ``lt`` exits when the metric drops below the threshold, 
# ``ge`` when it reaches it. ``Readout.CONTRACTION`` is derived from hidden_delta history, so it has no entry.
READOUT_METRICS: dict[Readout, tuple[str, Literal["lt", "ge"]]] = {
    Readout.DISPLACEMENT: ("normalized_displacement", "lt"),
    Readout.COSINE: ("cosine_distance", "lt"),
    Readout.ACCELERATION: ("normalized_acceleration", "lt"),
    Readout.ENTROPY: ("entropy", "lt"),
    Readout.ENTROPY_DELTA: ("entropy_delta", "lt"),
    Readout.CONFIDENCE: ("top1_confidence", "ge"),
    Readout.MARGIN: ("logit_margin", "ge"),
    Readout.TOP1_STABILITY: ("top1_stability", "ge"),
    Readout.PREDICTIVE_KL: ("predictive_kl", "lt"),
    Readout.NATIVE_GATE: ("native_gate", "ge"),
}

# Derived contraction readout: relative remaining path length under a geometric
# bound. Mirrors RecurrentConvergenceAnalyzer._contraction_hit.
CONTRACTION_COLUMN = "contraction_rel_remaining"

def readout_column(readout: Readout) -> str:
    if readout is Readout.CONTRACTION:
        return CONTRACTION_COLUMN
    return READOUT_METRICS[readout][0]


def readout_direction(readout: Readout) -> str:
    if readout is Readout.CONTRACTION:
        return "lt"
    return READOUT_METRICS[readout][1]


@dataclass(frozen=True)
class MetricCondition:
    """One calibrated threshold applied to one model-neutral readout."""

    readout: Readout
    threshold: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MetricCondition":
        data = dict(value)
        data["readout"] = Readout(data["readout"])
        data["threshold"] = float(data["threshold"])
        return cls(**data)


@dataclass(frozen=True)
class StoppingPolicyConfig:
    """Checkpoint-calibrated stopping policy, separate from model mechanics."""

    enabled: bool = False
    conditions: tuple[MetricCondition, ...] = ()
    combine: Literal["all", "any"] = "all"
    confirmation_conditions: tuple[MetricCondition, ...] = ()
    confirmation_combine: Literal["all", "any"] = "all"
    min_steps: int = 1
    patience: int = 1
    calibration_profile: Optional[str] = None

    def __post_init__(self) -> None:
        if self.min_steps < 1:
            raise ValueError("policy.min_steps must be at least 1")
        if self.patience < 1:
            raise ValueError("policy.patience must be at least 1")
        if self.combine not in {"all", "any"}:
            raise ValueError("policy.combine must be 'all' or 'any'")
        if self.confirmation_combine not in {"all", "any"}:
            raise ValueError("policy.confirmation_combine must be 'all' or 'any'")
        if self.enabled and not self.conditions:
            raise ValueError("enabled policy requires at least one condition")

    @staticmethod
    def _parse_conditions(
        values: Sequence[Mapping[str, Any] | MetricCondition],
    ) -> tuple[MetricCondition, ...]:
        return tuple(
            value
            if isinstance(value, MetricCondition)
            else MetricCondition.from_dict(value)
            for value in values
        )

    @classmethod
    def from_dict(cls, value: Optional[Mapping[str, Any]]) -> "StoppingPolicyConfig":
        if value is None:
            return cls()
        data = dict(value)

        # Backwards compatibility for the original one-readout policy shape.
        readout = data.pop("readout", None)
        threshold = data.pop("threshold", None)
        confirmation_readout = data.pop("confirmation_readout", None)
        confirmation_threshold = data.pop("confirmation_threshold", None)
        if "conditions" not in data and readout is not None:
            if threshold is None:
                raise ValueError("policy.threshold is required with policy.readout")
            data["conditions"] = [
                {"readout": readout, "threshold": threshold}
            ]
        if (
            "confirmation_conditions" not in data
            and confirmation_readout is not None
        ):
            if confirmation_threshold is None:
                raise ValueError(
                    "policy.confirmation_threshold is required with "
                    "confirmation_readout"
                )
            data["confirmation_conditions"] = [
                {
                    "readout": confirmation_readout,
                    "threshold": confirmation_threshold,
                }
            ]

        data["conditions"] = cls._parse_conditions(data.get("conditions", ()))
        data["confirmation_conditions"] = cls._parse_conditions(
            data.get("confirmation_conditions", ())
        )
        return cls(**data)

    @property
    def readouts(self) -> FrozenSet[Readout]:
        return frozenset(
            condition.readout
            for condition in self.conditions + self.confirmation_conditions
        )

    @property
    def readout_key(self) -> str:
        primary_separator = "&" if self.combine == "all" else "|"
        primary = primary_separator.join(
            condition.readout.value for condition in self.conditions
        )
        if not self.confirmation_conditions:
            return primary or "none"
        confirmation_separator = (
            "&" if self.confirmation_combine == "all" else "|"
        )
        confirmation = confirmation_separator.join(
            condition.readout.value
            for condition in self.confirmation_conditions
        )
        return f"{primary}+confirm:{confirmation}"


@dataclass(frozen=True)
class RecurrentAdapterCapabilities:
    """Readouts supplied by one recurrent-model adapter."""

    readouts: FrozenSet[Readout]

    def validate(self, policy: StoppingPolicyConfig) -> None:
        if not policy.enabled:
            return
        missing = policy.readouts.difference(self.readouts)
        if missing:
            names = ", ".join(sorted(readout.value for readout in missing))
            raise ValueError(f"adapter does not provide policy readout(s): {names}")


CONTRACTION_CAPABILITIES = RecurrentAdapterCapabilities(
    readouts=frozenset({Readout.CONTRACTION})
)


@dataclass(frozen=True)
class CalibrationManifest:
    """Portable, versioned identity for a calibrated stopping policy."""

    schema_version: int
    checkpoint_revision: str
    recurrence_cap: int
    precision: str
    domain: str
    policy: StoppingPolicyConfig

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise ValueError("schema_version must be at least 1")
        if self.recurrence_cap < 1:
            raise ValueError("recurrence_cap must be at least 1")
        if not self.checkpoint_revision or not self.precision or not self.domain:
            raise ValueError(
                "checkpoint_revision, precision, and domain must be non-empty"
            )

    @property
    def key(self) -> tuple[str, str, int, str, str]:
        return (
            self.checkpoint_revision,
            self.policy.readout_key,
            self.recurrence_cap,
            self.precision,
            self.domain,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for field_name in ("conditions", "confirmation_conditions"):
            data["policy"][field_name] = [
                {
                    "readout": condition.readout.value,
                    "threshold": condition.threshold,
                }
                for condition in getattr(self.policy, field_name)
            ]
        return data

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationManifest":
        data = dict(value)
        data["policy"] = StoppingPolicyConfig.from_dict(data["policy"])
        return cls(**data)
