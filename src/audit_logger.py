"""
Audit logging module for Multi-Signal Context Integration (Task 7.2).

Provides structured logging and audit trail functionality for:
- Metric refresh events
- Model training/retraining
- Deployment events
- Performance metrics tracking
- File integrity verification

Usage:
    from src.audit_logger import AuditLogger, MetricRefreshEvent, ModelTrainingEvent

    logger = AuditLogger(log_dir='logs/audit')

    # Log metric refresh
    event = MetricRefreshEvent(
        metrics=['author_influx', 'citation_velocity'],
        status='success',
        manifest_hash='abc123...'
    )
    logger.log_event(event)

    # Log model training
    event = ModelTrainingEvent(
        model_path='data/psc/out/models/msd_model.pkl',
        recall=0.87,
        precision=0.24
    )
    logger.log_event(event)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class EventType(Enum):
    """Types of audit events."""
    METRIC_REFRESH = "metric_refresh"
    FEATURE_GENERATION = "feature_generation"
    MODEL_TRAINING = "model_training"
    MODEL_DEPLOYMENT = "model_deployment"
    VALIDATION = "validation"
    ROLLBACK = "rollback"
    PIPELINE_RUN = "pipeline_run"


class EventStatus(Enum):
    """Status of audit events."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class AuditEvent:
    """Base class for audit events."""
    event_type: EventType
    timestamp: str
    status: EventStatus
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "status": self.status.value,
            "message": self.message,
            "details": self.details
        }


@dataclass
class MetricRefreshEvent(AuditEvent):
    """Audit event for metric refresh operations."""

    def __init__(
        self,
        metrics: list[str],
        status: EventStatus | str,
        manifest_hash: str | None = None,
        failed_metrics: list[str] | None = None,
        execution_time_seconds: float | None = None,
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        details = {
            "metrics_computed": metrics,
            "total_metrics": len(metrics),
            "manifest_hash": manifest_hash,
            "failed_metrics": failed_metrics or [],
            "execution_time_seconds": execution_time_seconds,
            **kwargs
        }

        message = f"Metric refresh: {len(metrics)} metrics computed"
        if failed_metrics:
            message += f", {len(failed_metrics)} failed"

        super().__init__(
            event_type=EventType.METRIC_REFRESH,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


@dataclass
class FeatureGenerationEvent(AuditEvent):
    """Audit event for feature generation operations."""

    def __init__(
        self,
        output_path: str,
        num_features: int,
        num_samples: int,
        context_features_enabled: bool,
        status: EventStatus | str,
        coverage: float | None = None,
        execution_time_seconds: float | None = None,
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        details = {
            "output_path": output_path,
            "num_features": num_features,
            "num_samples": num_samples,
            "context_features_enabled": context_features_enabled,
            "coverage": coverage,
            "execution_time_seconds": execution_time_seconds,
            **kwargs
        }

        message = f"Feature generation: {num_features} features, {num_samples} samples"
        if coverage is not None:
            message += f", {coverage:.1%} coverage"

        super().__init__(
            event_type=EventType.FEATURE_GENERATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


@dataclass
class ModelTrainingEvent(AuditEvent):
    """Audit event for model training operations."""

    def __init__(
        self,
        model_path: str,
        model_type: str,
        recall: float,
        precision: float,
        f1_score: float | None = None,
        pr_auc: float | None = None,
        status: EventStatus | str = EventStatus.SUCCESS,
        num_features: int | None = None,
        hyperparameters: dict[str, Any] | None = None,
        execution_time_seconds: float | None = None,
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        # Compute model hash
        model_hash = None
        if Path(model_path).exists():
            model_hash = _compute_file_hash(Path(model_path))

        details = {
            "model_path": model_path,
            "model_type": model_type,
            "model_hash": model_hash,
            "performance": {
                "recall": recall,
                "precision": precision,
                "f1_score": f1_score,
                "pr_auc": pr_auc
            },
            "num_features": num_features,
            "hyperparameters": hyperparameters,
            "execution_time_seconds": execution_time_seconds,
            **kwargs
        }

        message = f"Model training: {model_type}, recall={recall:.3f}, precision={precision:.3f}"

        super().__init__(
            event_type=EventType.MODEL_TRAINING,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


@dataclass
class ModelDeploymentEvent(AuditEvent):
    """Audit event for model deployment operations."""

    def __init__(
        self,
        model_path: str,
        previous_model_path: str | None,
        status: EventStatus | str,
        validation_passed: bool,
        deployed_by: str = "automated",
        git_commit: str | None = None,
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        model_hash = None
        if Path(model_path).exists():
            model_hash = _compute_file_hash(Path(model_path))

        previous_hash = None
        if previous_model_path and Path(previous_model_path).exists():
            previous_hash = _compute_file_hash(Path(previous_model_path))

        details = {
            "model_path": model_path,
            "model_hash": model_hash,
            "previous_model_path": previous_model_path,
            "previous_model_hash": previous_hash,
            "validation_passed": validation_passed,
            "deployed_by": deployed_by,
            "git_commit": git_commit,
            **kwargs
        }

        message = f"Model deployment: {model_path}"
        if not validation_passed:
            message += " (validation failed)"

        super().__init__(
            event_type=EventType.MODEL_DEPLOYMENT,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


@dataclass
class ValidationEvent(AuditEvent):
    """Audit event for validation operations."""

    def __init__(
        self,
        validation_type: str,
        status: EventStatus | str,
        checks_passed: int,
        checks_failed: int,
        errors: list[str] | None = None,
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        details = {
            "validation_type": validation_type,
            "checks_passed": checks_passed,
            "checks_failed": checks_failed,
            "total_checks": checks_passed + checks_failed,
            "errors": errors or [],
            **kwargs
        }

        message = f"Validation: {validation_type}, {checks_passed}/{checks_passed + checks_failed} passed"

        super().__init__(
            event_type=EventType.VALIDATION,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


@dataclass
class RollbackEvent(AuditEvent):
    """Audit event for rollback operations."""

    def __init__(
        self,
        rollback_target: str,
        reason: str,
        status: EventStatus | str,
        artifacts_restored: list[str],
        triggered_by: str = "automated",
        **kwargs
    ):
        if isinstance(status, str):
            status = EventStatus(status)

        details = {
            "rollback_target": rollback_target,
            "reason": reason,
            "artifacts_restored": artifacts_restored,
            "triggered_by": triggered_by,
            **kwargs
        }

        message = f"Rollback: {rollback_target} - {reason}"

        super().__init__(
            event_type=EventType.ROLLBACK,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=status,
            message=message,
            details=details
        )


class AuditLogger:
    """
    Structured audit logger for multi-signal context integration pipeline.

    Logs events to:
    - JSON files (one per event, timestamped)
    - JSONL append-only log (all events)
    - Python logging (for debugging)
    """

    def __init__(
        self,
        log_dir: str | Path = "logs/audit",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        log_level: int = logging.INFO
    ):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit log files
            enable_file_logging: Write individual JSON files per event
            enable_console_logging: Log to console via Python logging
            log_level: Logging level for console output
        """
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging

        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.jsonl_path = self.log_dir / "audit.jsonl"

        # Setup Python logger
        if self.enable_console_logging:
            self.logger = logging.getLogger("AuditLogger")
            self.logger.setLevel(log_level)

            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)

    def log_event(self, event: AuditEvent) -> None:
        """
        Log an audit event.

        Args:
            event: AuditEvent instance to log
        """
        event_dict = event.to_dict()

        # Log to console
        if self.enable_console_logging:
            log_level = logging.INFO if event.status == EventStatus.SUCCESS else logging.WARNING
            self.logger.log(log_level, f"[{event.event_type.value}] {event.message}")

        # Write individual JSON file
        if self.enable_file_logging:
            timestamp = event_dict["timestamp"].replace(":", "-").replace(".", "-")
            event_type = event_dict["event_type"]
            filename = f"{timestamp}_{event_type}.json"
            filepath = self.log_dir / filename

            with open(filepath, 'w') as f:
                json.dump(event_dict, f, indent=2)

        # Append to JSONL
        if self.enable_file_logging:
            with open(self.jsonl_path, 'a') as f:
                f.write(json.dumps(event_dict) + '\n')

    def query_events(
        self,
        event_type: EventType | None = None,
        status: EventStatus | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        limit: int | None = None
    ) -> list[dict[str, Any]]:
        """
        Query audit events from JSONL log.

        Args:
            event_type: Filter by event type
            status: Filter by status
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return

        Returns:
            List of event dictionaries matching criteria
        """
        if not self.enable_file_logging or not self.jsonl_path.exists():
            return []

        events = []

        with open(self.jsonl_path) as f:
            for line in f:
                try:
                    event = json.loads(line.strip())

                    # Apply filters
                    if event_type and event["event_type"] != event_type.value:
                        continue

                    if status and event["status"] != status.value:
                        continue

                    event_time = datetime.fromisoformat(event["timestamp"])

                    if start_time and event_time < start_time:
                        continue

                    if end_time and event_time > end_time:
                        continue

                    events.append(event)

                    if limit and len(events) >= limit:
                        break

                except json.JSONDecodeError:
                    continue

        return events

    def get_latest_deployment(self) -> dict[str, Any] | None:
        """Get the most recent deployment event."""
        deployments = self.query_events(
            event_type=EventType.MODEL_DEPLOYMENT,
            limit=1
        )
        return deployments[0] if deployments else None

    def get_performance_history(
        self,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get performance history from recent model training events.

        Returns:
            List of performance dictionaries with timestamp, recall, precision, etc.
        """
        training_events = self.query_events(
            event_type=EventType.MODEL_TRAINING,
            status=EventStatus.SUCCESS,
            limit=limit
        )

        history = []
        for event in training_events:
            perf = event["details"].get("performance", {})
            history.append({
                "timestamp": event["timestamp"],
                "recall": perf.get("recall"),
                "precision": perf.get("precision"),
                "f1_score": perf.get("f1_score"),
                "pr_auc": perf.get("pr_auc"),
                "model_type": event["details"].get("model_type")
            })

        return history


def _compute_file_hash(file_path: Path, algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    hasher = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


# Convenience function for scripts
def get_audit_logger(log_dir: str = "logs/audit") -> AuditLogger:
    """Get a configured audit logger instance."""
    return AuditLogger(log_dir=log_dir)


if __name__ == '__main__':
    # Demo usage
    logger = AuditLogger(log_dir="logs/audit")

    # Example: Log metric refresh
    event = MetricRefreshEvent(
        metrics=["author_influx", "citation_velocity", "reference_vitality"],
        status=EventStatus.SUCCESS,
        manifest_hash="abc123def456",
        execution_time_seconds=45.2
    )
    logger.log_event(event)

    # Example: Log model training
    event = ModelTrainingEvent(
        model_path="data/psc/out/models/msd_model_20251106.pkl",
        model_type="LightGBM",
        recall=0.868,
        precision=0.245,
        f1_score=0.382,
        pr_auc=0.412,
        num_features=55,
        hyperparameters={"n_estimators": 100, "max_depth": 7}
    )
    logger.log_event(event)

    print("\n[Demo] Audit logger initialized and events logged to logs/audit/")
