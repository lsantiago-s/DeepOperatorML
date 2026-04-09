from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "DeepOperatorML"
    entity: str | None = None
    group: str | None = None
    job_type: str = "train"
    mode: str = "online"
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "WandbConfig":
        raw = dict(raw or {})
        return cls(
            enabled=bool(raw.get("enabled", False)),
            project=str(raw.get("project", "DeepOperatorML")),
            entity=raw.get("entity"),
            group=raw.get("group"),
            job_type=str(raw.get("job_type", "train")),
            mode=str(raw.get("mode", "online")),
            name=raw.get("name"),
            tags=list(raw.get("tags", [])),
            notes=raw.get("notes"),
        )
