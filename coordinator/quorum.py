from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrustedClientQuorumPolicy:
    minimum: int = 2
    override: int | None = None

    def __post_init__(self) -> None:
        if type(self.minimum) is not int or self.minimum < 1:
            raise ValueError("minimum trusted Client quorum must be positive")
        if self.override is not None and (
            type(self.override) is not int or self.override < 1
        ):
            raise ValueError("trusted Client quorum override must be positive")

    def resolve(self, selected_client_count: int) -> int:
        if type(selected_client_count) is not int or selected_client_count < 1:
            raise ValueError("selected Client count must be positive")

        if self.override is not None:
            if self.override > selected_client_count:
                raise ValueError(
                    "trusted Client quorum override exceeds the selected Client count"
                )
            return self.override

        quorum = max(self.minimum, selected_client_count // 2 + 1)
        if quorum > selected_client_count:
            raise ValueError(
                "selected Client count is below the minimum trusted Client quorum"
            )
        return quorum

