from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from riichienv import Action, RiichiEnv

from .observation import PublicSnapshot

NATIVE_REPLAY_VERSION = "native_replay_context_v4"


class NativeReplayInterfaceError(RuntimeError):
    pass


class NativeReplayContextError(ValueError):
    pass


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event["type"],
        event.get("actor"),
        event.get("target"),
        event.get("pai"),
        tuple(event.get("consumed", ())),
    )


class NativeReplayBridge:
    """Compatibility protocol wrapper; all Mahjong context belongs to Rust.

    Snapshots are accepted to preserve callers' event-pairing API, but never
    supply native flags, kan commit decisions, or inferred private intentions.
    """

    def __init__(self, env: RiichiEnv):
        required = (
            "replay_context_version",
            "prepare_replay_kan",
            "validate_replay_event",
            "validate_replay_draw",
            "replay_kan_responses",
            # Capability retained in the minimum-interface check for older
            # clients; the bridge no longer calls it or writes native flags.
            "_record_replay_kan_ron_declines",
        )
        # Check seats before invoking a four-player-only native property.
        if hasattr(env, "riichi_declared") and len(env.riichi_declared) != 4:
            raise NativeReplayContextError("NativeReplayBridge requires a four-player environment")
        missing = [name for name in required if not hasattr(env, name)]
        if missing:
            raise NativeReplayInterfaceError(
                "Native replay context interfaces are missing; rebuild the RiichiEnv extension: " + ", ".join(missing)
            )
        if env.replay_context_version != NATIVE_REPLAY_VERSION:
            raise NativeReplayInterfaceError("Native replay context version differs; rebuild the RiichiEnv extension")
        self.env = env
        self._prepared: tuple[Any, ...] | None = None

    @property
    def kan_responses(self) -> dict[int, list[Action]] | None:
        # PyO3 returns detached action values; neither a caller nor this wrapper
        # can overwrite the native prefix cache by modifying the returned dict.
        return self.env.replay_kan_responses

    def preview_kan(self, actor: int, action: Action) -> dict[int, list[Action]]:
        if self._prepared is not None:
            raise NativeReplayContextError("Kan preview must precede prepare and native event application")
        if type(actor) is not int or not 0 <= actor < 4:
            raise NativeReplayContextError("Expected a four-player actor")
        try:
            return self.env.prepare_replay_kan(actor, action)
        except ValueError as error:
            raise NativeReplayContextError(str(error)) from error

    def prepare(self, event: Mapping[str, Any], snapshots_before: Sequence[PublicSnapshot] | None = None) -> None:
        if self._prepared is not None:
            raise NativeReplayContextError("after_apply must complete the preceding prepared event")
        if snapshots_before is not None and (
            len(snapshots_before) != 4
            or {snapshot.observer_player_id for snapshot in snapshots_before} != set(range(4))
        ):
            raise NativeReplayContextError("Four distinct observer snapshots are required")
        try:
            self.env.validate_replay_event(dict(event))
        except ValueError as error:
            raise NativeReplayContextError(str(error)) from error
        self._prepared = _event_key(event)

    def after_apply(self, event: Mapping[str, Any]) -> None:
        if self._prepared != _event_key(event):
            raise NativeReplayContextError("after_apply must match the prepared public event")
        self._prepared = None
