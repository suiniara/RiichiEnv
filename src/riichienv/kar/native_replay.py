from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from riichienv import Action, ActionType, RiichiEnv, convert

from .kan_transactions import COMMITTED_KAN_KINDS, pending_kan_commits
from .observation import PublicSnapshot

NATIVE_REPLAY_VERSION = "prefix_flags_bridge_v3_native_furiten"


class NativeReplayInterfaceError(RuntimeError):
    pass


class NativeReplayContextError(ValueError):
    pass


def _seat(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 4:
        raise NativeReplayContextError("Expected a four-player actor")
    return value


def _copy_action(action: Action) -> Action:
    return Action(action.action_type, action.tile, list(action.consume_tiles), action.actor)


def _event_key(event: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        event["type"],
        event.get("actor"),
        event.get("target"),
        event.get("pai"),
        tuple(event.get("consumed", ())),
    )


def _tile_name(value: str) -> str:
    return convert.tid_to_mjai(convert.mjai_to_tid(value))


class NativeReplayBridge:
    def __init__(self, env: RiichiEnv):
        required = (
            "ippatsu_cycle",
            "double_riichi_declared",
            "missed_agari_doujun",
            "missed_agari_riichi",
            "is_rinshan_flag",
            "preview_kan_responses",
            # Also rejects an older extension that discards the no-yaku result
            # in its MJAI Dahai handler, before any sample can be exported.
            "_record_replay_kan_ron_declines",
        )
        missing = [name for name in required if not hasattr(env, name)]
        if missing:
            raise NativeReplayInterfaceError(
                "Native replay context interfaces are missing; rebuild the RiichiEnv extension: " + ", ".join(missing)
            )
        self.env = env
        if len(env.riichi_declared) != 4:
            raise NativeReplayContextError("NativeReplayBridge requires a four-player environment")
        self._ippatsu = list(env.ippatsu_cycle)
        self._double = list(env.double_riichi_declared)
        for flags in (self._ippatsu, self._double, env.missed_agari_doujun, env.missed_agari_riichi):
            if len(flags) != 4 or any(type(value) is not bool for value in flags):
                raise NativeReplayInterfaceError("Native context flags must contain four booleans")
        self._rinshan = env.is_rinshan_flag
        self._pending_double: dict[int, bool] = {}
        self._interrupted = False
        self._replacement_actor: int | None = None
        self._offer_kind: str | None = None
        self._kan_responses: dict[int, list[Action]] | None = None
        self._preview_identity: tuple[int, ActionType, str, tuple[str, ...]] | None = None
        self._prepared: tuple[Any, ...] | None = None

    @property
    def kan_responses(self) -> dict[int, list[Action]] | None:
        if self._kan_responses is None:
            return None
        return {player: [_copy_action(action) for action in actions] for player, actions in self._kan_responses.items()}

    def preview_kan(self, actor: int, action: Action) -> dict[int, list[Action]]:
        actor = _seat(actor)
        if self._prepared is not None:
            raise NativeReplayContextError("Kan preview must precede prepare and native event application")
        if action.tile is None:
            raise NativeReplayContextError("A complete kan declaration must identify a tile")
        responses = {
            player: self.env.preview_kan_responses(actor, action, player) for player in range(4) if player != actor
        }
        self._kan_responses = responses
        self._preview_identity = (
            actor,
            action.action_type,
            convert.tid_to_mjai(action.tile),
            tuple(sorted(convert.tid_to_mjai(tile) for tile in action.consume_tiles)),
        )
        result = self.kan_responses
        assert result is not None
        return result

    def _sync_context(self) -> None:
        # Furiten lives exclusively in native state. MJAI Dahai can set it
        # even with no Ron offer; writing cached flags here would erase it.
        self.env.ippatsu_cycle = self._ippatsu.copy()
        self.env.double_riichi_declared = self._double.copy()
        self.env.is_rinshan_flag = self._rinshan

    def _reset(self) -> None:
        self._ippatsu = [False] * 4
        self._double = [False] * 4
        self._pending_double.clear()
        self._interrupted = False
        self._replacement_actor = None
        self._offer_kind = None
        self._kan_responses = None
        self._preview_identity = None
        self._rinshan = False

    def _commit_kan_responses(self, actor: int) -> None:
        if self._replacement_actor != actor or self._offer_kind != "kan" or self._kan_responses is None:
            raise NativeReplayContextError("Kan commitment lacks its prefix response window")
        declined = sorted(
            seat for seat, actions in self._kan_responses.items()
            if any(action.action_type == ActionType.RON for action in actions)
        )
        if declined:
            self.env._record_replay_kan_ron_declines(actor, declined)
        self._offer_kind = None
        self._kan_responses = None
        self._preview_identity = None
        self._interrupt()

    def _interrupt(self) -> None:
        self._ippatsu = [False] * 4
        self._interrupted = True

    def _check_preview(self, event: Mapping[str, Any]) -> None:
        if self._preview_identity is None or self._kan_responses is None:
            raise NativeReplayContextError("Validated kan preview is required before applying a declaration")
        actor, kind, tile, consumed = self._preview_identity
        expected_kind = ActionType.ANKAN if event["type"] == "ankan" else ActionType.KAKAN
        if actor != _seat(event["actor"]) or kind != expected_kind:
            raise NativeReplayContextError("Kan preview belongs to a different declaration")
        if expected_kind == ActionType.KAKAN:
            if _tile_name(event["pai"]) != tile:
                raise NativeReplayContextError("Kan preview tile differs from the observed declaration")
        elif tuple(sorted(_tile_name(value) for value in event["consumed"])) != consumed:
            raise NativeReplayContextError("Kan preview tiles differ from the observed declaration")

    def prepare(self, event: Mapping[str, Any], snapshots_before: Sequence[PublicSnapshot]) -> None:  # noqa: PLR0915
        if self._prepared is not None:
            raise NativeReplayContextError("after_apply must complete the preceding prepared event")
        snapshots = {snapshot.observer_player_id: snapshot for snapshot in snapshots_before}
        if len(snapshots_before) != 4 or set(snapshots) != set(range(4)):
            raise NativeReplayContextError("Four distinct observer snapshots are required")
        pending_actors = {snapshot.pending_kan_actor for snapshot in snapshots.values()}
        if len(pending_actors) != 1:
            raise NativeReplayContextError("Observer snapshots disagree on the public pending kan")
        pending_actor = pending_actors.pop()
        typ = event["type"]
        if typ in {"start_game", "start_kyoku"}:
            self._reset()
        elif typ == "reach":
            actor = _seat(event["actor"])
            snapshot = snapshots[actor]
            if not snapshot.history_complete:
                raise NativeReplayContextError("Complete public history is required to infer double riichi")
            if actor in self._pending_double or snapshot.riichi_accepted[actor]:
                raise NativeReplayContextError("Duplicate riichi declaration")
            self._pending_double[actor] = (
                not snapshot.discard_history[actor] and not self._interrupted and not any(snapshot.melds)
            )
        elif typ == "reach_accepted":
            actor = _seat(event["actor"])
            if actor not in self._pending_double:
                raise NativeReplayContextError("Riichi acceptance lacks its prefix declaration")
            self._double[actor] = self._pending_double.pop(actor)
            self._ippatsu[actor] = True
        elif typ == "dahai":
            actor = _seat(event["actor"])
            if self._offer_kind is not None:
                raise NativeReplayContextError("Previous response window has not resolved")
            self._ippatsu[actor] = False
            self._rinshan = False
        elif typ in {"chi", "pon", "kan", "daiminkan"}:
            if self._offer_kind == "kan" or pending_actor is not None:
                raise NativeReplayContextError("A kan robbery window cannot resolve as an ordinary call")
            self._offer_kind = None
            self._interrupt()
            self._replacement_actor = _seat(event["actor"]) if typ in {"kan", "daiminkan"} else None
            self._kan_responses = None
            self._preview_identity = None
            self._rinshan = False
        elif typ in {"ankan", "kakan"}:
            self._check_preview(event)
            self._replacement_actor = _seat(event["actor"])
        elif typ in {"dora", "tsumo"}:
            commits = pending_actor is not None and pending_kan_commits(
                event,
                actor=pending_actor,
                committed_kans=sum(
                    meld.kind in COMMITTED_KAN_KINDS for player in snapshots[0].melds for meld in player
                ),
                indicators=len(snapshots[0].dora_indicators),
            )
            if commits:
                assert pending_actor is not None
                self._commit_kan_responses(pending_actor)
            elif pending_actor is None and self._offer_kind == "kan":
                raise NativeReplayContextError("Kan response window and observer commitment state disagree")
            if typ == "tsumo":
                actor = _seat(event["actor"])
                self._offer_kind = None
                if self._replacement_actor is not None and self._replacement_actor != actor:
                    raise NativeReplayContextError("Replacement draw belongs to another player")
                self._rinshan = self._replacement_actor == actor
                self._replacement_actor = None
        elif typ in {"hora", "ryukyoku"}:
            self._offer_kind = None
        elif typ in {"end_kyoku", "end_game"}:
            self._offer_kind = None
            self._kan_responses = None
            self._preview_identity = None
            self._replacement_actor = None
        self._sync_context()
        self._prepared = _event_key(event)

    def after_apply(self, event: Mapping[str, Any]) -> None:
        if self._prepared != _event_key(event):
            raise NativeReplayContextError("after_apply must match the prepared public event")
        typ = event["type"]
        self._sync_context()
        if typ == "dahai":
            self._offer_kind = "discard"
        elif typ in {"ankan", "kakan"}:
            assert self._kan_responses is not None
            self._offer_kind = "kan"
        self._prepared = None
