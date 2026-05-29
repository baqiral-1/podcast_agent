"""Per-project series state for cross-episode signals.

The pipeline runs episodes in parallel via ``asyncio.gather``. To get useful
cross-episode tic suppression without serializing writing, we maintain a
single ``series_state.json`` at the project root that each episode's audit
pass updates atomically. Later episodes' writing payloads read whatever has
landed by the time their payload is assembled (a best-effort blocklist).

Concurrency note: the orchestrator holds an ``asyncio.Lock`` around
read-modify-write. File-system atomicity is preserved by writing to a temp
file and ``os.replace`` (POSIX atomic on the same filesystem).
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from podcast_agent.schemas.models import SeriesTicEpisodeEntry, SeriesTicState


SERIES_STATE_FILENAME = "series_state.json"


def _series_state_path(project_dir: Path) -> Path:
    return project_dir / SERIES_STATE_FILENAME


def load_series_state(project_dir: Path, project_id: str) -> SeriesTicState:
    """Load series state from ``project_dir/series_state.json``.

    Returns a default ``SeriesTicState`` when the file is missing or
    unreadable. Callers should be tolerant of the first-episode case.
    """
    path = _series_state_path(project_dir)
    if not path.exists():
        return SeriesTicState(project_id=project_id)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return SeriesTicState(project_id=project_id)
    try:
        return SeriesTicState.model_validate(raw)
    except Exception:
        return SeriesTicState(project_id=project_id)


def save_series_state(project_dir: Path, state: SeriesTicState) -> None:
    """Atomically write ``state`` to ``project_dir/series_state.json``."""
    project_dir.mkdir(parents=True, exist_ok=True)
    path = _series_state_path(project_dir)
    payload = state.model_dump(mode="json")
    fd, tmp_path = tempfile.mkstemp(
        prefix=".series_state_",
        suffix=".json.tmp",
        dir=str(project_dir),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def record_episode_tics(
    state: SeriesTicState,
    *,
    episode_number: int,
    family_counts: dict[str, int],
    surface_phrases: list[str],
) -> SeriesTicState:
    """Return a new state with the episode's tic counts merged in.

    Idempotent for the same ``episode_number``: a second call replaces the
    prior entry rather than appending. The cumulative_family_counts is
    recomputed from per_episode after the merge.
    """
    new_entry = SeriesTicEpisodeEntry(
        episode_number=episode_number,
        family_counts=dict(family_counts),
        surface_phrases=sorted(set(surface_phrases)),
        timestamp=datetime.now(UTC).isoformat(),
    )
    per_episode = [e for e in state.per_episode if e.episode_number != episode_number]
    per_episode.append(new_entry)
    per_episode.sort(key=lambda e: e.episode_number)
    cumulative: dict[str, int] = {}
    for entry in per_episode:
        for family, count in entry.family_counts.items():
            cumulative[family] = cumulative.get(family, 0) + int(count)
    return SeriesTicState(
        project_id=state.project_id,
        cumulative_family_counts=cumulative,
        per_episode=per_episode,
        schema_version=state.schema_version,
    )


def compute_writing_blocklist(
    state: SeriesTicState,
    *,
    current_episode: int,
    recent_window: int = 3,
    carryover_threshold: int = 3,
) -> list[str]:
    """Return surface phrases the writer must avoid as section openings.

    Union of:
      - surface_phrases from episodes in the window
        ``[current_episode - recent_window, current_episode - 1]``
      - canonical example phrases for any family whose cumulative count
        meets ``carryover_threshold``

    The recent-window membership uses surface phrases as they appeared in the
    prior episodes (high-precision). The carryover threshold is a coarser
    signal: it adds the family's canonical surface phrases even if those
    exact strings didn't occur, so the writer learns to vary the move type.
    """
    if current_episode < 1:
        return []
    window_start = max(1, current_episode - recent_window)
    window_end = current_episode - 1
    blocked: set[str] = set()
    for entry in state.per_episode:
        if window_start <= entry.episode_number <= window_end:
            for phrase in entry.surface_phrases:
                blocked.add(phrase)
    # Add carryover-family canonical openings (imported lazily to avoid the
    # tic_families module pulling in embeddings during pure-state reads).
    if state.cumulative_family_counts:
        from podcast_agent.pipeline.tic_families import TIC_FAMILY_SEEDS

        for family, count in state.cumulative_family_counts.items():
            if count >= carryover_threshold:
                for seed in TIC_FAMILY_SEEDS.get(family, []):
                    blocked.add(seed)
    return sorted(blocked)
