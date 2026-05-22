from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_agent.agents.primitive_function_tagging import (
    PrimitiveFunctionTaggingAgent,
)
from podcast_agent.langchain.runnables import (
    RetryableGenerationError,
    TransientLLMError,
)


def _mock_llm() -> MagicMock:
    return MagicMock()


def _base_primitive(primitive_id: str = "p_evt_1") -> dict[str, object]:
    return {
        "id": primitive_id,
        "substrate": "events",
        "title": "Embassy seizure",
        "core_passage_ids": ["passage_01"],
        "timeframe": "1979",
        "geography": "Tehran",
        "actor_ids": ["actor_01"],
        "event_type": "embassy occupation",
        "what_happened": "Students seize the embassy compound.",
    }


def _schema_retryable_error(invalid_payload: dict[str, object]) -> RetryableGenerationError:
    llm = _mock_llm()
    try:
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(invalid_payload)
    except Exception as validation_exc:
        try:
            raise RetryableGenerationError(
                "Schema validation failed for primitive_function_tagging_events",
                data={"raw_payload": invalid_payload},
            ) from validation_exc
        except RetryableGenerationError as retry_exc:
            return retry_exc
    raise AssertionError("expected invalid primitive-function-tagging payload")


def _overlay_payload(
    primitive_id: str,
    *,
    event_result: str = "The crisis hardens the revolutionary settlement.",
    functions: list[str] | None = None,
    pivot: dict[str, object] | None = None,
    texture: dict[str, object] | None = None,
    recurrence: dict[str, object] | None = None,
) -> dict[str, object]:
    overlay: dict[str, object] = {
        "event_result": event_result,
    }
    if functions is not None:
        overlay["functions"] = functions
    if pivot is not None:
        overlay["pivot"] = pivot
    if texture is not None:
        overlay["texture"] = texture
    if recurrence is not None:
        overlay["recurrence"] = recurrence
    return {
        "project_id": "proj",
        "overlays_by_id": {
            primitive_id: overlay,
        },
    }


def test_primitive_function_tagging_retry_feedback_for_payload_without_tag() -> None:
    llm = _mock_llm()
    base_primitive = {
        **_base_primitive(),
        "event_result": "The crisis hardens the revolutionary settlement.",
    }
    llm.generate_json.side_effect = [
        _schema_retryable_error(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
                texture={"what_it_anchors": "Students climbing the embassy walls."},
            )
        ),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot", "texture"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
                texture={"what_it_anchors": "Students climbing the embassy walls."},
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "function_feedback" not in first_kwargs["payload"]
    feedback = second_kwargs["payload"]["function_feedback"]
    assert feedback["issue"] == "payload_without_tag"
    assert feedback["substrate"] == "events"
    assert feedback["validation_errors"] == [
        {
            "path": "overlays_by_id.p_evt_1",
            "error_type": "value_error",
            "message": "Value error, texture justification is not allowed unless the function tag is present",
            "primitive_index": 0,
            "primitive_id": "p_evt_1",
            "substrate": "events",
            "function": "texture",
            "issue": "payload_without_tag",
            "required_fix": "Either add `texture` to `functions` or remove the `texture` payload.",
        }
    ]


def test_primitive_function_tagging_retry_feedback_for_tag_without_payload() -> None:
    llm = _mock_llm()
    base_primitive = {
        **_base_primitive(),
        "event_result": "The crisis hardens the revolutionary settlement.",
    }
    llm.generate_json.side_effect = [
        _schema_retryable_error(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot", "recurrence"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
            )
        ),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot", "recurrence"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
                recurrence={
                    "connects_to": ["p_later"],
                    "meaning_accrued": "The episode becomes a reusable anti-coup script.",
                },
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "function_feedback" not in first_kwargs["payload"]
    feedback = second_kwargs["payload"]["function_feedback"]
    assert feedback["issue"] == "tag_without_payload"
    assert feedback["validation_errors"] == [
        {
            "path": "overlays_by_id.p_evt_1",
            "error_type": "value_error",
            "message": "Value error, recurrence requires its paired justification payload",
            "primitive_index": 0,
            "primitive_id": "p_evt_1",
            "substrate": "events",
            "function": "recurrence",
            "issue": "tag_without_payload",
            "required_fix": "Either add a valid `recurrence` payload or remove `recurrence` from `functions`.",
        }
    ]


def test_primitive_function_tagging_transient_retry_does_not_add_feedback() -> None:
    llm = _mock_llm()
    base_primitive = {
        **_base_primitive(),
        "event_result": "The crisis hardens the revolutionary settlement.",
    }
    llm.generate_json.side_effect = [
        TransientLLMError("timeout"),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "function_feedback" not in first_kwargs["payload"]
    assert "function_feedback" not in second_kwargs["payload"]


def test_primitive_function_tagging_retry_feedback_for_missing_deferred_field() -> None:
    llm = _mock_llm()
    base_primitive = _base_primitive()
    llm.generate_json.side_effect = [
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
                event_result="",
            )
        ),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "function_feedback" not in first_kwargs["payload"]
    feedback = second_kwargs["payload"]["function_feedback"]
    assert feedback["issue"] == "missing_deferred_substrate_field"
    assert feedback["substrate"] == "events"
    assert feedback["validation_errors"] == [
        {
            "path": "overlays_by_id.p_evt_1.event_result",
            "error_type": "missing_deferred_substrate_field",
            "message": "Deferred substrate-detail field `event_result` must be filled before returning.",
            "primitive_index": 0,
            "primitive_id": "p_evt_1",
            "substrate": "events",
            "field": "event_result",
            "issue": "missing_deferred_substrate_field",
            "required_fix": "Fill non-empty `event_result` from the supplied evidence for this `events` primitive before returning.",
        }
    ]


def test_primitive_function_tagging_retry_feedback_for_mismatched_overlay_ids() -> None:
    llm = _mock_llm()
    base_primitive = {
        **_base_primitive(),
        "event_result": "The crisis hardens the revolutionary settlement.",
    }
    llm.generate_json.side_effect = [
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            {
                "project_id": "proj",
                "overlays_by_id": {
                    "p_evt_wrong": {
                        "event_result": "The crisis hardens the revolutionary settlement.",
                        "functions": ["pivot"],
                        "pivot": {
                            "what_changed": "The occupation collapses the provisional center.",
                            "irreversibility": "high",
                        },
                    }
                },
            }
        ),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    feedback = llm.generate_json.call_args_list[1].kwargs["payload"]["function_feedback"]
    assert feedback["issue"] == "mismatched_enrichment_overlay_ids"
    assert feedback["missing_primitive_ids"] == ["p_evt_1"]
    assert feedback["unexpected_primitive_ids"] == ["p_evt_wrong"]


def test_primitive_function_tagging_retry_feedback_for_invalid_overlay_field() -> None:
    llm = _mock_llm()
    base_primitive = {
        **_base_primitive(),
        "event_result": "The crisis hardens the revolutionary settlement.",
    }
    llm.generate_json.side_effect = [
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            {
                "project_id": "proj",
                "overlays_by_id": {
                    "p_evt_1": {
                        "event_result": "The crisis hardens the revolutionary settlement.",
                        "artifact_detail": "A detail from the wrong substrate.",
                        "functions": ["pivot"],
                        "pivot": {
                            "what_changed": "The occupation collapses the provisional center.",
                            "irreversibility": "high",
                        },
                    }
                },
            }
        ),
        PrimitiveFunctionTaggingAgent(
            llm, substrate="events", max_retry_attempts=2
        ).response_model.model_validate(
            _overlay_payload(
                "p_evt_1",
                functions=["pivot"],
                pivot={
                    "what_changed": "The occupation collapses the provisional center.",
                    "irreversibility": "high",
                },
            )
        ),
    ]
    agent = PrimitiveFunctionTaggingAgent(llm, substrate="events", max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(
            {
                "project_id": "proj",
                "podcast_mode": "full",
                "base_primitives": [base_primitive],
                "passage_list": [],
            }
        )

    assert result.project_id == "proj"
    feedback = llm.generate_json.call_args_list[1].kwargs["payload"]["function_feedback"]
    assert feedback["issue"] == "invalid_overlay_fields_for_substrate"
    assert feedback["validation_errors"] == [
        {
            "path": "overlays_by_id.p_evt_1.artifact_detail",
            "error_type": "invalid_overlay_field",
            "message": "Field `artifact_detail` is not allowed in the `events` enrichment overlay.",
            "primitive_index": None,
            "primitive_id": "p_evt_1",
            "substrate": "events",
            "field": "artifact_detail",
            "issue": "invalid_overlay_fields_for_substrate",
            "required_fix": "Remove `artifact_detail` from the returned overlay for this `events` batch.",
        }
    ]
