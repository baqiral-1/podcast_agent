from __future__ import annotations

from unittest.mock import MagicMock, patch

from podcast_agent.agents.synthesis_primitives import PrimitiveSubstrateExtractionAgent
from podcast_agent.langchain.runnables import (
    RetryableGenerationError,
    TransientLLMError,
)


def _mock_llm() -> MagicMock:
    return MagicMock()


def _base_payload() -> dict[str, object]:
    return {
        "project_id": "proj",
        "podcast_mode": "minified",
        "axes": [{"axis_id": "axis_01", "name": "Axis"}],
        "passages_by_axis": {
            "axis_01": [
                {
                    "book_id": "book_01",
                    "passages": [{"passage_id": "passage_01", "text": "Text"}],
                }
            ]
        },
        "cross_book_pairs": [],
        "books": [{"book_id": "book_01"}],
    }


def _schema_retryable_error(
    invalid_payload: dict[str, object],
) -> RetryableGenerationError:
    llm = _mock_llm()
    try:
        PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2).response_model.model_validate(
            invalid_payload
        )
    except Exception as validation_exc:
        try:
            raise RetryableGenerationError(
                "Schema validation failed for primitive_substrate_extraction",
                data={"raw_payload": invalid_payload},
            ) from validation_exc
        except RetryableGenerationError as retry_exc:
            return retry_exc
    raise AssertionError("expected invalid synthesis-primitives payload")


def test_synthesis_primitives_retry_feedback_for_missing_event_fields() -> None:
    llm = _mock_llm()
    llm.generate_json.side_effect = [
        _schema_retryable_error(
            {
                "project_id": "proj",
                "events": [
                    {
                        "title": "A coup lands",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1953",
                        "geography": "Tehran",
                        "actor_ids": ["actor_01"],
                    }
                ],
            }
        ),
        PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2).response_model.model_validate(
            {
                "project_id": "proj",
                "events": [
                    {
                        "title": "A coup lands",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1953",
                        "geography": "Tehran",
                        "actor_ids": ["actor_01"],
                        "event_type": "coup d'etat",
                        "what_happened": "A military coup topples the government.",
                    }
                ],
            }
        ),
    ]
    agent = PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(_base_payload())

    assert result.project_id == "proj"
    assert result.primitives[0].id == "e1"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "synthesis_feedback" not in first_kwargs["payload"]
    feedback = second_kwargs["payload"]["synthesis_feedback"]
    assert feedback["issue"] == "missing_required_substrate_field"
    assert feedback["validation_errors"] == [
        {
            "path": "events.0.event_type",
            "error_type": "missing",
            "message": "Field required",
            "primitive_index": 0,
            "primitive_id": None,
            "substrate": "events",
            "field": "event_type",
            "issue": "missing_required_substrate_field",
            "required_fix": "Add a non-empty `event_type` for this `events` primitive. Do not rely on `title` as a substitute.",
            "primitive_snapshot": {
                "substrate": "events",
                "title": "A coup lands",
                "core_passage_ids": ["passage_01"],
                "timeframe": "1953",
                "geography": "Tehran",
                "actor_ids": ["actor_01"],
            },
        },
        {
            "path": "events.0.what_happened",
            "error_type": "missing",
            "message": "Field required",
            "primitive_index": 0,
            "primitive_id": None,
            "substrate": "events",
            "field": "what_happened",
            "issue": "missing_required_substrate_field",
            "required_fix": "Add a non-empty `what_happened` for this `events` primitive. Do not rely on `title` as a substitute.",
            "primitive_snapshot": {
                "substrate": "events",
                "title": "A coup lands",
                "core_passage_ids": ["passage_01"],
                "timeframe": "1953",
                "geography": "Tehran",
                "actor_ids": ["actor_01"],
            },
        },
    ]


def test_synthesis_primitives_retry_feedback_for_invalid_substrate_enum() -> None:
    llm = _mock_llm()
    llm.generate_json.side_effect = [
        _schema_retryable_error(
            {
                "project_id": "proj",
                "acts": [
                    {
                        "title": "A tactical delay",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1978",
                        "geography": "Tehran",
                        "actor_ids": ["actor_01"],
                        "act_type": "tactical_pivot",
                        "act_summary": "An actor delays a move.",
                    }
                ],
            }
        ),
        PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2).response_model.model_validate(
            {
                "project_id": "proj",
                "acts": [
                    {
                        "title": "A tactical delay",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1978",
                        "geography": "Tehran",
                        "actor_ids": ["actor_01"],
                        "act_type": "delay",
                        "act_summary": "An actor delays a move.",
                    }
                ],
            }
        ),
    ]
    agent = PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(_base_payload())

    assert result.project_id == "proj"
    assert result.primitives[0].id == "a1"
    feedback = llm.generate_json.call_args_list[1].kwargs["payload"]["synthesis_feedback"]
    assert feedback["issue"] == "invalid_substrate_enum"
    assert feedback["validation_errors"] == [
        {
            "path": "acts.0.act_type",
            "error_type": "literal_error",
            "message": "Input should be 'decision', 'refusal', 'delay', 'deferral', 'order', 'defection' or 'other'",
            "primitive_index": 0,
            "primitive_id": None,
            "substrate": "acts",
            "field": "act_type",
            "issue": "invalid_substrate_enum",
            "required_fix": "Replace `act_type` with one of the schema-allowed literal values for this `acts` primitive.",
            "primitive_snapshot": {
                "substrate": "acts",
                "title": "A tactical delay",
                "core_passage_ids": ["passage_01"],
                "timeframe": "1978",
                "geography": "Tehran",
                "actor_ids": ["actor_01"],
                "act_type": "tactical_pivot",
                "act_summary": "An actor delays a move.",
            },
        }
    ]


def test_synthesis_primitives_retry_feedback_for_invalid_substrate_shape() -> None:
    llm = _mock_llm()
    llm.generate_json.side_effect = [
        _schema_retryable_error(
            {
                "project_id": "proj",
                "conditions": [
                    {
                        "title": "A standing strain",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1970s",
                        "geography": "Iran",
                        "condition_type": "fault_line",
                        "condition_summary": "",
                    }
                ],
            }
        ),
        PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2).response_model.model_validate(
            {
                "project_id": "proj",
                "conditions": [
                    {
                        "title": "A standing strain",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1970s",
                        "geography": "Iran",
                        "condition_type": "fault_line",
                        "condition_summary": "The field is unstable but not yet broken.",
                    }
                ],
            }
        ),
    ]
    agent = PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(_base_payload())

    assert result.project_id == "proj"
    assert result.primitives[0].id == "c1"
    feedback = llm.generate_json.call_args_list[1].kwargs["payload"]["synthesis_feedback"]
    assert feedback["issue"] == "invalid_substrate_shape"
    assert feedback["validation_errors"] == [
        {
            "path": "conditions.0.condition_summary",
            "error_type": "string_too_short",
            "message": "String should have at least 1 character",
            "primitive_index": 0,
            "primitive_id": None,
            "substrate": "conditions",
            "field": "condition_summary",
            "issue": "invalid_substrate_shape",
            "required_fix": "Correct `condition_summary` so it matches the required shape for this `conditions` primitive.",
            "primitive_snapshot": {
                "substrate": "conditions",
                "title": "A standing strain",
                "core_passage_ids": ["passage_01"],
                "timeframe": "1970s",
                "geography": "Iran",
                "condition_type": "fault_line",
                "condition_summary": "",
            },
        }
    ]


def test_synthesis_primitives_allows_deferred_mechanism_fields_in_extraction() -> None:
    artifact = PrimitiveSubstrateExtractionAgent(
        _mock_llm(), max_retry_attempts=2
    ).response_model.model_validate(
        {
            "project_id": "proj",
            "mechanisms": [
                {
                    "title": "A patronage circuit",
                    "core_passage_ids": ["passage_01"],
                    "mechanism_name": "patronage chain",
                }
            ],
        }
    )

    assert artifact.primitives_by_substrate()["mechanisms"][0].mechanism_name == "patronage chain"


def test_synthesis_primitives_transient_retry_does_not_add_feedback() -> None:
    llm = _mock_llm()
    llm.generate_json.side_effect = [
        TransientLLMError("timeout"),
        PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2).response_model.model_validate(
            {
                "project_id": "proj",
                "events": [
                    {
                        "title": "A coup lands",
                        "core_passage_ids": ["passage_01"],
                        "timeframe": "1953",
                        "geography": "Tehran",
                        "actor_ids": ["actor_01"],
                        "event_type": "coup d'etat",
                        "what_happened": "A military coup topples the government.",
                    }
                ],
            }
        ),
    ]
    agent = PrimitiveSubstrateExtractionAgent(llm, max_retry_attempts=2)

    with patch("podcast_agent.agents.base.time.sleep", return_value=None):
        result = agent.run(_base_payload())

    assert result.project_id == "proj"
    assert result.primitives[0].id == "e1"
    first_kwargs = llm.generate_json.call_args_list[0].kwargs
    second_kwargs = llm.generate_json.call_args_list[1].kwargs
    assert "synthesis_feedback" not in first_kwargs["payload"]
    assert "synthesis_feedback" not in second_kwargs["payload"]
