"""Prompt templates for LangChain-backed inference."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_prompt_template() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "{system_text}"),
            ("human", "{user_text}"),
        ]
    )

