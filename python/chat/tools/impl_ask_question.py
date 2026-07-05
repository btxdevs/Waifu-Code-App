"""AskUserQuestion tool — port of Assets/Scripts/ChatTools/AskUserQuestionToolExecutor.cs.

Spawns a modal via the injected `ask_modal` callable, awaits the user's
selection per question, and feeds the answers back into the LLM as a structured
string.
"""
from __future__ import annotations

import uuid

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


_CHIP_WIDTH = 12


class AskUserQuestionTool(ToolExecutor):
    name = "AskUserQuestion"
    permission = ToolPermission.READ_ONLY
    activity_label = "Asking a question…"
    defer_until_speech_caught_up = True
    description = (
        "Use this tool when you need to ask the user questions during execution. This allows you to:\n"
        "1. Gather user preferences or requirements\n"
        "2. Clarify ambiguous instructions\n"
        "3. Get decisions on implementation choices as you work\n"
        "4. Offer choices to the user about what direction to take.\n\n"
        "Usage notes:\n"
        "- Users will always be able to select \"Other\" to provide custom text input\n"
        "- Use multiSelect: true to allow multiple answers to be selected for a question\n"
        "- If you recommend a specific option, make that the first option in the list and add \"(Recommended)\" at the end of the label\n\n"
        "Plan mode note: In plan mode, use this tool to clarify requirements or choose between approaches BEFORE finalizing your plan."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "questions": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "The complete question to ask the user. Should be clear, specific, "
                                    "and end with a question mark. Example: \"Which library should we use "
                                    "for date formatting?\" If multiSelect is true, phrase it accordingly, "
                                    "e.g. \"Which features do you want to enable?\""
                                ),
                            },
                            "header": {
                                "type": "string",
                                "description": f"Very short label displayed as a chip/tag (max {_CHIP_WIDTH} chars). Examples: \"Auth method\", \"Library\", \"Approach\".",
                            },
                            "options": {
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "label": {
                                            "type": "string",
                                            "description": (
                                                "The display text for this option that the user will see and "
                                                "select. Should be concise (1-5 words) and clearly describe the choice."
                                            ),
                                        },
                                        "description": {
                                            "type": "string",
                                            "description": (
                                                "Explanation of what this option means or what will happen "
                                                "if chosen. Useful for providing context about trade-offs or implications."
                                            ),
                                        },
                                        "preview": {
                                            "type": "string",
                                            "description": (
                                                "Optional preview content rendered when this option is "
                                                "focused. Use for mockups, code snippets, or visual comparisons "
                                                "that help users compare options. See the tool description for "
                                                "the expected content format."
                                            ),
                                        },
                                    },
                                    "required": ["label", "description"],
                                },
                                "description": (
                                    "The available choices for this question. Must have 2-4 options. Each "
                                    "option should be a distinct, mutually exclusive choice (unless "
                                    "multiSelect is enabled). There should be no 'Other' option, that will "
                                    "be provided automatically."
                                ),
                            },
                            "multiSelect": {
                                "type": "boolean",
                                "default": False,
                                "description": (
                                    "Set to true to allow the user to select multiple options instead of "
                                    "just one. Use when choices are not mutually exclusive."
                                ),
                            },
                        },
                        "required": ["question", "header", "options", "multiSelect"],
                    },
                    "description": "Questions to ask the user (1-4 questions)",
                },
            },
            "required": ["questions"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        questions = arguments.get("questions")
        if not isinstance(questions, list) or not questions:
            return ToolResult(result_text="Error: 'questions' must be a non-empty array.", error="bad args")
        if ctx.ask_modal is None:
            return ToolResult(
                result_text="Error: ask_modal is not wired — cannot prompt the user.",
                error="no ask_modal",
            )

        answers: list[tuple[str, str]] = []
        for q in questions:
            if not isinstance(q, dict):
                continue
            question_text = str(q.get("question") or "").strip()
            header = str(q.get("header") or "").strip()
            multi_select = bool(q.get("multiSelect", False))
            options_raw = q.get("options") or []
            options: list[dict] = []
            for opt in options_raw:
                if not isinstance(opt, dict):
                    continue
                label = str(opt.get("label") or "").strip()
                desc = str(opt.get("description") or "")
                if not label:
                    continue
                opt_out = {"label": label, "description": desc}
                preview = opt.get("preview")
                if isinstance(preview, str) and preview:
                    opt_out["preview"] = preview
                options.append(opt_out)
            if not question_text or len(options) < 2:
                return ToolResult(
                    result_text="Error: each question needs 'question' text and at least 2 options.",
                    error="bad question",
                )

            env = {
                "id": "m_" + uuid.uuid4().hex,
                "type": "AskQuestion",
                "payload": {
                    "question": question_text,
                    "header": header,
                    "multiSelect": multi_select,
                    "options": options,
                },
            }
            try:
                reply = await ctx.ask_modal(env)
            except Exception as e:
                return ToolResult(
                    result_text=f"Error: question modal failed: {e}",
                    error=str(e),
                )
            if not isinstance(reply, dict) or reply.get("cancelled"):
                return ToolResult(result_text="User declined to answer questions.", error="user-declined")
            text = str(reply.get("text") or "").strip()
            if not text:
                return ToolResult(result_text="User declined to answer questions.", error="user-declined")
            answers.append((question_text, text))

        if not answers:
            return ToolResult(result_text="User declined to answer questions.", error="user-declined")
        parts = [f'"{q.replace(chr(34), chr(92) + chr(34))}"="{a.replace(chr(34), chr(92) + chr(34))}"' for q, a in answers]
        return ToolResult(
            result_text=(
                "User has answered your questions: " + ", ".join(parts) +
                ". You can now continue with the user's answers in mind."
            ),
        )
