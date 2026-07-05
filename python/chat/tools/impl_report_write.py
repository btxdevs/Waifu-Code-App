"""ReportWrite tool — port of Assets/Scripts/ChatTools/WriteReportToolExecutor.cs.

Calls ChatManager.register_report() to stage the report. The ChatManager owns
the persisting + UI attachment.
"""
from __future__ import annotations

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


class ReportWriteTool(ToolExecutor):
    name = "ReportWrite"
    permission = ToolPermission.READ_ONLY
    activity_label = "Writing report…"
    defer_until_speech_caught_up = True
    description = (
        "Display a markdown-formatted report to the user in the UI. Use for summaries, "
        "explanations, comparisons, lists, or any long-form / structured content — "
        "anything that would be awkward to recite aloud. The user reads the report in a "
        "modal panel; your spoken reply should briefly acknowledge it without repeating its contents."
    )

    def build_schema(self, session) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Short title shown at the top of the report panel "
                        "(e.g. \"Comparison: A vs B\", \"Setup steps\")."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Report body in Markdown. Supported syntax: # / ## / ### headers, "
                        "paragraphs, - and * bullet lists, 1. ordered lists, > blockquotes, "
                        "--- horizontal rules, **bold**, *italic*, `inline code`, "
                        "```fenced code blocks```, and [text](url) links (URL is dropped on render). "
                        "Use this for anything the user needs to read rather than hear — "
                        "your spoken reply should just acknowledge that you wrote it."
                    ),
                },
            },
            "required": ["title", "content"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        title = arguments.get("title")
        content = arguments.get("content")
        if not isinstance(title, str) or not title.strip():
            return ToolResult(result_text="Error: 'title' is required.", error="bad args")
        if not isinstance(content, str) or not content.strip():
            return ToolResult(result_text="Error: 'content' is required.", error="bad args")

        if ctx.chat_manager is None or not hasattr(ctx.chat_manager, "register_report"):
            return ToolResult(
                result_text="Error: ChatManager.register_report is not available.",
                error="missing chat_manager",
            )

        report_id = ctx.chat_manager.register_report(title.strip(), content)
        char_count = len(content)
        return ToolResult(
            result_text=(
                f"Report \"{title.strip()}\" written ({char_count} chars of markdown). "
                f"The user can re-open it from the chat bubble or history (id: {report_id})."
            ),
        )
