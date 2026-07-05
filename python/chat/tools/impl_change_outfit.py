"""ChangeOutfit tool — port of Assets/Scripts/ChatTools/ChangeOutfitToolExecutor.cs.

Returns a session_mutations.currentOutfit; the orchestrator merges it and
ChatManager._on_orch_executed_tool pushes Avatar.ApplyOutfit to Unity. No
direct Unity round-trip needed.

Schema is dynamic — the outfit enum is built from the active character's
outfits at schema-build time.
"""
from __future__ import annotations

from .base import ToolContext, ToolExecutor, ToolPermission, ToolResult


class ChangeOutfitTool(ToolExecutor):
    name = "ChangeOutfit"
    permission = ToolPermission.READ_ONLY
    activity_label = "Changing outfit…"
    defer_until_speech_caught_up = True
    description = "Changes the character's current outfit to one of their available outfits."

    def build_schema(self, session) -> dict | None:
        outfit_names: list[str] = []
        if session is not None and session.character is not None:
            outfit_names = [o.outfit_name for o in (session.character.outfits or []) if o.outfit_name]
        # Nothing to switch between unless there are at least two outfits — returning None drops the
        # tool from this turn's advertised set entirely (ToolManager.build_schemas skips None).
        if len(outfit_names) < 2:
            return None
        return {
            "type": "object",
            "properties": {
                "outfitName": {
                    "type": "string",
                    "description": "The name of the outfit to change into.",
                    "enum": outfit_names,
                },
            },
            "required": ["outfitName"],
        }

    async def execute(self, arguments: dict, ctx: ToolContext) -> ToolResult:
        if ctx.session is None or ctx.session.character is None:
            return ToolResult(result_text="Error: no active character.", error="no character")

        requested = arguments.get("outfitName")
        if not isinstance(requested, str) or not requested.strip():
            return ToolResult(result_text="Error: 'outfitName' is required.", error="bad args")

        target = ctx.session.character.get_outfit(requested)
        if target is None:
            available = ", ".join(o.outfit_name for o in (ctx.session.character.outfits or []) if o.outfit_name)
            return ToolResult(
                result_text=f"Error: outfit '{requested}' is not available. Choose from: {available or '(none)'}.",
                error="unknown outfit",
            )

        previous = ctx.session.current_outfit.outfit_name if ctx.session.current_outfit else "(none)"
        # The orchestrator merges session_mutations.currentOutfit and pushes
        # Avatar.ApplyOutfit on the next on_executed_tool callback. We just emit
        # the new name here.
        return ToolResult(
            result_text=f"Successfully changed outfit from '{previous}' to '{target.outfit_name}'.",
            session_mutations={"currentOutfit": target.outfit_name},
        )
