"""Character library + reports/todos/saves listing: register reports/todos, push lists, character CRUD, fetch."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

# The character profile picture rides the wire as a data URL of the processed JPEG
# (see ChatApi.pick_profile_image); on disk it's characters/<charId>/profile.jpg.
_PROFILE_DATAURL_PREFIX = "data:image/jpeg;base64,"

from ..character_store import CharacterStore, CharacterRecord, new_character_id, coerce_coordinates
from ..config import ChatBackendConfig
from ..llm_client import LlmClient
from ..models import ChatMessage, EmotionEntry, StructuredReply, ToolSchema
from ..orchestrator import (
    ChatOrchestrator, ChatSession, CharacterInfo, OutfitInfo, OrchestratorEvents,
    ToolExecutionResult, ToolRunner,
)
from ..save_load import (
    ChatSaveData, ReportEntry, SaveLoadManager, TodoSnapshotEntry, TodoItemSnapshot, TurnSnapshot,
)
from ..speech import SentenceSpeechPipeline
from ..text import EmotionStreamFilter
from .protocol import *  # noqa: F401,F403  (envelope-type + callable-alias constants)
from .view_models import (
    HistoryEntry, ReportRef, TodoItemRef,
    _tool_activity_label, _leading_emotion_tags,
    _EMOTION_TAG_REGEX, _LEADING_EMOTION_TAGS_REGEX,
)


class LibraryMixin:
    """Mixin for ChatManager — see chat.manager.core.ChatManager."""

    def register_report(self, title: str, markdown: str) -> str:
        """Tool-driven entry — captures a report's metadata, persists the body, queues
        the id for attachment to the current turn's assistant bubble."""
        rid = uuid.uuid4().hex
        # Walk back through history for the latest assistant message; that's the bubble
        # the report attaches to (orchestrator appends the assistant tool_calls message
        # BEFORE running tools).
        assistant_idx = -1
        if self.orchestrator is not None and self.orchestrator.session is not None:
            hist = self.orchestrator.session.history
            for i in range(len(hist) - 1, -1, -1):
                if hist[i].role == "assistant":
                    assistant_idx = i
                    break
        self._reports.append(ReportEntry(id=rid, title=title or "Report", history_index=assistant_idx))
        self._pending_report_ids_this_turn.append(rid)
        self.save.save_report(rid, markdown or "", self.current_slot)
        # Pop the report viewer right away.
        # Pop the report viewer locally via the app's task-window spawner. Unity has
        # no handler for ShowReport after the migration, so we don't send it over the WS.
        try:
            self._open_modal_fn({
                "id": "m_" + uuid.uuid4().hex,
                "type": "ShowReport",
                "payload": {"title": title or "Report", "markdown": markdown or ""},
            })
        except Exception as e:
            print(f"[ChatManager] open_modal failed (register_report): {e}", file=sys.stderr)
        return rid

    def register_todo_snapshot(self, items: list[dict]) -> str:
        """Tool-driven entry. Drops the prior pending snapshot for this turn (later
        TodoWrite calls overwrite the earlier one) and records a new one."""
        if self._pending_todo_snapshot_id_this_turn:
            self._todo_snapshots = [t for t in self._todo_snapshots
                                    if t.id != self._pending_todo_snapshot_id_this_turn]
            self._pending_todo_snapshot_id_this_turn = None

        tid = uuid.uuid4().hex
        assistant_idx = -1
        if self.orchestrator is not None and self.orchestrator.session is not None:
            hist = self.orchestrator.session.history
            for i in range(len(hist) - 1, -1, -1):
                if hist[i].role == "assistant":
                    assistant_idx = i
                    break
        snap = TodoSnapshotEntry(
            id=tid,
            history_index=assistant_idx,
            items=[TodoItemSnapshot(
                content=str(it.get("content") or ""),
                active_form=str(it.get("activeForm") or ""),
                status=str(it.get("status") or "pending"),
            ) for it in items if isinstance(it, dict)],
        )
        self._todo_snapshots.append(snap)
        self._pending_todo_snapshot_id_this_turn = tid
        return tid

    def push_saves_list(self) -> None:
        slots = [m.to_wire() for m in self.save.list_save_slots()]
        self._push_envelope(T_CHAT_SAVES_LIST, {"slots": slots})

    async def push_characters_list(self) -> None:
        chars = await self._fetch_characters_list()
        self._push_envelope(T_CHAT_CHARACTERS_LIST, {"characters": chars})

    async def _on_save_character(self, payload: dict) -> None:
        """Persist a character authored/edited on the creation page, then refresh the picker list.
        An `id` in the payload means EDIT (keep the same id + created_at so existing chats stay
        attached); no id means CREATE (mint a fresh id)."""
        char_id = str(payload.get("id") or "").strip()
        created_at = ""
        existing: CharacterRecord | None = None
        if char_id:
            existing = self._character_store.load(char_id)
            if existing is not None:
                created_at = existing.created_at_utc  # preserve original creation time on edit
        else:
            char_id = new_character_id()

        if not str(payload.get("name") or "").strip():
            print("[ChatManager] Chat.SaveCharacter with empty name — ignored", file=sys.stderr)
            return

        voices = await self._resolve_character_voices(char_id, payload.get("voices"), existing)
        picked_model = str(payload.get("modelPath") or "").replace("\\", "/")
        model_path = await self._resolve_character_model(char_id, picked_model)
        # The display name mirrors the pocket voice's clipName: the originally picked file's
        # name. On an unedited re-save the payload carries the copy's path (model_<hash>…),
        # so keep the name recorded when the file was first picked.
        model_name = Path(picked_model).name if picked_model else ""
        if existing is not None and model_path == (existing.model_path or "").replace("\\", "/") and existing.model_name:
            model_name = existing.model_name

        profile_image = payload.get("profileImage")
        if isinstance(profile_image, str):
            self._store_profile_image(char_id, profile_image)

        rec = CharacterRecord(
            id=char_id,
            name=str(payload.get("name") or "").strip(),
            display_name=str(payload.get("displayName") or ""),
            character_definition=str(payload.get("characterDefinition") or ""),
            initial_scenario=str(payload.get("initialScenario") or ""),
            initial_assistant_message=str(payload.get("initialAssistantMessage") or "Hello!"),
            system_prompt_template=str(payload.get("systemPromptTemplate") or ""),
            initial_emotion_label=str(payload.get("initialEmotionLabel") or "Neutral"),
            model_path=model_path,
            model_name=model_name,
            available_emotions=[str(e) for e in (payload.get("availableEmotions") or []) if isinstance(e, str)],
            coordinates=coerce_coordinates(payload.get("coordinates")),
            default_outfit_index=int(payload["defaultOutfitIndex"])
                if isinstance(payload.get("defaultOutfitIndex"), (int, float)) else -1,
            voices=voices,
            created_at_utc=created_at,
        )
        self._character_store.save(rec)
        self._character_cache.pop(rec.id, None)  # drop any stale CharacterInfo
        self._characters_cached = None           # force the picker list to rebuild
        await self.push_characters_list()

        # If the edited character is the one in the LIVE session, reload that chat now so the edits
        # (persona / system prompt / model / voice) take effect immediately — same as reopening it
        # from home. Done here, after the store write + cache drop above, so the reload's
        # _fetch_character rebuilds from the fresh record (no save/load race). We persist the
        # in-memory history first so the round-trip through disk keeps the latest messages.
        session = self.orchestrator.session if self.orchestrator is not None else None
        if session is not None and self.current_slot and session.character.id == rec.id:
            self._persist()
            await self._on_load_save(self.current_slot)

    async def _resolve_character_voices(
        self, char_id: str, payload_voices, existing: CharacterRecord | None
    ) -> dict:
        """Turn the editor's `voices` payload into the persisted per-provider map.

        ElevenLabs entries are stored verbatim (just the opaque voice id). Pocket
        entries carry a reference clip path from the editor; we hash the clip bytes
        and only (re)encode the embedding when the hash differs from what's stored —
        deleting the previous embedding file first. An unchanged clip (or an edit
        that doesn't touch the pocket voice) keeps the existing embedding."""
        payload_voices = payload_voices if isinstance(payload_voices, dict) else {}
        prev = dict(existing.voices) if (existing and isinstance(existing.voices, dict)) else {}
        out: dict = {}

        # --- ElevenLabs: opaque voice id, stored as-is. ---
        el = payload_voices.get("elevenlabs")
        if isinstance(el, dict):
            voice_id = str(el.get("voiceId") or "").strip()
            if voice_id:
                out["elevenlabs"] = {"voiceId": voice_id}

        # --- Pocket: reference clip → embedding (.npy) in the character's folder. ---
        pk = payload_voices.get("pocket")
        if isinstance(pk, dict):
            prev_pk = prev.get("pocket") if isinstance(prev.get("pocket"), dict) else {}
            clip_path = str(pk.get("clipPath") or "").strip()
            prev_hash = str(prev_pk.get("clipHash") or "")
            prev_embed = str(prev_pk.get("embeddingFile") or "")
            if clip_path:
                try:
                    clip_bytes = Path(clip_path).read_bytes()
                    clip_hash = hashlib.sha256(clip_bytes).hexdigest()[:16]
                except OSError as e:
                    self._push_error("voice_clip", f"Couldn't read the voice clip: {e}")
                    clip_hash = ""
                if clip_hash and clip_hash == prev_hash and prev_embed and Path(prev_embed).exists():
                    # Same clip as before — reuse the embedding we already encoded.
                    out["pocket"] = dict(prev_pk)
                elif clip_hash:
                    entry = await self._encode_pocket_embedding(char_id, clip_path, clip_hash)
                    if entry is not None:
                        out["pocket"] = entry
                    elif prev_pk and prev_embed and Path(prev_embed).exists():
                        # Encoding failed — keep the previous working voice rather than losing it.
                        out["pocket"] = dict(prev_pk)
            elif prev_pk:
                # Editor didn't re-pick a clip; keep the existing pocket voice untouched.
                out["pocket"] = dict(prev_pk)

        # Any pocket embedding files left orphaned by this save (hash changed / voice
        # removed) are cleaned up against the final keep-set.
        self._prune_pocket_embeddings(char_id, keep=out.get("pocket"))
        return out

    async def _encode_pocket_embedding(self, char_id: str, clip_path: str, clip_hash: str) -> dict | None:
        """Encode `clip_path` into a mimi embedding saved as <charFolder>/voice_<hash>.npy
        and return the pocket voice entry. Returns None (and surfaces a Chat.Error) if no
        encoder is wired or encoding fails. Runs the heavy ONNX encode off the event loop."""
        if self._encode_pocket_voice_fn is None:
            self._push_error("voice_encoder", "Pocket-TTS voice encoding is unavailable.")
            return None
        folder = self._character_store.folder_for(char_id)
        folder.mkdir(parents=True, exist_ok=True)
        out_path = folder / f"voice_{clip_hash}.npy"
        try:
            await asyncio.to_thread(self._encode_pocket_voice_fn, clip_path, str(out_path))
        except Exception as e:  # noqa: BLE001 — surface any encode/model-load failure to the UI
            print(f"[ChatManager] pocket voice encode failed: {e}", file=sys.stderr)
            self._push_error("voice_encode", f"Couldn't generate the pocket voice: {e}")
            return None
        return {
            "clipHash": clip_hash,
            "embeddingFile": str(out_path).replace("\\", "/"),
            "clipName": Path(clip_path).name,
        }

    def _store_profile_image(self, char_id: str, data_url: str) -> None:
        """Write (or, for an empty string, remove) the character's profile picture at
        characters/<charId>/profile.jpg. The editor sends the already-processed JPEG as a
        data URL; anything not in that form is ignored."""
        dest = self._character_store.folder_for(char_id) / "profile.jpg"
        if not data_url:
            try:
                dest.unlink(missing_ok=True)
            except OSError:
                pass
            return
        if not data_url.startswith(_PROFILE_DATAURL_PREFIX):
            return
        try:
            raw = base64.b64decode(data_url[len(_PROFILE_DATAURL_PREFIX):], validate=False)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
        except (OSError, ValueError) as e:
            print(f"[ChatManager] profile image write failed: {e}", file=sys.stderr)
            self._push_error("profile_image", f"Couldn't save the profile picture: {e}")

    def _profile_image_data_url(self, char_id: str) -> str:
        """The character's profile.jpg as a data URL for the renderer ('' when none)."""
        p = self._character_store.folder_for(char_id) / "profile.jpg"
        try:
            if not p.exists():
                return ""
            return _PROFILE_DATAURL_PREFIX + base64.b64encode(p.read_bytes()).decode("ascii")
        except OSError:
            return ""

    def _prune_pocket_embeddings(self, char_id: str, keep: dict | None) -> None:
        """Delete this character's embedding files except the one in `keep` (if any) —
        orphans from a changed/removed clip."""
        keep_path = ""
        if isinstance(keep, dict):
            keep_path = str(keep.get("embeddingFile") or "").replace("\\", "/")
        for p in self._character_store.folder_for(char_id).glob("voice_*.npy"):
            if str(p).replace("\\", "/") == keep_path:
                continue
            try:
                p.unlink()
            except OSError:
                pass

    async def _resolve_character_model(self, char_id: str, model_path: str) -> str:
        """Copy the picked model file into the character's folder (model_<hash>.<ext>) and
        return the copy's path, so the character keeps working when the originally picked
        file is moved or deleted — same idea as the pocket voice embeddings. A path already
        inside the folder (an unedited re-save) is kept as-is; re-picking identical file
        content reuses the existing copy (matched by hash). Old copies are pruned after each
        save. On a copy failure the original path is stored so the character still works
        while that file exists."""
        if not model_path:
            return ""
        model_path = model_path.replace("\\", "/")
        src = Path(model_path)
        folder = self._character_store.folder_for(char_id)
        try:
            if src.resolve().parent == folder.resolve():
                self._prune_model_copies(char_id, keep=model_path)
                return model_path
        except OSError:
            pass
        result = model_path
        try:
            result = await asyncio.to_thread(self._copy_model_file, char_id, src)
        except OSError as e:
            print(f"[ChatManager] model copy failed: {e}", file=sys.stderr)
            self._push_error("model_copy", f"Couldn't copy the model file: {e}")
        self._prune_model_copies(char_id, keep=result)
        return result

    def _copy_model_file(self, char_id: str, src: Path) -> str:
        """Hash `src` and copy it to <charFolder>/model_<hash>.<ext> (atomically, via a
        temp file) unless that copy already exists. Sync + heavy (model files are tens of
        MB) — run off the event loop. Raises OSError on read/copy failure."""
        digest = hashlib.sha256()
        with src.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)
        file_hash = digest.hexdigest()[:16]
        folder = self._character_store.folder_for(char_id)
        out = folder / f"model_{file_hash}{src.suffix.lower()}"
        if not out.exists():
            folder.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(out.suffix + ".tmp")
            shutil.copyfile(src, tmp)
            os.replace(tmp, out)
        return str(out).replace("\\", "/")

    def _prune_model_copies(self, char_id: str, keep: str | None) -> None:
        """Delete this character's model copies except the one in `keep` (if any) —
        orphans left by a re-picked model."""
        keep_path = str(keep or "").replace("\\", "/")
        for p in self._character_store.folder_for(char_id).glob("model_*"):
            if str(p).replace("\\", "/") == keep_path:
                continue
            try:
                p.unlink()
            except OSError:
                pass

    async def _on_import_character(self, path: str) -> None:
        """Import a .wcc character bundle picked via ChatApi.import_character. The store
        mints a fresh id and rewrites the bundle's machine-local paths, so the character
        is ready to chat immediately; we just refresh the picker list."""
        if not path:
            return
        rec, err = await asyncio.to_thread(self._character_store.import_bundle, path)
        if err or rec is None:
            self._push_error("character_import", err or "Import failed.")
            return
        self._characters_cached = None
        await self.push_characters_list()

    async def _on_delete_character(self, char_id: str) -> None:
        """Remove a character from the store and refresh the picker. Existing chats that
        referenced it are left in place; they'll just report 'character no longer exists' if
        someone tries to resume them. The store delete removes the character's whole folder
        (definition + model copy + voice embedding)."""
        if not char_id:
            return
        self._character_store.delete(char_id)
        self._character_cache.pop(char_id, None)
        self._characters_cached = None
        await self.push_characters_list()

    async def capture_self_view(self) -> dict:
        """RPC Unity to capture a screenshot of the avatar's current on-screen view. Returns the reply
        payload {base64, format, width, height, error} (empty dict if Unity sent no id). Used by the
        LookAtYourself tool so the AI can see how it looks right now."""
        return await self._request_with_correlation(
            T_CHARACTER_CAPTURE_VIEW, {}, self._capture_futures)

    async def _on_inspect_model_emotions(self, model_path: str) -> None:
        """Relay a .vrm emotion-inspection to Unity and push the result back to the renderer.
        Used by the creation page when the user picks a model, so emotions are set at creation."""
        emotions: list[str] = []
        error = ""
        if model_path:
            reply = await self._request_with_correlation(
                T_CHARACTER_INSPECT_MODEL_EMOTIONS, {"modelPath": model_path}, self._character_futures)
            emotions = [str(e) for e in (reply.get("emotions") or []) if isinstance(e, str)]
            error = str(reply.get("error") or "")
        else:
            error = "No modelPath provided."
        self._push_envelope(T_CHAT_MODEL_EMOTIONS,
                            {"modelPath": model_path, "emotions": emotions, "error": error})

    async def _on_inspect_model_coordinates(self, model_path: str) -> None:
        """Load a KK model's outfit/coordinate list and push it to the editor so the user can
        name/describe each outfit. Read locally (no Unity round-trip) from the .kkm zip
        archive, where KK_Coordinates.json is packed inside. VRM models, and KK models with
        no coordinates data, just get an empty list (no error)."""
        coordinates: list[dict] = []
        error = ""
        if not model_path:
            self._push_envelope(T_CHAT_MODEL_COORDINATES,
                                {"modelPath": model_path, "coordinates": [], "error": "No modelPath provided."})
            return
        ext = Path(model_path).suffix.lower()
        try:
            data = None
            if ext == ".kkm":
                with zipfile.ZipFile(model_path) as zf:
                    # Match by basename in case the export ever nests it under a folder.
                    entry = next((n for n in zf.namelist()
                                  if n.rsplit("/", 1)[-1].lower() == "kk_coordinates.json"), None)
                    if entry is not None:
                        data = json.loads(zf.read(entry).decode("utf-8"))
            raw = data.get("coordinates") if isinstance(data, dict) else data
            coordinates = coerce_coordinates(raw)
        except Exception as e:  # noqa: BLE001 — surface a read/parse failure to the editor
            error = f"Couldn't read KK_Coordinates.json: {e}"
        self._push_envelope(T_CHAT_MODEL_COORDINATES,
                            {"modelPath": model_path, "coordinates": coordinates, "error": error})

    async def _fetch_character(self, char_id: str) -> CharacterInfo | None:
        # Characters are app-owned and keyed by stable id: build CharacterInfo from the stored
        # record (model_path + emotion vocabulary captured at creation + KK outfits from coordinates).
        if not char_id:
            return None
        if char_id in self._character_cache:
            return self._character_cache[char_id]
        rec = self._character_store.load(char_id)
        if rec is None:
            print(f"[ChatManager] Character id '{char_id}' not found in the app store.", file=sys.stderr)
            return None
        outfits = self._outfits_from_coordinates(rec.coordinates)
        info = CharacterInfo(
            id=rec.id,
            character_name=rec.name,
            display_name=rec.display_name or rec.name,
            character_definition=rec.character_definition,
            initial_scenario=rec.initial_scenario,
            initial_assistant_message=rec.initial_assistant_message or "Hello!",
            system_prompt_template=rec.system_prompt_template,
            initial_emotion_label=rec.initial_emotion_label or "Neutral",
            model_path=rec.model_path,
            available_emotions=list(rec.available_emotions),
            outfits=outfits,
            default_outfit_name=outfits[0].outfit_name if outfits else "",
            default_outfit_index=rec.default_outfit_index,
            voices=dict(rec.voices),
        )
        self._character_cache[char_id] = info
        return info

    @staticmethod
    def _outfits_from_coordinates(coordinates: list[dict]) -> list[OutfitInfo]:
        """Map a KK character's stored coordinates to the orchestrator's OutfitInfo list (drives the
        system prompt's clothing section + the ChangeOutfit tool). The coordinate's `index` is carried
        through as the KK outfit id so Unity can switch to the matching coordinate; the editable name
        (falling back to "Outfit NN") and description become the LLM-facing label + blurb."""
        outfits: list[OutfitInfo] = []
        for c in (coordinates or []):
            if not isinstance(c, dict):
                continue
            idx = c.get("index")
            idx = int(idx) if isinstance(idx, (int, float)) else len(outfits)
            name = str(c.get("name") or "").strip() or f"Outfit {idx:02d}"
            outfits.append(OutfitInfo(
                outfit_name=name,
                description=str(c.get("description") or "").strip(),
                index=idx,
            ))
        return outfits

    async def _fetch_characters_list(self) -> list[dict]:
        # App-owned characters only — full wire records so the renderer can list them
        # (id + display name) AND prefill the edit form without an extra round-trip.
        if self._characters_cached is not None:
            return self._characters_cached
        records = []
        for rec in self._character_store.list_records():
            wire = rec.to_wire()
            wire["profileImage"] = self._profile_image_data_url(rec.id)
            records.append(wire)
        self._characters_cached = records
        return records
