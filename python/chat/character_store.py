"""App-owned character library.

Character definitions now live on the app side (not Unity's CharacterRegistry):
one folder per character in `characters/<charId>/`, holding everything the character
needs — `character.json` (the definition), `model_<hash>.<ext>` (the app-owned copy of
its .vrm/.kkm) and `voice_<hash>.npy` (its pocket-TTS embedding). The creation page
writes them, the chat manager reads them to build the system prompt and to tell Unity
which model to load (Session.Begin.modelPath) and which emotions are allowed.

Outfits are intentionally not modeled — a runtime-loaded VRM is a single model. The
emotion vocabulary is decided at creation time by inspecting the chosen .vrm
(Character.InspectModelEmotions → Unity), and stored here as `available_emotions`.

Atomic writes (temp-file + rename) and filename sanitization mirror save_load.py.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path


from .app_paths import APP_ROOT

# APP_ROOT is CompanionApp/ from source, or the .exe folder when frozen.
_DEFAULT_CHARACTERS_DIR = APP_ROOT / "characters"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def new_character_id() -> str:
    """Stable, opaque id for a character. Chats reference this (not the name), so a
    character can be renamed without orphaning its chats."""
    return uuid.uuid4().hex


def coerce_coordinates(raw) -> list[dict]:
    """Normalize a coordinates list (from a wire payload or a KK_Coordinates.json) into the
    stored shape: [{index:int, name:str, description:str, screenshots:{...}}]. KK models ship
    this list beside the model file; the editor lets the user name/describe each outfit. Junk
    entries are dropped; missing fields default."""
    out: list[dict] = []
    if not isinstance(raw, list):
        return out
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        screenshots = item.get("screenshots")
        out.append({
            "index": int(idx) if isinstance(idx, (int, float)) else i,
            "name": str(item.get("name") or ""),
            "description": str(item.get("description") or ""),
            "screenshots": dict(screenshots) if isinstance(screenshots, dict) else {},
        })
    return out


@dataclass
class CharacterRecord:
    """A persisted character definition. `id` is the stable key (and filename stem); `name`
    is a human-facing label that can be edited freely without breaking existing chats."""
    id: str = ""
    name: str = ""
    display_name: str = ""
    character_definition: str = ""
    initial_scenario: str = ""
    initial_assistant_message: str = "Hello!"
    system_prompt_template: str = ""
    initial_emotion_label: str = "Neutral"
    # Path to the character's model — the app-owned copy in the character's folder
    # (model_<hash>.<ext>); sent to Unity as Session.Begin.modelPath. In memory this is
    # always absolute; on disk it's stored relative to the character's folder whenever
    # it lives inside it (see CharacterStore.save/_resolve_relative_paths), so the
    # folder stays portable across machines.
    model_path: str = ""
    # The originally picked file's name (e.g. "Tomoko.kkm") — what the editor displays,
    # same as the pocket voice's clipName.
    model_name: str = ""
    # Emotion vocabulary the model supports, captured at creation via Character.InspectModelEmotions.
    available_emotions: list[str] = field(default_factory=list)
    # KK outfit/coordinate list (loaded from the KK_Coordinates.json inside a .kkm model),
    # with user-edited name/description per outfit. Empty for VRM models. Each entry:
    # {index:int, name:str, description:str, screenshots:{front?, back?}}.
    coordinates: list[dict] = field(default_factory=list)
    # Stable coordinate index of the outfit a NEW chat starts in (like the initial
    # emotion). -1 = unset → the first outfit.
    default_outfit_index: int = -1
    # Per-provider TTS voice, keyed by provider ('elevenlabs' | 'pocket'). At most one entry per
    # provider. ElevenLabs: {"voiceId": str}. Pocket: {"clipHash": str, "embeddingFile": abs path,
    # "clipName": str} — the embedding is encoded from a reference clip at save time and stored in
    # the character's folder (voice_<hash>.npy). The active TTS provider decides which entry is
    # used at synth time.
    voices: dict = field(default_factory=dict)
    created_at_utc: str = ""
    updated_at_utc: str = ""

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "displayName": self.display_name,
            "characterDefinition": self.character_definition,
            "initialScenario": self.initial_scenario,
            "initialAssistantMessage": self.initial_assistant_message,
            "systemPromptTemplate": self.system_prompt_template,
            "initialEmotionLabel": self.initial_emotion_label,
            "modelPath": self.model_path,
            "modelName": self.model_name,
            "availableEmotions": list(self.available_emotions),
            "coordinates": [dict(c) for c in self.coordinates],
            "defaultOutfitIndex": self.default_outfit_index,
            "voices": dict(self.voices),
            "createdAtUtc": self.created_at_utc,
            "updatedAtUtc": self.updated_at_utc,
        }

    @classmethod
    def from_wire(cls, d: dict) -> "CharacterRecord":
        return cls(
            id=str(d.get("id") or ""),
            name=str(d.get("name") or ""),
            display_name=str(d.get("displayName") or ""),
            character_definition=str(d.get("characterDefinition") or ""),
            initial_scenario=str(d.get("initialScenario") or ""),
            initial_assistant_message=str(d.get("initialAssistantMessage") or "Hello!"),
            system_prompt_template=str(d.get("systemPromptTemplate") or ""),
            initial_emotion_label=str(d.get("initialEmotionLabel") or "Neutral"),
            model_path=str(d.get("modelPath") or ""),
            model_name=str(d.get("modelName") or ""),
            available_emotions=[str(e) for e in (d.get("availableEmotions") or []) if isinstance(e, str)],
            coordinates=coerce_coordinates(d.get("coordinates")),
            default_outfit_index=int(d["defaultOutfitIndex"]) if isinstance(d.get("defaultOutfitIndex"), (int, float)) else -1,
            voices=dict(d.get("voices")) if isinstance(d.get("voices"), dict) else {},
            created_at_utc=str(d.get("createdAtUtc") or ""),
            updated_at_utc=str(d.get("updatedAtUtc") or ""),
        )


class CharacterStore:
    """CRUD over per-character folders (`<charId>/character.json` + the character's model
    copy and voice embedding) in the characters directory."""

    def __init__(self, characters_dir: Path | str | None = None):
        self.dir = Path(characters_dir) if characters_dir else _DEFAULT_CHARACTERS_DIR
        self.dir.mkdir(parents=True, exist_ok=True)

    def folder_for(self, char_id: str) -> Path:
        """The character's own folder — everything it owns (definition JSON, model copy,
        voice embedding) lives inside, so deleting the folder deletes the character."""
        stem = _SAFE_NAME_RE.sub("_", (char_id or "").strip()) or "character"
        return self.dir / stem

    def _path_for(self, char_id: str) -> Path:
        return self.folder_for(char_id) / "character.json"

    def _resolve_relative_paths(self, rec: CharacterRecord, folder: Path) -> CharacterRecord:
        """Bundled characters (shipped with the repo) carry folder-relative model /
        embedding paths so they survive a clone at any location; live saves write
        absolute paths. Resolve relatives against the character's own folder at load
        time so the rest of the app only ever sees absolute paths."""
        if rec.model_path and not Path(rec.model_path).is_absolute():
            rec.model_path = str(folder / rec.model_path).replace("\\", "/")
        pk = rec.voices.get("pocket") if isinstance(rec.voices, dict) else None
        if isinstance(pk, dict):
            emb = str(pk.get("embeddingFile") or "")
            if emb and not Path(emb).is_absolute():
                pk["embeddingFile"] = str(folder / emb).replace("\\", "/")
        return rec

    def list_records(self) -> list[CharacterRecord]:
        """All character records (skips files with no `id`), sorted by display name then name."""
        records: list[CharacterRecord] = []
        for path in sorted(self.dir.glob("*/character.json")):
            try:
                with path.open("r", encoding="utf-8") as f:
                    rec = self._resolve_relative_paths(CharacterRecord.from_wire(json.load(f)), path.parent)
                if rec.id:
                    records.append(rec)
                else:
                    print(f"[CharacterStore] Skipping '{path.parent.name}/' — no id (recreate it).", file=sys.stderr)
            except Exception as e:  # noqa: BLE001 — one bad file shouldn't hide the rest
                print(f"[CharacterStore] Skipping unreadable '{path.parent.name}/': {e}", file=sys.stderr)
        # Creation order (oldest first) — with duplicates from an import, the rightmost
        # card in the picker is reliably the newest, so the user can tell them apart.
        records.sort(key=lambda r: (r.created_at_utc or "", r.name.casefold()))
        return records

    def load(self, char_id: str) -> CharacterRecord | None:
        if not char_id:
            return None
        path = self._path_for(char_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                return self._resolve_relative_paths(CharacterRecord.from_wire(json.load(f)), path.parent)
        except Exception as e:  # noqa: BLE001
            print(f"[CharacterStore] Failed to read '{path.name}': {e}", file=sys.stderr)
            return None

    def exists(self, char_id: str) -> bool:
        return bool(char_id) and self._path_for(char_id).exists()

    @staticmethod
    def _relativize(value: str, folder: Path) -> str:
        """Inverse of _resolve_relative_paths for the on-disk JSON: an absolute path
        that lives inside the character's own folder is stored relative to it (the
        character's files always sit beside character.json, so this keeps the folder
        portable — clones, moved checkouts, bundled characters). Paths outside the
        folder (or unresolvable ones) are kept as-is."""
        if not value:
            return value
        try:
            p = Path(value)
            if p.is_absolute():
                return p.resolve().relative_to(folder.resolve()).as_posix()
        except (ValueError, OSError):
            pass
        return value

    def save(self, record: CharacterRecord) -> None:
        if not record.id:
            raise ValueError("CharacterRecord.id is required")
        now = _utc_now()
        if not record.created_at_utc:
            record.created_at_utc = now
        record.updated_at_utc = now

        # Relativize only the wire dict, not the caller's record — in-memory records
        # keep absolute paths (Session.Begin.modelPath, TTS embedding loads, …).
        folder = self.folder_for(record.id)
        wire = record.to_wire()
        wire["modelPath"] = self._relativize(wire.get("modelPath") or "", folder)
        pk = wire.get("voices", {}).get("pocket")
        if isinstance(pk, dict) and pk.get("embeddingFile"):
            pk["embeddingFile"] = self._relativize(str(pk["embeddingFile"]), folder)

        path = self._path_for(record.id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(wire, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on the same filesystem

    def delete(self, char_id: str) -> bool:
        """Remove the character's whole folder — definition, model copy and voice embedding
        go together."""
        folder = self.folder_for(char_id)
        if not char_id or not folder.is_dir():
            return False
        shutil.rmtree(folder, ignore_errors=True)
        return True

    # ------------------------------------------------------------------------
    # Character bundles (.wcc) — export/import everything a character needs
    # ------------------------------------------------------------------------

    def export_bundle(self, char_id: str, dest_path: str | os.PathLike) -> str | None:
        """Zip the character — character.json, model file, pocket voice embedding,
        profile picture — into `dest_path` so it can be imported ready-to-chat on
        another install. Returns an error string, or None on success."""
        rec = self.load(char_id)
        if rec is None:
            return "Character not found."
        model_src = Path(rec.model_path) if rec.model_path else None
        if model_src is None or not model_src.exists():
            return "The character's model file is missing — re-pick it in the editor first."
        entries: list[tuple[Path, str]] = [(model_src, model_src.name)]
        pk = rec.voices.get("pocket") if isinstance(rec.voices, dict) else None
        if isinstance(pk, dict):
            emb = Path(str(pk.get("embeddingFile") or ""))
            if str(emb) != "." and emb.exists():
                entries.append((emb, emb.name))
        pfp = self.folder_for(char_id) / "profile.jpg"
        if pfp.exists():
            entries.append((pfp, "profile.jpg"))
        try:
            with zipfile.ZipFile(dest_path, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("character.json", json.dumps(rec.to_wire(), ensure_ascii=False, indent=2))
                for src, arc in entries:
                    zf.write(src, arc)
        except OSError as e:
            return f"Couldn't write the bundle: {e}"
        return None

    def import_bundle(self, src_path: str | os.PathLike) -> tuple[CharacterRecord | None, str | None]:
        """Import a .wcc bundle into a fresh character folder. A NEW id is always minted
        (importing the same bundle twice yields two independent characters) and the
        record's machine-local paths (model, voice embedding) are rewritten to the new
        folder. Returns (record, None) on success, (None, error) on failure."""
        try:
            zf = zipfile.ZipFile(src_path)
        except (OSError, zipfile.BadZipFile) as e:
            return None, f"Couldn't open the bundle: {e}"
        folder: Path | None = None
        try:
            with zf:
                # Basename → entry map: bundles are flat, but tolerate nested entries.
                names = {Path(n).name: n for n in zf.namelist() if not n.endswith("/")}
                if "character.json" not in names:
                    return None, "Not a character bundle (no character.json inside)."
                try:
                    rec = CharacterRecord.from_wire(json.loads(zf.read(names["character.json"]).decode("utf-8")))
                except (ValueError, UnicodeDecodeError) as e:
                    return None, f"Unreadable character.json in the bundle: {e}"
                if not rec.name:
                    return None, "The bundle's character has no name."
                old_model_name = Path(rec.model_path).name if rec.model_path else ""
                rec.id = new_character_id()
                rec.created_at_utc = ""  # fresh timestamps on this install
                folder = self.folder_for(rec.id)
                folder.mkdir(parents=True, exist_ok=True)
                # Model — the entry the record referenced, else any model-typed entry.
                model_entry = names.get(old_model_name) or next(
                    (n for base, n in names.items() if base.lower().endswith((".vrm", ".kkm"))), None)
                if model_entry is None:
                    shutil.rmtree(folder, ignore_errors=True)
                    return None, "The bundle has no model file."
                data = zf.read(model_entry)
                model_path = folder / f"model_{hashlib.sha256(data).hexdigest()[:16]}{Path(model_entry).suffix.lower()}"
                model_path.write_bytes(data)
                rec.model_path = str(model_path).replace("\\", "/")
                rec.model_name = rec.model_name or Path(model_entry).name
                # Pocket voice embedding — rewrite to the new folder, or drop a dead entry.
                pk = rec.voices.get("pocket") if isinstance(rec.voices, dict) else None
                if isinstance(pk, dict):
                    emb_name = Path(str(pk.get("embeddingFile") or "")).name
                    entry = names.get(emb_name) or next(
                        (n for base, n in names.items() if base.lower().endswith(".npy")), None)
                    if entry is not None:
                        data = zf.read(entry)
                        clip_hash = str(pk.get("clipHash") or "") or hashlib.sha256(data).hexdigest()[:16]
                        emb_path = folder / f"voice_{clip_hash}.npy"
                        emb_path.write_bytes(data)
                        pk["embeddingFile"] = str(emb_path).replace("\\", "/")
                    else:
                        rec.voices.pop("pocket", None)
                if "profile.jpg" in names:
                    (folder / "profile.jpg").write_bytes(zf.read(names["profile.jpg"]))
        except (OSError, zipfile.BadZipFile) as e:
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            return None, f"Import failed: {e}"
        self.save(rec)
        return rec, None
