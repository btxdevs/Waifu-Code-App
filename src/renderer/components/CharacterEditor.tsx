import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft, FolderOpen, Loader2, Mic, Plus, Trash2, User } from 'lucide-react';
import {
  TYPE_CHAT_SAVE_CHARACTER,
  TYPE_CHAT_INSPECT_MODEL_EMOTIONS,
  TYPE_CHAT_MODEL_EMOTIONS,
  TYPE_CHAT_INSPECT_MODEL_COORDINATES,
  TYPE_CHAT_MODEL_COORDINATES,
  type Envelope,
} from '../protocol';
import type { CharacterRecordWire, CharacterVoices, KkCoordinate } from '../payloads/character';
import type { ChatModelEmotionsPayload, ChatModelCoordinatesPayload } from '../payloads/chat';

/** KK models (.kkm) ship an outfit/coordinate list inside them; VRM models don't. */
const isKkModel = (p: string) => /\.kkm$/i.test(p);

/** A coordinate name is "unlabelled" when it's blank or the exporter's default "Outfit NN"
 *  (the KK outfit folder name). Those carry no user intent worth keeping over a real edit. */
const isDefaultOutfitName = (name: string) => {
  const n = (name || '').trim();
  return !n || /^outfit\s*\d+$/i.test(n);
};

/** Merge a freshly-loaded kkm coordinate list onto the ones already in the editor, matched by
 *  index. When the incoming name/description is empty or a default "Outfit NN", keep whatever the
 *  user had already typed for that index — so re-picking a model (or the same one) doesn't wipe
 *  edited labels. A real incoming name/description always wins. */
const mergeCoordinates = (prev: KkCoordinate[], incoming: KkCoordinate[]): KkCoordinate[] => {
  const prevByIndex = new Map(prev.map(c => [c.index, c]));
  return incoming.map(inc => {
    const old = prevByIndex.get(inc.index);
    if (!old) return inc;
    const name = isDefaultOutfitName(inc.name) && !isDefaultOutfitName(old.name) ? old.name : inc.name;
    const description = !(inc.description || '').trim() && (old.description || '').trim()
      ? old.description
      : inc.description;
    return { ...inc, name, description };
  });
};

/** Voice providers selectable in the editor, in display order. Each character may have at
 *  most one voice per provider; the active TTS provider (Settings) decides which is used. */
const VOICE_PROVIDERS: { key: 'pocket' | 'elevenlabs'; label: string }[] = [
  { key: 'pocket', label: 'Pocket TTS' },
  { key: 'elevenlabs', label: 'ElevenLabs' },
];

const baseName = (p: string) => p.split(/[\\/]/).pop() || p;

interface Props {
  onClose: () => void;
  /** Called after the save is dispatched, with the just-edited record (id present on edits). Lets the
   *  parent optimistically refresh its picker list so reopening the editor reflects the change right
   *  away, instead of waiting for the async save+voice-encode round-trip to push a fresh list. */
  onSaved?: (record: CharacterRecordWire) => void;
  /** When set, the form opens in EDIT mode prefilled from this record (same id is kept on save). */
  character?: CharacterRecordWire | null;
}

/**
 * Create or edit a app-owned character: fill in the text fields, pick a .vrm model, and the
 * model's supported emotions are detected (Unity inspects the file without loading it) and stored
 * with the character. The chat then constrains the LLM to exactly those emotions. Editing keeps the
 * character's stable id, so existing chats stay attached.
 */
export function CharacterEditor({ onClose, onSaved, character }: Props) {
  const editing = !!character;
  const [name, setName] = useState(character?.name ?? '');
  const [displayName, setDisplayName] = useState(character?.displayName ?? '');
  // Profile picture as a data URL of the processed JPEG (the backend picker crops/encodes
  // before returning, so what's previewed here is exactly what gets stored). '' = none.
  const [profileImage, setProfileImage] = useState(character?.profileImage ?? '');
  const [definition, setDefinition] = useState(character?.characterDefinition
    ?? '{{char}} is a friendly, helpful assistant. She is warm and upbeat, speaks casually like a close friend, and gives short, direct answers. She is happy to help {{user}} with anything.');
  const [scenario, setScenario] = useState(character?.initialScenario ?? '');
  const [greeting, setGreeting] = useState(character?.initialAssistantMessage ?? 'Hello!');
  const [systemPrompt, setSystemPrompt] = useState(character?.systemPromptTemplate ?? '');

  const [modelPath, setModelPath] = useState(character?.modelPath ?? '');
  // Shown instead of the path (which points at the app-owned copy after a save) — the
  // originally picked file's name, same treatment as the pocket voice clip.
  const [modelName, setModelName] = useState(character?.modelName ?? '');
  const [emotions, setEmotions] = useState<string[]>(character?.availableEmotions ?? []);
  const [initialEmotion, setInitialEmotion] = useState(character?.initialEmotionLabel ?? 'Neutral');
  const [inspecting, setInspecting] = useState(false);
  const [emotionError, setEmotionError] = useState('');

  // KK outfits/coordinates loaded from the KK_Coordinates.json inside a .kkm model. Each has a
  // user-editable name + description; index/screenshots are preserved. Empty for VRM models.
  const [coordinates, setCoordinates] = useState<KkCoordinate[]>(character?.coordinates ?? []);
  const [coordsError, setCoordsError] = useState('');
  // Stable coordinate index of the outfit a new chat starts in (like the initial emotion).
  // -1 = unset. Resolved against the current coordinate list at render/save time, so a
  // model re-pick that drops the chosen index falls back to the first outfit.
  const [defaultOutfitIndex, setDefaultOutfitIndex] = useState<number>(character?.defaultOutfitIndex ?? -1);
  const effectiveDefaultOutfit = coordinates.some(c => c.index === defaultOutfitIndex)
    ? defaultOutfitIndex
    : (coordinates[0]?.index ?? -1);

  // Per-provider voices. On edit this prefills from the stored record; pocket entries
  // there describe an already-encoded embedding (no clipPath until a new clip is picked).
  const [voices, setVoices] = useState<CharacterVoices>(character?.voices ?? {});

  // Receive the inspected emotion set for the picked model (Python relays it from Unity).
  useEffect(() => {
    const off = window.app.onChatEvent((env) => {
      if (env.type !== TYPE_CHAT_MODEL_EMOTIONS) return;
      const p = (env as Envelope).payload as ChatModelEmotionsPayload;
      setInspecting(false);
      setEmotionError(p.error || '');
      const list = p.emotions ?? [];
      setEmotions(list);
      setInitialEmotion(prev => (list.length && !list.includes(prev) ? list[0] : prev));
    });
    return () => off();
  }, []);

  // Receive the KK outfit/coordinate list for the picked model (Python reads it from the
  // KK_Coordinates.json beside the model file).
  useEffect(() => {
    const off = window.app.onChatEvent((env) => {
      if (env.type !== TYPE_CHAT_MODEL_COORDINATES) return;
      const p = (env as Envelope).payload as ChatModelCoordinatesPayload;
      setCoordsError(p.error || '');
      // Merge (don't replace): keep edited names/descriptions where the kkm's are empty/default.
      setCoordinates(prev => mergeCoordinates(prev, p.coordinates ?? []));
    });
    return () => off();
  }, []);

  // On opening an edit for a KK character that has no stored coordinates yet (e.g. saved before
  // this feature), load them from the model's KK_Coordinates.json so they can be labelled.
  useEffect(() => {
    if (isKkModel(modelPath) && coordinates.length === 0) {
      window.app.sendChat(TYPE_CHAT_INSPECT_MODEL_COORDINATES, { modelPath });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const browseModel = useCallback(async () => {
    const picked = await window.app.pickVrmFile(modelPath || undefined);
    if (!picked) return;
    setModelPath(picked);
    setModelName(baseName(picked));
    setEmotions([]);
    setEmotionError('');
    setInspecting(true);
    window.app.sendChat(TYPE_CHAT_INSPECT_MODEL_EMOTIONS, { modelPath: picked });
    setCoordsError('');
    if (isKkModel(picked)) {
      // Keep the current (possibly edited) coordinates so the incoming kkm's empty/default
      // "Outfit NN" labels are merged onto them rather than clobbering them (see mergeCoordinates).
      window.app.sendChat(TYPE_CHAT_INSPECT_MODEL_COORDINATES, { modelPath: picked });
    } else {
      setCoordinates([]);  // VRM models have no outfits.
    }
  }, [modelPath]);

  const browseProfileImage = useCallback(async () => {
    const dataUrl = await window.app.pickProfileImage();
    if (dataUrl) setProfileImage(dataUrl);
  }, []);

  const updateCoordinate = useCallback((i: number, patch: Partial<KkCoordinate>) => {
    setCoordinates(cs => cs.map((c, idx) => (idx === i ? { ...c, ...patch } : c)));
  }, []);

  const addVoice = useCallback((provider: 'pocket' | 'elevenlabs') => {
    setVoices(v => ({ ...v, [provider]: provider === 'elevenlabs' ? { voiceId: '' } : {} }));
  }, []);
  const removeVoice = useCallback((provider: 'pocket' | 'elevenlabs') => {
    setVoices(v => { const next = { ...v }; delete next[provider]; return next; });
  }, []);
  const pickPocketClip = useCallback(async () => {
    const picked = await window.app.pickAudioFile(voices.pocket?.clipPath || undefined);
    if (!picked) return;
    // A freshly picked clip clears the stored hash/embedding so the backend re-encodes it.
    setVoices(v => ({ ...v, pocket: { clipPath: picked, clipName: baseName(picked) } }));
  }, [voices.pocket]);

  const canSave = name.trim().length > 0 && modelPath.length > 0 && emotions.length > 0
    && definition.trim().length > 0 && greeting.trim().length > 0 && !inspecting;

  const save = useCallback(() => {
    if (!canSave) return;
    const cleanName = name.trim();
    // The just-edited record. Sent to the backend AND handed to the parent for an optimistic list
    // refresh — the backend's own refreshed list arrives only after the (multi-second) pocket-voice
    // encode, so without this a quick reopen would show pre-save values.
    const record: CharacterRecordWire = {
      id: character?.id ?? '',
      name: cleanName,
      displayName: displayName.trim() || cleanName,
      profileImage,
      characterDefinition: definition,
      initialScenario: scenario,
      initialAssistantMessage: greeting || 'Hello!',
      systemPromptTemplate: systemPrompt,
      initialEmotionLabel: initialEmotion || 'Neutral',
      modelPath,
      modelName: modelName || baseName(modelPath),
      availableEmotions: emotions,
      // KK outfit labels (empty for VRM models). Persisted with the character.
      coordinates,
      defaultOutfitIndex: effectiveDefaultOutfit,
      // Per-provider voices. Pocket entries with a new clipPath get re-encoded backend-side;
      // entries without one keep their existing embedding.
      voices,
    };
    window.app.sendChat(TYPE_CHAT_SAVE_CHARACTER, {
      ...record,
      // id present → edit the existing character (keeps it attached to its chats); absent → create.
      id: character?.id,
    });
    onSaved?.(record);
    onClose();
  }, [canSave, character, name, displayName, profileImage, definition, scenario, greeting,
      systemPrompt, initialEmotion, modelPath, modelName, emotions, coordinates,
      effectiveDefaultOutfit, voices, onSaved, onClose]);

  const modelHint = inspecting
    ? <span className="config-hint"><Loader2 size={12} className="spin" /> Reading the model’s emotions…</span>
    : emotionError
      ? <span className="config-hint config-error">Couldn’t read emotions: {emotionError}</span>
      : emotions.length
        ? <span className="config-hint">{emotions.length} emotions supported: {emotions.join(', ')}</span>
        : <span className="config-hint">Pick a model (.vrm / .kkm) to detect the emotions it supports.</span>;

  return (
    <div className="config-panel">
      <div className="config-header">
        <button className="icon-btn" onClick={onClose} title="Back" aria-label="Back">
          <ArrowLeft size={16} />
        </button>
        <h1>{editing ? 'Edit Character' : 'New Character'}</h1>
        <div className="config-header-right">
          <button className="config-save" onClick={save} disabled={!canSave}>{editing ? 'Save' : 'Create'}</button>
        </div>
      </div>

      <div className="config-body">
        <section className="config-section">
          <h2>Identity</h2>
          <label className="config-field">
            <span>Name <span className="config-required">*</span></span>
            <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Touka" />
          </label>
          <label className="config-field">
            <span>Display name</span>
            <input type="text" value={displayName} onChange={e => setDisplayName(e.target.value)}
                   placeholder="(defaults to the name)" />
          </label>
          {/* A <div> (not <label>) so the buttons don't become the field's associated control. */}
          <div className="config-field">
            <span>Profile picture</span>
            <div className="config-row">
              <span className="config-avatar">
                {profileImage ? <img src={profileImage} alt="" /> : <User size={22} />}
              </span>
              <button className="icon-btn" onClick={browseProfileImage}
                      title="Browse for an image" aria-label="Browse for an image">
                <FolderOpen size={16} />
              </button>
              {profileImage && (
                <button className="icon-btn icon-btn-danger" onClick={() => setProfileImage('')}
                        title="Remove profile picture" aria-label="Remove profile picture">
                  <Trash2 size={14} />
                </button>
              )}
            </div>
            <span className="config-hint">
              Pick any image.
            </span>
          </div>
        </section>

        <section className="config-section">
          <h2>Model</h2>
          <label className="config-field">
            <span>Model <span className="config-required">*</span></span>
            <div className="config-row">
              <input type="text" value={modelName || baseName(modelPath)} readOnly placeholder="No model selected" />
              <button className="icon-btn" onClick={browseModel} title="Browse for a model (.vrm / .kkm)" aria-label="Browse for a model">
                <FolderOpen size={16} />
              </button>
            </div>
            {modelHint}
          </label>
          <label className="config-field">
            <span>Initial emotion</span>
            <select value={initialEmotion} onChange={e => setInitialEmotion(e.target.value)} disabled={!emotions.length}>
              {emotions.length === 0 && <option value="Neutral">Neutral</option>}
              {emotions.map(em => <option key={em} value={em}>{em}</option>)}
            </select>
          </label>
        </section>

        {isKkModel(modelPath) && (
          <section className="config-section">
            <h2>Outfits</h2>
            {coordinates.length === 0 ? (
              <span className="config-hint">
                {coordsError
                  ? `Couldn’t read outfits: ${coordsError}`
                  : 'No KK_Coordinates.json found beside this model — no outfits to label.'}
              </span>
            ) : (
              <>
                <span className="config-hint">
                  Name and describe each outfit from this KK model — saved with the character.
                </span>
                {coordinates.length > 1 && (
                  <label className="config-field">
                    <span>Default outfit</span>
                    <select
                      value={String(effectiveDefaultOutfit)}
                      onChange={e => setDefaultOutfitIndex(Number(e.target.value))}
                    >
                      {coordinates.map(c => (
                        <option key={c.index} value={String(c.index)}>
                          {c.name || `Outfit ${c.index}`}
                        </option>
                      ))}
                    </select>
                    <span className="config-hint">The outfit a new chat starts in.</span>
                  </label>
                )}
                {coordinates.map((c, i) => (
                  // A <div> (not <label>) so the description textarea isn't auto-associated with
                  // the name input.
                  <div className="config-field" key={c.index}>
                    <div className="config-field-head"><span>Outfit {c.index}</span></div>
                    <input
                      type="text"
                      value={c.name}
                      onChange={e => updateCoordinate(i, { name: e.target.value })}
                      placeholder={`Outfit ${c.index}`}
                    />
                    <textarea
                      rows={2}
                      value={c.description}
                      onChange={e => updateCoordinate(i, { description: e.target.value })}
                      placeholder="Description (when/where this outfit is worn…)"
                    />
                  </div>
                ))}
              </>
            )}
          </section>
        )}

        <section className="config-section">
          <h2>Voice</h2>
          <span className="config-hint">
            One per provider. The active TTS engine (Settings → Voice) decides which
            voice is used; with no entry for it, that provider's default voice plays.
          </span>

          {voices.elevenlabs && (
            // A <div> (not <label>) so the trash button — the first labelable descendant —
            // doesn't become the field's associated control, which would make hovering the
            // input/hint light up the delete button's hover state.
            <div className="config-field">
              <div className="config-field-head">
                <span>ElevenLabs voice id</span>
                <button className="icon-btn icon-btn-danger" onClick={() => removeVoice('elevenlabs')}
                        title="Remove ElevenLabs voice" aria-label="Remove ElevenLabs voice">
                  <Trash2 size={14} />
                </button>
              </div>
              <input
                type="text"
                value={voices.elevenlabs.voiceId ?? ''}
                onChange={e => setVoices(v => ({ ...v, elevenlabs: { voiceId: e.target.value } }))}
                placeholder="e.g. 21m00Tcm4TlvDq8ikWAM"
              />
              <span className="config-hint">The voice's id on your ElevenLabs-compatible server.</span>
            </div>
          )}

          {voices.pocket && (
            <div className="config-field">
              <div className="config-field-head">
                <span>Pocket TTS voice clip</span>
                <button className="icon-btn icon-btn-danger" onClick={() => removeVoice('pocket')}
                        title="Remove Pocket voice" aria-label="Remove Pocket voice">
                  <Trash2 size={14} />
                </button>
              </div>
              <div className="config-row">
                <input
                  type="text"
                  readOnly
                  value={voices.pocket.clipPath
                    ? baseName(voices.pocket.clipPath)
                    : (voices.pocket.clipName || '')}
                  placeholder="No clip selected"
                />
                <button className="icon-btn" onClick={pickPocketClip}
                        title="Browse for a voice clip" aria-label="Browse for a voice clip">
                  <FolderOpen size={16} />
                </button>
              </div>
              <span className="config-hint">
                {voices.pocket.clipPath
                  ? 'New clip — its embedding is generated when you save.'
                  : voices.pocket.embeddingFile
                    ? 'Using the saved embedding. Pick a new clip to replace it.'
                    : 'Pick a short, clean voice sample (wav/flac/ogg).'}
              </span>
            </div>
          )}

          <div className="config-row">
            {VOICE_PROVIDERS.filter(p => !voices[p.key]).map(p => (
              <button key={p.key} className="config-add" onClick={() => addVoice(p.key)}>
                <Plus size={14} /> {p.label}
              </button>
            ))}
            {VOICE_PROVIDERS.every(p => voices[p.key]) && (
              <span className="config-hint"><Mic size={12} /> Both providers configured.</span>
            )}
          </div>
        </section>

        <section className="config-section">
          <h2>Personality</h2>
          <span className="config-hint">
            {'You can use {{char}} (character name) and {{user}} (user name) in any of these fields — replaced when the chat prompt is built.'}
          </span>
          <label className="config-field">
            <span>Character definition <span className="config-required">*</span></span>
            <textarea rows={5} value={definition} onChange={e => setDefinition(e.target.value)}
                      placeholder="Personality, background, how they speak…" />
          </label>
          <label className="config-field">
            <span>Initial scenario</span>
            <textarea rows={2} value={scenario} onChange={e => setScenario(e.target.value)}
                      placeholder="Scene-setting context at the start of a chat." />
          </label>
          <label className="config-field">
            <span>First message <span className="config-required">*</span></span>
            <input type="text" value={greeting} onChange={e => setGreeting(e.target.value)} placeholder="Hello!" />
          </label>
          <label className="config-field">
            <span>System prompt template</span>
            <textarea rows={3} value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)}
                      placeholder="Leave empty to use the default template." />
            <span className="config-hint">Advanced — overrides the default system prompt for this character.</span>
          </label>
        </section>
      </div>
    </div>
  );
}
