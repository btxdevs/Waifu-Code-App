import { useCallback, useEffect, useState } from 'react';
import { FolderOpen, Loader2, Plus, Trash2, Volume2, VolumeX, X } from 'lucide-react';
import type { ChatSettings } from '../payloads/chat';

interface Props {
  /** 'create' fetches defaults from the global config; 'edit' prefills from `initial`. */
  mode: 'create' | 'edit';
  /** Required in edit mode — the active chat's current settings to prefill. */
  initial?: ChatSettings;
  /** Providers the character actually has a voice configured for ("pocket"/"elevenlabs").
   *  Only these are selectable; an empty list means voice can't be enabled at all. */
  availableProviders: string[];
  onCancel: () => void;
  onConfirm: (settings: ChatSettings) => void;
}

const PROVIDER_LABELS: Record<string, string> = {
  pocket: 'Pocket TTS (local)',
  elevenlabs: 'ElevenLabs (HTTP API)',
};

/** Modal shown when starting a new chat (and re-openable from the sidebar to edit the
 *  active chat). Captures the per-chat user name, voice mode, voice provider, and the
 *  workspace folders the file tools may use for THIS chat. In create mode the fields are
 *  prefilled from the global config (window.app.getConfig). The chosen values are
 *  saved with the chat and applied to its session. */
export function NewChatDialog({ mode, initial, availableProviders, onCancel, onConfirm }: Props) {
  const [userName, setUserName] = useState(initial?.userName ?? '');
  const [voiceMode, setVoiceMode] = useState(initial?.voiceMode ?? true);
  const [voiceProvider, setVoiceProvider] = useState(initial?.voiceProvider ?? 'pocket');
  const [roots, setRoots] = useState<string[]>(initial?.workspaceRoots ?? []);
  // LLM configs to pick from (loaded from the global config) + the selected id for this chat.
  const [llmConfigs, setLlmConfigs] = useState<{ id: string; name: string }[]>([]);
  const [llmConfigId, setLlmConfigId] = useState(initial?.llmConfigId ?? '');
  // Create mode fetches defaults from config; edit mode is ready immediately.
  const [loading, setLoading] = useState(mode === 'create');

  // A character with no configured voice can't run in voice mode at all.
  const hasVoice = availableProviders.length > 0;

  // Keep the selected provider valid: clamp to one the character actually has, and force
  // voice off when the character has no voice configured.
  useEffect(() => {
    if (!hasVoice) {
      setVoiceMode(false);
      return;
    }
    setVoiceProvider(p => (availableProviders.includes(p) ? p : availableProviders[0]));
  }, [hasVoice, availableProviders]);

  useEffect(() => {
    let cancelled = false;
    window.app.getConfig()
      .then(c => {
        if (cancelled || !c) return;
        // LLM config picker — load the available configs in BOTH modes. Default the selection to
        // the chat's saved config (edit) or the global default (create); fall back to the default
        // when the saved id no longer exists.
        const configs = (c.llm?.configs ?? []).map(x => ({ id: x.id, name: x.name }));
        setLlmConfigs(configs);
        setLlmConfigId(prev => {
          const wanted = (mode === 'edit' ? (initial?.llmConfigId || '') : '') || prev || c.llm?.defaultId || '';
          return configs.some(x => x.id === wanted) ? wanted : (c.llm?.defaultId || configs[0]?.id || '');
        });
        // The remaining defaults are only seeded for a brand-new chat; edit mode keeps `initial`.
        if (mode !== 'create') return;
        setUserName(c.user_name || '');
        // Prefer the global config's provider, but only if THIS character has a voice for it.
        // Setting the raw config provider unconditionally can leave an unavailable provider
        // selected (the clamp effect above only re-runs on a prop change, not on this async
        // set), which the backend then switches to, finds no voice for, and reports the chat
        // as voice-unavailable — disabling the in-chat toggle.
        setVoiceProvider(prev => {
          const preferred = c.tts?.provider;
          if (preferred && availableProviders.includes(preferred)) return preferred;
          return availableProviders.includes(prev) ? prev : (availableProviders[0] ?? prev);
        });
        setRoots(c.workspace?.allowedRoots ?? []);
        if (hasVoice) setVoiceMode(true);
      })
      .catch(() => { /* fall back to empty defaults */ })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [mode]);

  const addRoot = useCallback(async () => {
    const picked = await window.app.pickDirectory();
    if (!picked) return;
    setRoots(prev => (prev.includes(picked) ? prev : [...prev, picked]));
  }, []);

  const browseRoot = useCallback(async (index: number) => {
    const picked = await window.app.pickDirectory(roots[index]);
    if (!picked) return;
    setRoots(prev => { const next = prev.slice(); next[index] = picked; return next; });
  }, [roots]);

  const removeRoot = useCallback((index: number) => {
    setRoots(prev => { const next = prev.slice(); next.splice(index, 1); return next; });
  }, []);

  const confirm = useCallback(() => {
    onConfirm({
      userName: userName.trim() || 'User',
      voiceMode: hasVoice && voiceMode,
      // No configured voice → leave the provider unset so we don't force a needless TTS switch.
      voiceProvider: hasVoice ? voiceProvider : '',
      llmConfigId,
      workspaceRoots: roots.map(r => r.trim()).filter(Boolean),
    });
  }, [userName, hasVoice, voiceMode, voiceProvider, llmConfigId, roots, onConfirm]);

  const title = mode === 'create' ? 'New chat' : 'Chat settings';
  const confirmLabel = mode === 'create' ? 'Start chat' : 'Save';

  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>{title}</h2>
          <button className="icon-btn" onClick={onCancel} title="Cancel" aria-label="Cancel">
            <X size={16} />
          </button>
        </div>

        {loading ? (
          <div className="dialog-body config-loading"><Loader2 size={16} className="spin" /> Loading…</div>
        ) : (
          <div className="dialog-body">
            <label className="config-field">
              <span>Your name</span>
              <input
                type="text"
                value={userName}
                onChange={e => setUserName(e.target.value)}
                placeholder="User"
                autoFocus
              />
              <span className="config-hint">Shown as your speaker label and substituted into the system prompt.</span>
            </label>

            {llmConfigs.length > 0 && (
              <label className="config-field">
                <span>LLM config</span>
                <select value={llmConfigId} onChange={e => setLlmConfigId(e.target.value)}>
                  {llmConfigs.map(cfg => (
                    <option key={cfg.id} value={cfg.id}>{cfg.name || cfg.id}</option>
                  ))}
                </select>
                <span className="config-hint">
                  Which model/endpoint this chat talks to. Manage the available configs in Settings → LLM.
                </span>
              </label>
            )}

            <label className="config-field">
              <span>Assistant voice</span>
              <button
                type="button"
                className={`dialog-toggle ${voiceMode ? 'on' : 'off'}`}
                onClick={() => setVoiceMode(v => !v)}
                disabled={!hasVoice}
              >
                {voiceMode ? <Volume2 size={14} /> : <VolumeX size={14} />}
                {voiceMode ? 'Voice — speak replies aloud' : 'No voice — lip-sync to text'}
              </button>
              <span className="config-hint">
                {hasVoice
                  ? 'Speaks replies aloud using the provider below.'
                  : 'This character has no voice configured — add one in the character editor to enable voice.'}
              </span>
            </label>

            {hasVoice && (
              <label className="config-field">
                <span>Voice provider</span>
                <select value={voiceProvider} onChange={e => setVoiceProvider(e.target.value)}>
                  {availableProviders.map(p => (
                    <option key={p} value={p}>{PROVIDER_LABELS[p] ?? p}</option>
                  ))}
                </select>
                <span className="config-hint">
                  Only providers this character has a voice for are listed. Switching the active engine
                  on a fresh provider reloads its model, so the first reply may lag a moment.
                </span>
              </label>
            )}

            <div className="config-field">
              <span>Workspace folders</span>
              <span className="config-hint">
                Folders the assistant's file tools may read/write for this chat. The first entry is the
                default working directory. Replaces the global config roots while this chat is open.
              </span>
              <ul className="config-list">
                {roots.map((root, i) => (
                  <li key={i} className="config-list-row">
                    <input
                      type="text"
                      value={root}
                      onChange={e => setRoots(prev => { const n = prev.slice(); n[i] = e.target.value; return n; })}
                      placeholder="C:/Path/To/Workspace"
                    />
                    <button className="icon-btn" onClick={() => browseRoot(i)} title="Browse for folder" aria-label="Browse for folder">
                      <FolderOpen size={16} />
                    </button>
                    <button className="icon-btn icon-btn-danger" onClick={() => removeRoot(i)} title="Remove this folder" aria-label="Remove this folder">
                      <Trash2 size={16} />
                    </button>
                  </li>
                ))}
                {roots.length === 0 && (
                  <li className="config-empty">No folders. The assistant's file tools will be unavailable.</li>
                )}
              </ul>
              <button className="config-add" onClick={addRoot}>
                <Plus size={14} /> Add folder…
              </button>
            </div>
          </div>
        )}

        <div className="dialog-footer">
          <button className="dialog-btn" onClick={onCancel}>Cancel</button>
          <button className="dialog-btn primary" onClick={confirm} disabled={loading}>{confirmLabel}</button>
        </div>
      </div>
    </div>
  );
}
