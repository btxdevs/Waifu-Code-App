import { useCallback, useEffect, useState } from 'react';
import { ArrowLeft, Copy, FolderOpen, Loader2, Plus, Trash2, Eye, EyeOff } from 'lucide-react';
import type {
  AppConfig,
  ElevenLabsConfigBlock,
  LlmConfigBlock,
  PocketTtsConfigBlock,
  TtsConfigBlock,
  WorkspaceConfigBlock,
} from '../payloads/config';

interface Props {
  onClose: () => void;
}

/** In-chat-window settings panel. Loads on mount, edits a local copy of the
 *  config, and writes back via `window.app.saveConfig`. The two on-disk
 *  files (llm.config.json + app.config.json) are merged into one shape
 *  for the UI's convenience — the Python side preserves any keys this panel
 *  doesn't expose.
 *
 *  No live-apply: most fields (LLM URL/key, workspace roots) are read once at
 *  startup, so saving shows a "restart needed" note. */
export function ConfigView({ onClose }: Props) {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [showElevenKey, setShowElevenKey] = useState(false);
  // Which LLM config the LLM tab is currently editing (id into config.llm.configs).
  const [selectedLlmId, setSelectedLlmId] = useState<string>('');
  // Settings are split across tabs (the list grows as features are added). Add new tabs here.
  const [tab, setTab] = useState<'general' | 'llm' | 'tts'>('general');

  useEffect(() => {
    let cancelled = false;
    window.app.getConfig()
      .then(c => {
        if (cancelled) return;
        setConfig(c);
        setSelectedLlmId(c.llm?.defaultId || c.llm?.configs?.[0]?.id || '');
      })
      .catch(e => { if (!cancelled) setLoadError(String(e?.message ?? e)); });
    return () => { cancelled = true; };
  }, []);

  // Patch the CURRENTLY-SELECTED LLM config (within config.llm.configs).
  const patchLlm = useCallback((patch: Partial<LlmConfigBlock>) => {
    setConfig(c => {
      if (!c) return c;
      const configs = c.llm.configs.map(cfg =>
        cfg.id === selectedLlmId ? { ...cfg, ...patch } : cfg);
      return { ...c, llm: { ...c.llm, configs } };
    });
  }, [selectedLlmId]);

  const addLlmConfig = useCallback(() => {
    const id = (globalThis.crypto?.randomUUID?.() ?? `llm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
    setConfig(c => {
      if (!c) return c;
      const n = c.llm.configs.length + 1;
      const fresh: LlmConfigBlock = {
        id,
        name: `Config ${n}`,
        api_url: '',
        api_key: '',
        model: '',
        temperature: 1,
        request_timeout_seconds: 30,
        thinking: 'unset',
        send_system_prompt_as_user: false,
        supports_vision: false,
        max_tool_call_rounds: 0,
        vision_max_images: 10,
      };
      return { ...c, llm: { ...c.llm, configs: [...c.llm.configs, fresh] } };
    });
    setSelectedLlmId(id);
  }, []);

  const duplicateLlmConfig = useCallback((id: string) => {
    const newId = (globalThis.crypto?.randomUUID?.() ?? `llm_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`);
    setConfig(c => {
      if (!c) return c;
      const src = c.llm.configs.find(cfg => cfg.id === id);
      if (!src) return c;
      // Clone every field but the identity — same provider/key, ready to tweak the model.
      const copy: LlmConfigBlock = { ...src, id: newId, name: `${src.name || 'Config'} copy` };
      return { ...c, llm: { ...c.llm, configs: [...c.llm.configs, copy] } };
    });
    setSelectedLlmId(newId);
  }, []);

  const removeLlmConfig = useCallback((id: string) => {
    setConfig(c => {
      if (!c || c.llm.configs.length <= 1) return c; // never delete the last config
      const configs = c.llm.configs.filter(cfg => cfg.id !== id);
      const defaultId = c.llm.defaultId === id ? configs[0].id : c.llm.defaultId;
      // Keep the editor pointed at a valid config.
      setSelectedLlmId(prev => (prev === id ? configs[0].id : prev));
      return { ...c, llm: { ...c.llm, configs, defaultId } };
    });
  }, []);

  const setDefaultLlm = useCallback((id: string) => {
    setConfig(c => c ? { ...c, llm: { ...c.llm, defaultId: id } } : c);
  }, []);
  const patchWorkspace = useCallback((patch: Partial<WorkspaceConfigBlock>) => {
    setConfig(c => c ? { ...c, workspace: { ...c.workspace, ...patch } } : c);
  }, []);
  const patchUserName = useCallback((value: string) => {
    setConfig(c => c ? { ...c, user_name: value } : c);
  }, []);
  const patchTts = useCallback((patch: Partial<TtsConfigBlock>) => {
    setConfig(c => c ? { ...c, tts: { ...c.tts, ...patch } } : c);
  }, []);
  const patchEleven = useCallback((patch: Partial<ElevenLabsConfigBlock>) => {
    setConfig(c => c ? { ...c, tts: { ...c.tts, elevenlabs: { ...c.tts.elevenlabs, ...patch } } } : c);
  }, []);
  const patchPocket = useCallback((patch: Partial<PocketTtsConfigBlock>) => {
    setConfig(c => c ? { ...c, tts: { ...c.tts, pocket: { ...c.tts.pocket, ...patch } } } : c);
  }, []);

  const pickWorkspaceDir = useCallback(async (index: number) => {
    if (!config) return;
    const current = config.workspace.allowedRoots[index] ?? '';
    const picked = await window.app.pickDirectory(current);
    if (!picked) return;
    const next = config.workspace.allowedRoots.slice();
    next[index] = picked;
    patchWorkspace({ allowedRoots: next });
  }, [config, patchWorkspace]);

  const addWorkspaceRoot = useCallback(async () => {
    const picked = await window.app.pickDirectory();
    if (!picked) return;
    setConfig(c => c
      ? { ...c, workspace: { ...c.workspace, allowedRoots: [...c.workspace.allowedRoots, picked] } }
      : c);
  }, []);

  const removeWorkspaceRoot = useCallback((index: number) => {
    setConfig(c => {
      if (!c) return c;
      const next = c.workspace.allowedRoots.slice();
      next.splice(index, 1);
      return { ...c, workspace: { ...c.workspace, allowedRoots: next } };
    });
  }, []);

  const save = useCallback(async () => {
    if (!config || saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      const result = await window.app.saveConfig(config);
      if (result?.ok) {
        // No "Saved." flash — hot-reload makes the change live by the time we
        // return, so just bounce back to the chat surface.
        onClose();
        return;
      }
      setSaveError(result?.error || 'Save failed.');
    } catch (e) {
      setSaveError(String((e as Error)?.message ?? e));
    } finally {
      setSaving(false);
    }
  }, [config, saving, onClose]);

  if (loadError) {
    return (
      <div className="config-panel">
        <div className="config-header">
          <button className="icon-btn" onClick={onClose} title="Back to chat" aria-label="Back to chat">
            <ArrowLeft size={18} />
          </button>
          <h1>Settings</h1>
        </div>
        <div className="config-body">
          <div className="config-error">Failed to load config: {loadError}</div>
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="config-panel">
        <div className="config-header">
          <button className="icon-btn" onClick={onClose} title="Back to chat" aria-label="Back to chat">
            <ArrowLeft size={18} />
          </button>
          <h1>Settings</h1>
        </div>
        <div className="config-body config-loading">
          <Loader2 size={16} className="spin" /> Loading…
        </div>
      </div>
    );
  }

  // The LLM config the LLM tab edits — the selected one, falling back to the first.
  const selectedLlm = config.llm.configs.find(c => c.id === selectedLlmId) ?? config.llm.configs[0];
  const isDefaultLlm = !!selectedLlm && config.llm.defaultId === selectedLlm.id;

  return (
    <div className="config-panel">
      <div className="config-header">
        <button className="icon-btn" onClick={onClose} title="Back to chat" aria-label="Back to chat">
          <ArrowLeft size={18} />
        </button>
        <h1>Settings</h1>
        <div className="config-header-right">
          <button className="config-save" onClick={save} disabled={saving}>
            {saving ? (<><Loader2 size={14} className="spin" /> Saving…</>) : 'Save'}
          </button>
        </div>
      </div>

      <nav className="config-tabs">
        <button
          className={'config-tab' + (tab === 'general' ? ' active' : '')}
          onClick={() => setTab('general')}
        >
          General
        </button>
        <button
          className={'config-tab' + (tab === 'llm' ? ' active' : '')}
          onClick={() => setTab('llm')}
        >
          LLM
        </button>
        <button
          className={'config-tab' + (tab === 'tts' ? ' active' : '')}
          onClick={() => setTab('tts')}
        >
          TTS
        </button>
      </nav>

      <div className="config-body">
        {saveError && <div className="config-error">{saveError}</div>}

        {tab === 'general' && <>
        {/* ---- User ---- */}
        <section className="config-section">
          <h2>Identity</h2>
          <label className="config-field">
            <span>Display name</span>
            <input
              type="text"
              value={config.user_name}
              onChange={e => patchUserName(e.target.value)}
              placeholder="User"
            />
            <span className="config-hint">
              Substituted into <code>{'{{user}}'}</code> in the system prompt and shown as the
              speaker label on your chat bubbles.
            </span>
          </label>
        </section>
        </>}

        {tab === 'llm' && selectedLlm && <>
        {/* ---- LLM config picker ---- */}
        <section className="config-section">
          <h2>LLM configs</h2>
          <span className="config-hint">
            Define one or more model/endpoint configs. The default is preselected when you start a
            new chat; each chat can pick a different one (in the new-chat and chat-settings dialogs).
          </span>
          <label className="config-field">
            <span>Editing</span>
            <div className="config-row">
              <select
                style={{ flex: 1 }}
                value={selectedLlm.id}
                onChange={e => setSelectedLlmId(e.target.value)}
              >
                {config.llm.configs.map(cfg => (
                  <option key={cfg.id} value={cfg.id}>
                    {(cfg.name || cfg.id) + (config.llm.defaultId === cfg.id ? ' (default)' : '')}
                  </option>
                ))}
              </select>
              <button
                className="icon-btn"
                onClick={addLlmConfig}
                title="Add a config"
                aria-label="Add a config"
              >
                <Plus size={16} />
              </button>
              <button
                className="icon-btn"
                onClick={() => duplicateLlmConfig(selectedLlm.id)}
                title="Duplicate this config"
                aria-label="Duplicate this config"
              >
                <Copy size={16} />
              </button>
              <button
                className="icon-btn icon-btn-danger"
                onClick={() => removeLlmConfig(selectedLlm.id)}
                disabled={config.llm.configs.length <= 1}
                title={config.llm.configs.length <= 1 ? 'Keep at least one config' : 'Delete this config'}
                aria-label="Delete this config"
              >
                <Trash2 size={16} />
              </button>
            </div>
          </label>
          <label className="config-field">
            <span>Name</span>
            <input
              type="text"
              value={selectedLlm.name}
              onChange={e => patchLlm({ name: e.target.value })}
              placeholder="My config"
            />
          </label>
          <label className="config-checkbox">
            <input
              type="checkbox"
              checked={isDefaultLlm}
              disabled={isDefaultLlm}
              onChange={e => { if (e.target.checked) setDefaultLlm(selectedLlm.id); }}
            />
            <span>Default for new chats</span>
          </label>
        </section>

        {/* ---- LLM (selected config) ---- */}
        <section className="config-section">
          <h2>Backend</h2>
          <label className="config-field">
            <span>API URL</span>
            <input
              type="text"
              value={selectedLlm.api_url}
              onChange={e => patchLlm({ api_url: e.target.value })}
              placeholder="https://api.deepseek.com/v1/chat/completions"
            />
          </label>
          <label className="config-field">
            <span>API key</span>
            <div className="config-row">
              <input
                type={showApiKey ? 'text' : 'password'}
                value={selectedLlm.api_key}
                onChange={e => patchLlm({ api_key: e.target.value })}
                placeholder="sk-…"
              />
              <button
                className="icon-btn"
                onClick={() => setShowApiKey(s => !s)}
                title={showApiKey ? 'Hide key' : 'Show key'}
                aria-label={showApiKey ? 'Hide key' : 'Show key'}
              >
                {showApiKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
            </div>
          </label>
          <label className="config-field">
            <span>Model</span>
            <input
              type="text"
              value={selectedLlm.model}
              onChange={e => patchLlm({ model: e.target.value })}
              placeholder="deepseek-chat"
            />
          </label>
          <div className="config-row config-row-split">
            <label className="config-field">
              <span>Temperature</span>
              <input
                type="number"
                min={0} max={2} step={0.1}
                value={selectedLlm.temperature}
                onChange={e => patchLlm({ temperature: Number(e.target.value) })}
              />
            </label>
            <label className="config-field">
              <span>Request timeout (s)</span>
              <input
                type="number"
                min={1} step={1}
                value={selectedLlm.request_timeout_seconds}
                onChange={e => patchLlm({ request_timeout_seconds: Math.max(1, Math.floor(Number(e.target.value) || 30)) })}
              />
            </label>
            <label className="config-field">
              <span>Thinking</span>
              <select
                value={selectedLlm.thinking}
                onChange={e => patchLlm({ thinking: e.target.value })}
              >
                <option value="unset">Unset</option>
                <option value="enabled">Enabled</option>
                <option value="disabled">Disabled</option>
              </select>
            </label>
          </div>
          <div className="config-row config-row-split">
            <label className="config-checkbox">
              <input
                type="checkbox"
                checked={selectedLlm.send_system_prompt_as_user}
                onChange={e => patchLlm({ send_system_prompt_as_user: e.target.checked })}
              />
              <span>Send system prompt as user</span>
            </label>
            <label className="config-checkbox">
              <input
                type="checkbox"
                checked={selectedLlm.supports_vision}
                onChange={e => patchLlm({ supports_vision: e.target.checked })}
              />
              <span>Vision-capable backend</span>
            </label>
            <label className="config-field config-field-narrow">
              <span>Max tool rounds (0 = uncapped)</span>
              <input
                type="number"
                min={0} step={1}
                value={selectedLlm.max_tool_call_rounds}
                onChange={e => patchLlm({ max_tool_call_rounds: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
              />
            </label>
          </div>
          {selectedLlm.supports_vision && (
            <div className="config-row config-row-split">
              <label className="config-field config-field-narrow">
                <span>Max images sent (per origin, 0 = none)</span>
                <input
                  type="number"
                  min={0} step={1}
                  value={selectedLlm.vision_max_images}
                  onChange={e => patchLlm({ vision_max_images: Math.max(0, Math.floor(Number(e.target.value) || 0)) })}
                />
                <span className="config-hint">
                  How many of the most-recent attached/tool images are sent to the model — separately
                  for your attachments and tool screenshots. Older ones are dropped to save context.
                </span>
              </label>
            </div>
          )}
        </section>
        </>}

        {tab === 'general' && <>
        {/* ---- Workspace ---- */}
        <section className="config-section">
          <h2>Workspace</h2>
          <span className="config-hint">
            Folders the assistant can read and write under. The first entry doubles as the
            cwd for relative paths and the default working directory for Bash / PowerShell.
          </span>
          <ul className="config-list">
            {config.workspace.allowedRoots.map((root, i) => (
              <li key={i} className="config-list-row">
                <input
                  type="text"
                  value={root}
                  onChange={e => {
                    const next = config.workspace.allowedRoots.slice();
                    next[i] = e.target.value;
                    patchWorkspace({ allowedRoots: next });
                  }}
                  placeholder="C:/Path/To/Workspace"
                />
                <button
                  className="icon-btn"
                  onClick={() => pickWorkspaceDir(i)}
                  title="Browse for folder"
                  aria-label="Browse for folder"
                >
                  <FolderOpen size={16} />
                </button>
                <button
                  className="icon-btn icon-btn-danger"
                  onClick={() => removeWorkspaceRoot(i)}
                  title="Remove this root"
                  aria-label="Remove this root"
                >
                  <Trash2 size={16} />
                </button>
              </li>
            ))}
            {config.workspace.allowedRoots.length === 0 && (
              <li className="config-empty">No allowed roots configured. Add one to enable file tools.</li>
            )}
          </ul>
          <button className="config-add" onClick={addWorkspaceRoot}>
            <Plus size={14} /> Add folder…
          </button>
        </section>

        {/* ---- Permissions ---- */}
        <section className="config-section">
          <h2>Permissions</h2>
          <label className="config-checkbox">
            <input
              type="checkbox"
              checked={config.workspace.fullAccess}
              onChange={e => patchWorkspace({ fullAccess: e.target.checked })}
            />
            <span>Full permission mode</span>
          </label>
          <span className="config-hint">
            When on, the assistant never asks for approval and its file tools can read and write
            anywhere on disk — not just the workspace folders above. Leave off unless you trust the
            model with unrestricted access.
          </span>
        </section>

        </>}

        {tab === 'tts' && <>
        {/* ---- Voice (TTS) ---- */}
        <section className="config-section">
          <h2>Voice</h2>
          <span className="config-hint">
            Which engine speaks the assistant's replies. Applied on save — switching to Pocket TTS
            for the first time reloads its local model, so the next reply may lag a moment.
          </span>
          <label className="config-field">
            <span>Provider</span>
            <select
              value={config.tts.provider}
              onChange={e => patchTts({ provider: e.target.value })}
            >
              <option value="pocket">Pocket TTS (local)</option>
              <option value="elevenlabs">ElevenLabs (HTTP API)</option>
            </select>
          </label>
        </section>

        {config.tts.provider === 'pocket' && (
          <section className="config-section">
            <h2>Pocket TTS</h2>
            <span className="config-hint">
              Both settings rebuild the local model on save, so the next spoken reply may lag a
              moment while the engine reloads.
            </span>
            <label className="config-field">
              <span>Performance ⇄ quality ({config.tts.pocket.lsdSteps} flow step{config.tts.pocket.lsdSteps === 1 ? '' : 's'})</span>
              <input
                type="range"
                min={1} max={4} step={1}
                value={config.tts.pocket.lsdSteps}
                onChange={e => patchPocket({ lsdSteps: Number(e.target.value) })}
              />
              <div className="config-slider-labels">
                <span>Faster</span>
                <span>Cleaner voice</span>
              </div>
              <span className="config-hint">
                How many refinement steps shape each audio frame. More steps sound cleaner and more
                stable (especially sentence starts) but cost CPU per sentence; fewer steps keep
                synthesis snappy on weaker machines.
              </span>
            </label>
            <label className="config-field">
              <span>Quantization</span>
              <select
                value={config.tts.pocket.precision}
                onChange={e => patchPocket({ precision: e.target.value })}
              >
                <option value="fp32">Full precision (fp32) — best quality</option>
                <option value="int8">Quantized (int8) — faster, lighter</option>
              </select>
              <span className="config-hint">
                Switching for the first time downloads that variant's model files
                (a few hundred MB) before the voice comes back.
              </span>
            </label>
          </section>
        )}

        {config.tts.provider === 'elevenlabs' && (
          <section className="config-section">
            <h2>ElevenLabs</h2>
              <label className="config-field">
                <span>Base URL</span>
                <input
                  type="text"
                  value={config.tts.elevenlabs.baseUrl}
                  onChange={e => patchEleven({ baseUrl: e.target.value })}
                  placeholder="http://127.0.0.1:8000/v1/text-to-speech"
                />
                <span className="config-hint">
                  The voice id is appended per-character — the request goes to
                  <code>{' {baseUrl}/{voiceId}/stream'}</code>.
                </span>
              </label>
              <label className="config-field">
                <span>API key</span>
                <div className="config-row">
                  <input
                    type={showElevenKey ? 'text' : 'password'}
                    value={config.tts.elevenlabs.apiKey}
                    onChange={e => patchEleven({ apiKey: e.target.value })}
                    placeholder="(optional — sent as xi-api-key)"
                  />
                  <button
                    className="icon-btn"
                    onClick={() => setShowElevenKey(s => !s)}
                    title={showElevenKey ? 'Hide key' : 'Show key'}
                    aria-label={showElevenKey ? 'Hide key' : 'Show key'}
                  >
                    {showElevenKey ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
              </label>
              <label className="config-field">
                <span>Model</span>
                <input
                  type="text"
                  value={config.tts.elevenlabs.model}
                  onChange={e => patchEleven({ model: e.target.value })}
                  placeholder="eleven_multilingual_v2"
                />
              </label>
              <div className="config-row config-row-split">
                <label className="config-field">
                  <span>Stability</span>
                  <input
                    type="number"
                    min={0} max={1} step={0.05}
                    value={config.tts.elevenlabs.stability}
                    onChange={e => patchEleven({ stability: Number(e.target.value) })}
                  />
                </label>
                <label className="config-field">
                  <span>Similarity boost</span>
                  <input
                    type="number"
                    min={0} max={1} step={0.05}
                    value={config.tts.elevenlabs.similarityBoost}
                    onChange={e => patchEleven({ similarityBoost: Number(e.target.value) })}
                  />
                </label>
                <label className="config-field">
                  <span>Speed</span>
                  <input
                    type="number"
                    min={0.25} max={4} step={0.05}
                    value={config.tts.elevenlabs.speed}
                    onChange={e => patchEleven({ speed: Number(e.target.value) })}
                  />
                </label>
              </div>
              <div className="config-row config-row-split">
                <label className="config-checkbox">
                  <input
                    type="checkbox"
                    checked={config.tts.elevenlabs.useSpeakerBoost}
                    onChange={e => patchEleven({ useSpeakerBoost: e.target.checked })}
                  />
                  <span>Use speaker boost</span>
                </label>
                <label className="config-field config-field-narrow">
                  <span>Request timeout (s)</span>
                  <input
                    type="number"
                    min={1} step={1}
                    value={config.tts.elevenlabs.requestTimeoutSeconds}
                    onChange={e => patchEleven({ requestTimeoutSeconds: Math.max(1, Math.floor(Number(e.target.value) || 30)) })}
                  />
                </label>
              </div>
          </section>
        )}
        </>}
      </div>
    </div>
  );
}
