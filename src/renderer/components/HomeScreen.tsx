import { useEffect, useMemo, useState } from 'react';
import { Download, FolderOpen, MessageSquare, MoreVertical, Pencil, Plus, Search, Trash2, Upload, User, X } from 'lucide-react';
import type { ChatSaveMetadata } from '../payloads/chat';
import type { CharacterRecordWire } from '../payloads/character';

interface HomeScreenProps {
  characters: CharacterRecordWire[];
  saves: ChatSaveMetadata[];
  activeSlot: string;
  /** True once a session exists in Python — enables "Resume current chat". */
  chatReady: boolean;
  error: { code: string; message: string } | null;
  onDismissError: () => void;
  onStartChat: (characterId: string) => void;
  onEditCharacter: (record: CharacterRecordWire) => void;
  onDeleteCharacter: (record: CharacterRecordWire, e?: React.MouseEvent) => void;
  onResume: (slot: string) => void;
  onDelete: (slot: string, e: React.MouseEvent) => void;
  onCreateCharacter: () => void;
  /** Import a .wcc character bundle (opens the OS file picker). */
  onImportCharacter: () => void;
  /** Export a character as a .wcc bundle (opens the OS save dialog). */
  onExportCharacter: (record: CharacterRecordWire) => void;
  /** Present only when a live session exists — jumps back into it without reloading. */
  onResumeCurrent?: () => void;
  formatTime: (iso?: string) => string;
}

/** The landing page shown when the app opens. Lists the available characters (each
 *  starts a fresh chat) and the saved chats (each resumes). A chat is only opened once the
 *  user picks one of these — there's no auto-open, and an empty store just shows the
 *  "create your first character" prompt. */
/** Compact, human-readable label for a chat's work folder(s): the leaf folder name of the
 *  first root, with "+N" when several. Empty roots → "Default workspace". */
function workFolderLabel(roots?: string[]): string {
  if (!roots || roots.length === 0) return 'Default workspace';
  const leaf = (p: string) => p.replace(/[\\/]+$/, '').split(/[\\/]/).pop() || p;
  const first = leaf(roots[0]);
  return roots.length > 1 ? `${first} +${roots.length - 1}` : first;
}

export function HomeScreen({
  characters, saves, activeSlot, chatReady, error, onDismissError,
  onStartChat, onEditCharacter, onDeleteCharacter, onResume, onDelete,
  onCreateCharacter, onImportCharacter, onExportCharacter, onResumeCurrent, formatTime,
}: HomeScreenProps) {
  const [charSearch, setCharSearch] = useState('');
  const [search, setSearch] = useState('');
  // Character id whose ⋮ menu is open (one at a time). Any click outside closes it —
  // the toggler stops propagation so its own click doesn't immediately re-close it.
  const [menuOpenId, setMenuOpenId] = useState<string | null>(null);
  useEffect(() => {
    if (!menuOpenId) return;
    const close = () => setMenuOpenId(null);
    document.addEventListener('click', close);
    document.addEventListener('contextmenu', close);
    return () => {
      document.removeEventListener('click', close);
      document.removeEventListener('contextmenu', close);
    };
  }, [menuOpenId]);

  // "Drop to import" overlay while a file is dragged over the window. Purely visual —
  // the actual import happens via the Python-side drop handler (App.FilesDropped). A
  // depth counter pairs dragenter/dragleave, which fire for every child element crossed.
  const [dragging, setDragging] = useState(false);
  useEffect(() => {
    let depth = 0;
    const isFileDrag = (e: DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
    const enter = (e: DragEvent) => { if (!isFileDrag(e)) return; depth++; setDragging(true); };
    const leave = (e: DragEvent) => { if (!isFileDrag(e)) return; depth = Math.max(0, depth - 1); if (depth === 0) setDragging(false); };
    const end = () => { depth = 0; setDragging(false); };
    document.addEventListener('dragenter', enter);
    document.addEventListener('dragleave', leave);
    // Capture phase: pywebview's injected body drop handler (the import path) calls
    // stopPropagation, so a bubble-phase document listener never sees the drop — the
    // capture phase runs first, before the body handler can swallow it.
    document.addEventListener('drop', end, true);
    window.addEventListener('dragend', end);
    return () => {
      document.removeEventListener('dragenter', enter);
      document.removeEventListener('dragleave', leave);
      document.removeEventListener('drop', end, true);
      window.removeEventListener('dragend', end);
    };
  }, []);

  // Filter characters by display name / name.
  const filteredCharacters = useMemo(() => {
    const q = charSearch.trim().toLowerCase();
    if (!q) return characters;
    return characters.filter(c =>
      (c.displayName || '').toLowerCase().includes(q) || (c.name || '').toLowerCase().includes(q));
  }, [characters, charSearch]);

  // Filter the saves by character name OR any work-folder path (full paths matched, not just
  // the displayed leaf, so a parent-dir search still finds it).
  const filteredSaves = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return saves;
    return saves.filter(s => {
      const inName = (s.characterName || '').toLowerCase().includes(q);
      const inFolder = (s.workspaceRoots || []).some(r => r.toLowerCase().includes(q));
      return inName || inFolder;
    });
  }, [saves, search]);

  return (
    <div className="home-screen">
      {dragging && (
        <div className="home-drop-overlay">
          <div className="home-drop-box">
            <Download size={28} />
            <span>Drop a character bundle (.wcc) to import it</span>
          </div>
        </div>
      )}
      {error && (
        <div className="error-banner">
          <span className="error-text">{error.message || 'Something went wrong.'}</span>
          <button className="error-dismiss" onClick={onDismissError} title="Dismiss" aria-label="Dismiss error">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="home-body">
        <section className="home-section home-section-characters">
          <div className="home-section-head">
            <h2>Characters</h2>
            <div className="home-head-actions">
              <button className="new-chat-btn" onClick={onImportCharacter}
                      title="Import a character bundle (.wcc)">
                <FolderOpen size={14} /> Import
              </button>
              <button className="new-chat-btn" onClick={onCreateCharacter} title="Create a new character">
                <Plus size={14} /> New Character
              </button>
            </div>
          </div>
          {characters.length > 0 && (
            <div className="home-search">
              <Search size={13} className="home-search-icon" />
              <input
                type="text"
                className="home-search-input"
                placeholder="Search characters…"
                value={charSearch}
                onChange={(e) => setCharSearch(e.target.value)}
              />
              {charSearch && (
                <button className="home-search-clear" onClick={() => setCharSearch('')} title="Clear search" aria-label="Clear search">
                  <X size={13} />
                </button>
              )}
            </div>
          )}
          {characters.length === 0 ? (
            <div className="home-empty">
              No characters yet. Create one — pick a .vrm or .kkm model and it's ready to chat.
            </div>
          ) : filteredCharacters.length === 0 ? (
            <div className="home-empty">No characters match “{charSearch}”.</div>
          ) : (
            <div className="home-char-row">
              {filteredCharacters.map(char => {
                const label = char.displayName || char.name;
                return (
                  <div
                    key={char.id}
                    className="home-char-card"
                    onContextMenu={(e) => {
                      // Right-clicking anywhere on the card opens the same ⋮ options menu.
                      e.preventDefault();
                      e.stopPropagation();
                      setMenuOpenId(id => (id === char.id ? null : char.id));
                    }}
                  >
                    <button
                      className="home-char-menu-btn"
                      onClick={(e) => { e.stopPropagation(); setMenuOpenId(id => (id === char.id ? null : char.id)); }}
                      title="Character options"
                      aria-label={`Options for ${label}`}
                      aria-expanded={menuOpenId === char.id}
                    >
                      <MoreVertical size={14} />
                    </button>
                    {menuOpenId === char.id && (
                      <div className="home-char-menu">
                        <button onClick={() => { setMenuOpenId(null); onEditCharacter(char); }}>
                          <Pencil size={12} /> Edit
                        </button>
                        <button onClick={() => { setMenuOpenId(null); onExportCharacter(char); }}>
                          <Upload size={12} /> Export
                        </button>
                        <button className="danger" onClick={(e) => { setMenuOpenId(null); onDeleteCharacter(char, e); }}>
                          <Trash2 size={12} /> Delete
                        </button>
                      </div>
                    )}
                    <span className="home-char-avatar">
                      {char.profileImage ? <img src={char.profileImage} alt="" /> : <User size={22} />}
                    </span>
                    <span className="home-char-name" title={label}>{label}</span>
                    <button
                      className="home-char-start"
                      onClick={() => onStartChat(char.id)}
                      title={`Start a new chat with ${label}`}
                    >
                      <MessageSquare size={12} /> New chat
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        <section className="home-section home-section-saves">
          <div className="home-section-head">
            <h2>Recent chats</h2>
          </div>
          {saves.length > 0 && (
            <div className="home-search">
              <Search size={13} className="home-search-icon" />
              <input
                type="text"
                className="home-search-input"
                placeholder="Search name or work folder…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <button className="home-search-clear" onClick={() => setSearch('')} title="Clear search" aria-label="Clear search">
                  <X size={13} />
                </button>
              )}
            </div>
          )}
          {saves.length === 0 ? (
            <div className="home-empty">No saved chats yet.</div>
          ) : filteredSaves.length === 0 ? (
            <div className="home-empty">No chats match “{search}”.</div>
          ) : (
            <div className="home-saves">
              {filteredSaves.map(save => {
                const isActive = chatReady && activeSlot === save.slot;
                const folderLabel = workFolderLabel(save.workspaceRoots);
                const folderTitle = (save.workspaceRoots && save.workspaceRoots.length)
                  ? save.workspaceRoots.join('\n')
                  : 'No per-chat work folder — uses the default workspace';
                return (
                  <div
                    key={save.slot}
                    className={`chat-item ${isActive ? 'active' : ''}`}
                    // Clicking the chat that's already the live session just jumps back into it
                    // (no reload); any other chat is loaded fresh.
                    onClick={() => (isActive && onResumeCurrent ? onResumeCurrent() : onResume(save.slot))}
                  >
                    <div className="chat-item-info">
                      <div className="chat-item-header">
                        <span className="chat-item-name">{save.characterName}</span>
                        <span className="chat-item-time">{formatTime(save.savedAtUtc)}</span>
                      </div>
                      <div className="chat-item-snippet">{save.lastMessageText || 'New conversation'}</div>
                      <div className="chat-item-folder" title={folderTitle}>
                        <FolderOpen size={11} />
                        <span className="chat-item-folder-text">{folderLabel}</span>
                      </div>
                    </div>
                    <button className="delete-chat-btn" onClick={(e) => onDelete(save.slot, e)} title="Delete this chat">
                      <Trash2 size={13} />
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
