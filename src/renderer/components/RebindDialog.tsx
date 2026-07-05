import { User, X } from 'lucide-react';
import type { CharacterRecordWire } from '../payloads/character';

interface Props {
  /** Display name of the (deleted) character the chat belonged to, for the explainer text. */
  characterName: string;
  characters: CharacterRecordWire[];
  onCancel: () => void;
  onConfirm: (characterId: string) => void;
}

/** Shown when resuming a chat whose character was deleted (e.g. a duplicate cleanup after
 *  an import): pick any existing character to re-bind the chat to — it doesn't have to be
 *  the same one. The history is kept as-is; the chosen character takes over from there. */
export function RebindDialog({ characterName, characters, onCancel, onConfirm }: Props) {
  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Character missing</h2>
          <button className="icon-btn" onClick={onCancel} title="Cancel" aria-label="Cancel">
            <X size={16} />
          </button>
        </div>
        <div className="dialog-body">
          <p className="rebind-text">
            {characterName
              ? `The character for this chat (“${characterName}”) was deleted.`
              : 'The character for this chat was deleted.'}{' '}
            Pick a character to continue the chat with — the conversation is kept as-is.
          </p>
          {characters.length === 0 ? (
            <span className="config-hint">No characters exist — create or import one first.</span>
          ) : (
            <div className="rebind-list">
              {characters.map(c => {
                const label = c.displayName || c.name;
                return (
                  <button key={c.id} className="rebind-item" onClick={() => onConfirm(c.id)}>
                    <span className="rebind-avatar">
                      {c.profileImage ? <img src={c.profileImage} alt="" /> : <User size={16} />}
                    </span>
                    <span className="rebind-name" title={label}>{label}</span>
                  </button>
                );
              })}
            </div>
          )}
        </div>
        <div className="dialog-footer">
          <button className="dialog-btn" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  );
}
