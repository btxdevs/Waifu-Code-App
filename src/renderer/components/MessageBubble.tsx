import { useLayoutEffect, useRef, useState } from 'react';
import {
  Cat, CheckCircle2, Circle, CircleDot, FileText, Globe, Hand, ListChecks, MessageSquare,
  Pencil, Search, Shirt, Terminal, Undo2, Wrench, X,
} from 'lucide-react';
import type { ChatHistoryEntry, ChatTodoEntry } from '../payloads/chat';
import { AttachmentChips } from './AttachmentChips';

interface MessageBubbleProps {
  entry: ChatHistoryEntry;
  arrayIndex: number;
  onOpenReport: (id: string) => void;
  onRollback: (turnIndex: number) => void;
  /** `removedAttachments` = indexes into entry.attachments the user unattached during the
   *  edit (empty when only the text changed); `turnIndex` locates the turn snapshot. */
  onEdit: (arrayIndex: number, historyIndex: number, text: string,
           removedAttachments: number[], turnIndex: number) => void;
  /** Editing is disabled while a turn is streaming. */
  editable?: boolean;
  live?: boolean;
}

export function MessageBubble({ entry, arrayIndex, onOpenReport, onRollback, onEdit, editable, live }: MessageBubbleProps) {
  const [editing, setEditing] = useState(false);
  const [editDraft, setEditDraft] = useState('');
  // Attachment indexes unattached during the current edit session; applied on save.
  const [editRemoved, setEditRemoved] = useState<Set<number>>(new Set());
  const editRef = useRef<HTMLTextAreaElement | null>(null);

  // Focus + size the textarea when entering edit mode.
  useLayoutEffect(() => {
    if (!editing) return;
    const el = editRef.current;
    if (!el) return;
    el.focus();
    el.setSelectionRange(el.value.length, el.value.length);
    el.style.height = 'auto';
    el.style.height = `${el.scrollHeight}px`;
  }, [editing]);

  const beginEdit = () => {
    setEditDraft(entry.text);
    setEditRemoved(new Set());
    setEditing(true);
  };
  const commitEdit = () => {
    const next = editDraft.trim();
    setEditing(false);
    // Only fire if something actually changed and we have a backing history index. A blanked
    // textarea keeps the old text (an edit can't erase the message) but removals still apply.
    const textChanged = !!next && next !== entry.text;
    const removed = Array.from(editRemoved).sort((a, b) => a - b);
    if ((textChanged || removed.length > 0) && entry.historyIndex != null && entry.historyIndex >= 0) {
      onEdit(arrayIndex, entry.historyIndex, textChanged ? next : entry.text, removed, entry.turnIndex);
    }
  };
  const cancelEdit = () => setEditing(false);
  // tool_activity rows render as compact event lines between speech bubbles, not as
  // chat bubbles. Widgets attach inline to the row that produced them:
  //  * single report → the whole row becomes the "view report" button (the row's text
  //    already carries the title, so a separate chip would just duplicate it)
  //  * multiple reports (rare) → the row stays static and chips render inline after it
  //  * TodoWrite → todo list renders below since it's substantive content, not a button
  // A user action (e.g. a touch/caress) is something the USER did to the avatar — not a tool the
  // assistant ran. Its own category, rendered on the right like the user's messages and kept
  // entirely separate from the assistant's left-aligned tool rows (so the two never get confused
  // as more of each kind are added).
  if (entry.role === 'user_action') {
    return (
      <div className="user-action">
        <div className="user-action-line">
          <span className="user-action-icon">{iconForUserAction(entry.toolName)}</span>
          <span className="user-action-text">{entry.text}</span>
          {entry.canRollback && (
            <button
              type="button"
              className="rollback-btn user-action-cancel"
              title="Cancel this action and the character's response."
              onClick={() => onRollback(entry.turnIndex)}
            >
              <X size={11} /> cancel
            </button>
          )}
        </div>
      </div>
    );
  }

  if (entry.role === 'tool_activity') {
    const onlyReport = entry.reports && entry.reports.length === 1 ? entry.reports[0] : null;
    // Touch rows are rewindable — canceling removes the touch interaction + the AI's reply.
    const cancelBtn = entry.canRollback ? (
      <button
        type="button"
        className="rollback-btn tool-activity-cancel"
        title="Cancel this touch and the character's response."
        onClick={() => onRollback(entry.turnIndex)}
      >
        <X size={11} /> cancel
      </button>
    ) : null;
    const innerLine = (
      <>
        <span className="tool-activity-icon">{iconForTool(entry.toolName)}</span>
        <span className="tool-activity-text">{entry.text}</span>
      </>
    );

    if (onlyReport) {
      return (
        <div className="tool-activity">
          <button
            type="button"
            className="tool-activity-line tool-activity-clickable"
            title="Open report"
            onClick={() => onOpenReport(onlyReport.id)}
          >
            {innerLine}
          </button>
          {entry.todos && entry.todos.length > 0 && <TodoChecklist items={entry.todos} />}
        </div>
      );
    }

    return (
      <div className="tool-activity">
        <div className="tool-activity-line">
          {innerLine}
          {entry.reports && entry.reports.length > 1 && entry.reports.map(r => (
            <button
              key={r.id}
              className="report-chip"
              title={r.id}
              onClick={() => onOpenReport(r.id)}
            >
              <FileText size={11} /> {r.title || 'report'}
            </button>
          ))}
          {cancelBtn}
        </div>
        {entry.todos && entry.todos.length > 0 && <TodoChecklist items={entry.todos} />}
      </div>
    );
  }

  const isUser = entry.role === 'user';
  const canEdit = editable && !live && entry.historyIndex != null && entry.historyIndex >= 0;
  return (
    <div className={`bubble ${isUser ? 'user' : 'assistant'} ${live ? 'live' : ''} ${editing ? 'editing' : ''}`}>
      <div className="bubble-header">
        <span className="bubble-speaker">{entry.speaker || (isUser ? 'You' : 'Assistant')}</span>
        <div className="bubble-actions">
          {!editing && canEdit && (
            <button
              className="edit-btn"
              title="Edit this message."
              onClick={beginEdit}
            >
              <Pencil size={11} /> edit
            </button>
          )}
          {isUser && entry.canRollback && (
            <button
              className="rollback-btn"
              title="Rewind the conversation to right before this message."
              onClick={() => onRollback(entry.turnIndex)}
            >
              <Undo2 size={11} /> rewind
            </button>
          )}
        </div>
      </div>
      {editing ? (
        <div className="bubble-edit">
          <textarea
            ref={editRef}
            className="bubble-edit-input"
            value={editDraft}
            onChange={e => {
              setEditDraft(e.target.value);
              e.target.style.height = 'auto';
              e.target.style.height = `${e.target.scrollHeight}px`;
            }}
            onKeyDown={e => {
              // Enter saves, Shift+Enter newline, Escape cancels.
              if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); commitEdit(); }
              else if (e.key === 'Escape') { e.preventDefault(); cancelEdit(); }
            }}
          />
          {entry.attachments && entry.attachments.length > 0 && (
            /* Unattach while editing: chips get remove buttons; removals apply on Save
             * (Cancel forgets them). Indexes are into the ORIGINAL attachment list. */
            <AttachmentChips
              attachments={entry.attachments.filter((_, i) => !editRemoved.has(i))}
              onRemove={path => {
                const i = entry.attachments!.findIndex(a => a.path === path);
                if (i >= 0) setEditRemoved(prev => new Set(prev).add(i));
              }}
            />
          )}
          <div className="bubble-edit-actions">
            <button className="bubble-edit-cancel" onClick={cancelEdit}>Cancel</button>
            <button className="bubble-edit-save" onClick={commitEdit}>Save</button>
          </div>
        </div>
      ) : (
        <div className="bubble-text">{entry.text}</div>
      )}
      {!editing && entry.attachments && entry.attachments.length > 0 && (
        <AttachmentChips attachments={entry.attachments} expandable />
      )}
      {entry.reports && entry.reports.length > 0 && (
        <div className="bubble-reports">
          {entry.reports.map(r => (
            <button
              key={r.id}
              className="report-chip"
              title={r.id}
              onClick={() => onOpenReport(r.id)}
            >
              <FileText size={12} /> {r.title || 'report'}
            </button>
          ))}
        </div>
      )}
      {entry.todos && entry.todos.length > 0 && <TodoChecklist items={entry.todos} />}
    </div>
  );
}

/** Pick a lucide icon for the inline tool_activity row. Falls back to a generic
 *  wrench when the tool name doesn't match anything in particular. */
function iconForTool(toolName: string | undefined) {
  switch (toolName) {
    case 'ReportWrite':                          return <FileText size={13} />;
    case 'Read':
    case 'Open':                                 return <FileText size={13} />;
    case 'Write':
    case 'Edit':                                 return <Pencil size={13} />;
    case 'Glob':
    case 'Grep':                                 return <Search size={13} />;
    case 'Bash':
    case 'PowerShell':                           return <Terminal size={13} />;
    case 'WebFetch':
    case 'WebPageRead':
    case 'WebPageOutline':                       return <Globe size={13} />;
    case 'WebSearch':                            return <Search size={13} />;
    case 'TodoWrite':                            return <ListChecks size={13} />;
    case 'ChangeOutfit':                         return <Shirt size={13} />;
    case 'AskUserQuestion':                      return <MessageSquare size={13} />;
    case 'UwUAgent':
    case 'CheckUwUHelpers':
    case 'DismissUwUHelper':                     return <Cat size={13} />;
    default:                                     return <Wrench size={13} />;
  }
}

/** Pick a lucide icon for a `user_action` row (something the user did to the avatar — kept
 *  separate from the tool icon set above). Falls back to a hand for any interaction. */
function iconForUserAction(actionName: string | undefined) {
  switch (actionName) {
    case 'Touch':                                return <Hand size={13} />;
    case 'ChangeOutfit':                         return <Shirt size={13} />;
    case 'UwUAgent':                             return <Cat size={13} />;  // background helper reported back
    default:                                     return <Hand size={13} />;
  }
}

interface TodoChecklistProps {
  items: ChatTodoEntry[];
}

/** Read-only mirror of the TodoWrite list as of the turn that produced it. Status icons:
 *  pending = empty ring, in_progress = filled dot, completed = checkmark + strikethrough. */
function TodoChecklist({ items }: TodoChecklistProps) {
  return (
    <ul className="bubble-todos">
      {items.map((t, i) => {
        const status = t.status ?? 'pending';
        const label = status === 'in_progress' ? (t.activeForm || t.content) : (t.content || t.activeForm);
        const icon =
          status === 'completed' ? <CheckCircle2 size={13} /> :
          status === 'in_progress' ? <CircleDot size={13} /> :
          <Circle size={13} />;
        return (
          <li key={i} className={`todo-item todo-${status}`}>
            <span className="todo-icon">{icon}</span>
            <span className="todo-text">{label}</span>
          </li>
        );
      })}
    </ul>
  );
}
