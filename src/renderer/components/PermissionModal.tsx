import { useEffect } from 'react';
import type { PermissionDecisionPayload, RequestPermissionPayload } from '../payloads/task';

interface Props {
  payload: RequestPermissionPayload;
  onDecision: (d: PermissionDecisionPayload) => void;
}

export function PermissionModal({ payload, onDecision }: Props) {
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') {
        e.preventDefault();
        onDecision({ allow: false, scope: 'Once' });
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onDecision]);

  // Scope label names the actual tool rather than guessing from the tier — the
  // DangerFullAccess tier is shared by Bash, Screenshot, etc., so "shell" was wrong
  // for non-shell tools. Falls back to a generic phrase if toolName is missing.
  const sessionLabel = payload.toolName
    ? `Allow ${payload.toolName} this session`
    : 'Allow this session';

  return (
    <div className="card">
      {/* Title (incl. tier + tool) is shown in the window titlebar — don't repeat it here. */}
      <div className="detail">{payload.detail}</div>
      <div className="actions">
        <button className="deny" onClick={() => onDecision({ allow: false, scope: 'Once' })}>
          Deny
        </button>
        <button className="allow" onClick={() => onDecision({ allow: true, scope: 'Once' })}>
          Allow once
        </button>
        <button className="allow-strong" onClick={() => onDecision({ allow: true, scope: 'Session' })}>
          {sessionLabel}
        </button>
      </div>
    </div>
  );
}
