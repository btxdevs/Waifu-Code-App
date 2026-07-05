import { useMemo } from 'react';
import {
  Envelope,
  TYPE_ASK_QUESTION,
  TYPE_REQUEST_PERMISSION,
  TYPE_SHOW_REPORT,
  TYPE_PERMISSION_DECISION,
  TYPE_QUESTION_ANSWER,
  TYPE_REPORT_CLOSED,
} from './protocol';
import type { AskQuestionPayload, RequestPermissionPayload, ShowReportPayload } from './payloads/task';
import { ReportModal } from './components/ReportModal';
import { QuestionModal } from './components/QuestionModal';
import { PermissionModal } from './components/PermissionModal';
import { ChatView } from './components/ChatView';
import { Titlebar } from './components/Titlebar';

interface HashState {
  kind: string | null;
  envelope: Envelope | null;
}

function readHash(): HashState {
  const raw = window.location.hash.startsWith('#') ? window.location.hash.slice(1) : window.location.hash;
  const out: HashState = { kind: null, envelope: null };
  for (const pair of raw.split('&')) {
    const eq = pair.indexOf('=');
    if (eq < 0) continue;
    const key = pair.slice(0, eq);
    const val = pair.slice(eq + 1);
    if (key === 'kind') out.kind = decodeURIComponent(val);
    else if (key === 'envelope') {
      try { out.envelope = JSON.parse(decodeURIComponent(val)) as Envelope; }
      catch (e) { console.error('[App] Bad envelope in URL hash:', e); }
    }
  }
  return out;
}

export function App() {
  const hash = useMemo(readHash, []);

  if (hash.kind === 'chat') {
    // ChatView draws its own frameless titlebar (it owns the settings state the titlebar
    // toggles), so the shell lives inside it rather than here.
    return <ChatView />;
  }

  const env = hash.envelope;
  if (!env) {
    return <div className="empty-state">No task envelope in URL hash — this window was opened directly.</div>;
  }

  switch (env.type) {
    case TYPE_SHOW_REPORT: {
      const payload = env.payload as ShowReportPayload;
      const title = payload.title ? `Report — ${payload.title}` : 'Report';
      // Reports are reading material — allow minimize + maximize + close.
      return (
        <div className="app-shell">
          <Titlebar title={title} showMinimize showMaximize />
          <ReportModal payload={payload} onClose={() => window.app.reply(TYPE_REPORT_CLOSED, {})} />
        </div>
      );
    }
    case TYPE_ASK_QUESTION: {
      const payload = env.payload as AskQuestionPayload;
      // Dialogs demand an answer — close only (no minimize/maximize). Closing replies with
      // the safe default (cancelled), synthesized by the window's on_closed handler.
      return (
        <div className="app-shell">
          <Titlebar title="Question" showMinimize={false} />
          <QuestionModal payload={payload} onAnswer={ans => window.app.reply(TYPE_QUESTION_ANSWER, ans)} />
        </div>
      );
    }
    case TYPE_REQUEST_PERMISSION: {
      const payload = env.payload as RequestPermissionPayload;
      // Full descriptor lives in the titlebar (incl. the tier — it matters for an approval),
      // so the modal doesn't repeat it as a heading.
      const title = `Approve ${payload.tier} tool: ${payload.toolName || 'tool'}`;
      return (
        <div className="app-shell">
          <Titlebar title={title} showMinimize={false} />
          <PermissionModal payload={payload} onDecision={d => window.app.reply(TYPE_PERMISSION_DECISION, d)} />
        </div>
      );
    }
    default:
      return <div className="empty-state">Unknown task type: {env.type}</div>;
  }
}
