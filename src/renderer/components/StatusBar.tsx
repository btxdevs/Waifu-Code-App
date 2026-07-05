import { useEffect, useState } from 'react';
import { AlertTriangle, Cat, CheckCircle2, ChevronDown, ChevronUp, LoaderCircle, Mic, Terminal, X } from 'lucide-react';
import type { ChatBgTaskEntry } from '../payloads/chat';

interface StatusBarProps {
  voiceRecording: boolean;
  voiceBusy: boolean;
  voiceBusyReason: 'loading' | 'transcribing' | null;
  /** Background tasks (UwU helpers + background commands). Running tasks show a
   *  live chip with elapsed time and a dismiss ✕; finished-but-not-yet-announced
   *  ones show a "waiting to report" chip. Announced/dismissed tasks are hidden. */
  bgTasks: ChatBgTaskEntry[];
  onDismissTask: (taskId: string) => void;
}

function formatElapsed(startedAt: number | undefined): string {
  if (!startedAt) return '';
  const s = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

/** Strip at the bottom of the chat window. Left side always tells you what the voice
 * subsystem is doing (stays visible with a "Ready" string even when idle so the layout
 * doesn't jump). Right side tracks background tasks while any exist — a one-chip
 * summary by default, expandable (click) into the full per-task list with dismiss
 * buttons, collapsible again via the chevron. */
export function StatusBar({ voiceRecording, voiceBusy, voiceBusyReason, bgTasks, onDismissTask }: StatusBarProps) {
  const visible = bgTasks.filter(t =>
    t.status === 'running' || ((t.status === 'completed' || t.status === 'failed') && !t.announced));
  const running = visible.filter(t => t.status === 'running');
  const waiting = visible.filter(t => t.status !== 'running');
  const anyRunning = running.length > 0;

  const [expanded, setExpanded] = useState(false);
  // Snap shut when the last task leaves so the next batch starts collapsed.
  useEffect(() => {
    if (visible.length === 0) setExpanded(false);
  }, [visible.length]);

  // Tick once a second while something is running so the elapsed labels move.
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!anyRunning) return;
    const id = setInterval(() => setTick(t => t + 1), 1000);
    return () => clearInterval(id);
  }, [anyRunning]);

  let icon: React.ReactNode;
  let text: string;
  let cls = 'status-idle';
  if (voiceBusy && voiceBusyReason === 'loading') {
    icon = <LoaderCircle size={12} className="spin" />;
    text = 'Loading speech model… (first run downloads ~660 MB)';
    cls = 'status-loading';
  } else if (voiceBusy && voiceBusyReason === 'transcribing') {
    icon = <LoaderCircle size={12} className="spin" />;
    text = 'Transcribing audio…';
    cls = 'status-busy';
  } else if (voiceRecording) {
    icon = <Mic size={12} />;
    text = 'Recording — click mic again to stop';
    cls = 'status-recording';
  } else {
    icon = null;
    text = 'Ready';
  }

  // Collapsed summary: a single task shows its own label; several fold into counts.
  const summaryBits: string[] = [];
  if (running.length === 1) {
    summaryBits.push(`${running[0].label} · ${formatElapsed(running[0].startedAt)}`);
  } else if (running.length > 1) {
    const oldest = Math.min(...running.map(t => t.startedAt ?? Date.now()));
    summaryBits.push(`${running.length} running · ${formatElapsed(oldest)}`);
  }
  if (waiting.length > 0) summaryBits.push(`${waiting.length} waiting to report`);
  const anyFailed = waiting.some(t => t.status === 'failed');
  const titleAll = visible.map(t => `${t.kind}: ${t.label}`).join('\n');

  return (
    <div className={`status-bar ${cls}`}>
      {icon}
      <span>{text}</span>
      {visible.length > 0 && !expanded && (
        <span className="status-bar-tasks">
          <button
            className="bg-chip bg-chip-summary"
            title={`${titleAll}\n(click to expand)`}
            onClick={() => setExpanded(true)}
          >
            {anyRunning
              ? <LoaderCircle size={11} className="spin" />
              : anyFailed ? <AlertTriangle size={11} /> : <CheckCircle2 size={11} />}
            <span className="bg-chip-label">{summaryBits.join(' · ')}</span>
            <ChevronUp size={11} />
          </button>
        </span>
      )}
      {visible.length > 0 && expanded && (
        <span className="status-bar-tasks expanded">
          {/* The chip list wraps on its own; the collapse toggle sits OUTSIDE the
              wrap flow in its own column, so chips can never land on top of it. */}
          <span className="bg-chip-list">
            {visible.map(t => (
              <span key={t.id} className={`bg-chip bg-chip-${t.status}`} title={`${t.kind}: ${t.label}`}>
                {t.status === 'running'
                  ? <LoaderCircle size={11} className="spin" />
                  : t.status === 'failed'
                    ? <AlertTriangle size={11} />
                    : <CheckCircle2 size={11} />}
                {t.kind === 'shell' ? <Terminal size={11} /> : <Cat size={11} />}
                <span className="bg-chip-label">{t.label}</span>
                {t.status === 'running'
                  ? <span className="bg-chip-time">{formatElapsed(t.startedAt)}</span>
                  : <span className="bg-chip-time">waiting to report</span>}
                {t.status === 'running' && (
                  <button
                    className="bg-chip-x"
                    title="Cancel this task"
                    onClick={() => onDismissTask(t.id)}
                  >
                    <X size={10} />
                  </button>
                )}
              </span>
            ))}
          </span>
          <button className="bg-tasks-toggle" title="Collapse" onClick={() => setExpanded(false)}>
            <ChevronDown size={12} />
          </button>
        </span>
      )}
    </div>
  );
}
