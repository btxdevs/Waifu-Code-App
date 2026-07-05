import { Check, Download, LoaderCircle } from 'lucide-react';

interface DownloadProgress {
  completed: number;
  total: number;
  percent: number; // 0..1
  label: string;
}

interface Props {
  /** Per-subsystem readiness driven by Chat.Loading / Chat.Ready from the backend. */
  status: { model: boolean; tts: boolean; stt: boolean };
  /** First-run model-download progress, or null when nothing is downloading. */
  download?: DownloadProgress | null;
}

const ROWS: { key: 'model' | 'tts' | 'stt'; label: string }[] = [
  { key: 'model', label: 'Loading character model' },
  { key: 'tts', label: 'Preparing voice (TTS)' },
  { key: 'stt', label: 'Preparing microphone (STT)' },
];

function formatBytes(n: number): string {
  if (!n || n <= 0) return '0 MB';
  const mb = n / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${Math.round(mb)} MB`;
}

/** Full-window overlay shown while a chat is starting/resuming, blocking interaction until the
 *  Unity model + TTS + STT have all loaded. Modeled on ConnectionOverlay; clears itself when the
 *  backend pushes Chat.Ready. On first run it also shows a progress bar for the (large) STT/TTS
 *  model downloads so the wait isn't an indefinite spinner. */
export function ChatLoadingOverlay({ status, download }: Props) {
  // A bar with no known total yet reads as indeterminate.
  const hasTotal = !!download && download.total > 0;
  const pct = download ? Math.min(100, Math.max(0, Math.round(download.percent * 100))) : 0;

  return (
    <div className="connection-overlay" role="alertdialog" aria-live="assertive">
      <div className="connection-overlay-card">
        <div className="connection-overlay-icon">
          <LoaderCircle size={28} className="spin" />
        </div>
        <div className="connection-overlay-title">Preparing chat…</div>
        <div className="connection-overlay-text">
          Getting everything ready. This will clear automatically once the character and voice are loaded.
        </div>
        <ul className="chat-loading-stages">
          {ROWS.map(({ key, label }) => (
            <li key={key} className={`chat-loading-stage ${status[key] ? 'done' : 'pending'}`}>
              {status[key] ? <Check size={14} /> : <LoaderCircle size={14} className="spin" />}
              <span>{label}</span>
            </li>
          ))}
        </ul>

        {download && (
          <div className="chat-loading-download">
            <div className="chat-loading-download-head">
              <Download size={14} />
              <span>Downloading models (first run)</span>
              {hasTotal && <span className="chat-loading-download-pct">{pct}%</span>}
            </div>
            <div className={`chat-loading-progress ${hasTotal ? '' : 'indeterminate'}`}>
              <div className="chat-loading-progress-fill" style={hasTotal ? { width: `${pct}%` } : undefined} />
            </div>
            {hasTotal && (
              <div className="chat-loading-download-bytes">
                {formatBytes(download.completed)} / {formatBytes(download.total)}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
