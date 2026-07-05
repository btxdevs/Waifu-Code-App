import { useMemo, useState } from 'react';
import MarkdownIt from 'markdown-it';
import type { ShowReportPayload } from '../payloads/task';

interface Props {
  payload: ShowReportPayload;
  onClose: () => void;
}

// Single shared markdown-it instance — cheap to reuse across renders.
const md = new MarkdownIt({
  html: false,        // do not let report markdown inject raw HTML
  linkify: true,
  breaks: false,
  typographer: true,
});

export function ReportModal({ payload, onClose }: Props) {
  // useMemo so we don't re-parse markdown on every parent rerender.
  const html = useMemo(() => md.render(payload.markdown ?? ''), [payload.markdown]);
  const [copied, setCopied] = useState(false);

  const copySource = async () => {
    try {
      await navigator.clipboard.writeText(payload.markdown ?? '');
      setCopied(true);
      // Revert the label after a moment so the button is reusable.
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard can be unavailable (e.g. denied permission); ignore silently.
    }
  };

  return (
    <div className="card report-card">
      {/* Title already shown in the window titlebar — don't repeat it here. */}
      <div
        className="report-body"
        // markdown-it output is sanitized by `html: false` above; safe to inject.
        dangerouslySetInnerHTML={{ __html: html }}
      />
      <div className="actions">
        <button className="ghost" onClick={copySource}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
        <button onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
