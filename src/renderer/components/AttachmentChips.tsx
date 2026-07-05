import { useEffect, useState } from 'react';
import { FileText, Image as ImageIcon, X } from 'lucide-react';
import type { ChatAttachmentRef } from '../payloads/chat';

/** Classify a path the same way the Python side does (IMAGE_EXTS). Used by the composer,
 *  which only has raw paths; sent bubbles get the kind from the wire. */
export function attachmentRefForPath(path: string): ChatAttachmentRef {
  const name = path.replace(/\\/g, '/').split('/').pop() || path;
  const kind = /\.(png|jpe?g|gif|webp|bmp)$/i.test(name) ? 'image' : 'file';
  return { name, path, kind };
}

interface AttachmentChipsProps {
  attachments: ChatAttachmentRef[];
  /** Composer mode: render a remove button on each chip. */
  onRemove?: (path: string) => void;
  /** Bubble mode: clicking an image chip toggles a larger inline preview below the row. */
  expandable?: boolean;
}

/** The ONE way attachments are displayed everywhere (composer + sent bubbles): a compact
 *  chip row, identical for every attachment kind. Image chips lead with a tiny thumbnail
 *  (file chips with a file icon) and, when `expandable`, click-toggle a larger preview. */
export function AttachmentChips({ attachments, onRemove, expandable }: AttachmentChipsProps) {
  const [expandedPath, setExpandedPath] = useState<string | null>(null);
  if (!attachments.length) return null;
  const expanded = expandable ? attachments.find(a => a.path === expandedPath) : undefined;
  return (
    <div className="attachment-chips">
      {attachments.map(a => {
        const clickable = expandable && a.kind === 'image';
        return (
          <span
            key={a.path}
            className={`attachment-chip ${clickable ? 'attachment-chip-clickable' : ''} ${a.path === expandedPath ? 'attachment-chip-expanded' : ''}`}
            title={a.path}
            role={clickable ? 'button' : undefined}
            onClick={clickable ? () => setExpandedPath(p => (p === a.path ? null : a.path)) : undefined}
          >
            {a.kind === 'image' ? <ChipThumb path={a.path} /> : <FileText size={12} />}
            <span className="attachment-chip-name">{a.name}</span>
            {onRemove && (
              <button
                className="attachment-chip-remove"
                onClick={e => { e.stopPropagation(); onRemove(a.path); }}
                title="Remove attachment"
                aria-label="Remove attachment"
              >
                <X size={11} />
              </button>
            )}
          </span>
        );
      })}
      {expanded && <AttachmentPreview key={expanded.path} path={expanded.path} name={expanded.name} />}
    </div>
  );
}

/** Loads an image attachment's bytes as a data URL via the bridge; null until loaded/failed. */
function useImageDataUrl(path: string): string | null {
  const [src, setSrc] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    window.app.readImageDataUrl(path)
      .then(url => { if (!cancelled) setSrc(url); })
      .catch(() => { if (!cancelled) setSrc(null); });
    return () => { cancelled = true; };
  }, [path]);
  return src;
}

/** Tiny in-chip thumbnail; falls back to the generic image icon while loading / on failure. */
function ChipThumb({ path }: { path: string }) {
  const src = useImageDataUrl(path);
  if (!src) return <ImageIcon size={12} />;
  return <img className="attachment-chip-thumb" src={src} alt="" />;
}

/** The click-to-expand larger preview under the chip row. Renders nothing if unreadable. */
function AttachmentPreview({ path, name }: { path: string; name: string }) {
  const src = useImageDataUrl(path);
  if (!src) return null;
  return <img className="attachment-preview" src={src} alt={name} title={name} />;
}
