import { useEffect, useState } from 'react';
import { Shirt, X } from 'lucide-react';
import type { KkCoordinate } from '../payloads/character';

interface Props {
  /** The character's KK outfit list (from its KK_Coordinates.json). */
  coordinates: KkCoordinate[];
  /** Stable coordinate id of the outfit currently worn (-1 = unknown). */
  currentIndex: number;
  /** The character's .kkm path — outfit screenshots are read from inside it. */
  modelPath: string;
  onCancel: () => void;
  onConfirm: (outfitIndex: number) => void;
}

/** Modal to change the avatar's outfit as a USER action (the character reacts to it).
 *  Shows one card per outfit with its screenshot from the .kkm (right-click flips between
 *  the front and back shots), name and description. Confirm is enabled once a different
 *  outfit than the worn one is selected. Only opened for KK models with 2+ outfits. */
export function OutfitPickerDialog({ coordinates, currentIndex, modelPath, onCancel, onConfirm }: Props) {
  const [selected, setSelected] = useState(currentIndex);
  return (
    <div className="dialog-overlay" onClick={onCancel}>
      <div className="dialog outfit-dialog" onClick={e => e.stopPropagation()}>
        <div className="dialog-header">
          <h2>Change outfit</h2>
          <button className="icon-btn" onClick={onCancel} title="Cancel" aria-label="Cancel">
            <X size={16} />
          </button>
        </div>
        <div className="dialog-body outfit-grid">
          {coordinates.map(c => (
            <OutfitCard
              key={c.index}
              coord={c}
              modelPath={modelPath}
              selected={selected === c.index}
              worn={c.index === currentIndex}
              onSelect={() => setSelected(c.index)}
            />
          ))}
        </div>
        <div className="dialog-footer">
          <span className="outfit-hint">Tip: right-click an outfit to see its back view.</span>
          <button className="dialog-btn" onClick={onCancel}>Cancel</button>
          <button
            className="dialog-btn primary"
            disabled={selected < 0 || selected === currentIndex}
            onClick={() => onConfirm(selected)}
          >
            Change outfit
          </button>
        </div>
      </div>
    </div>
  );
}

interface CardProps {
  coord: KkCoordinate;
  modelPath: string;
  selected: boolean;
  worn: boolean;
  onSelect: () => void;
}

/** One outfit card. Left-click selects; right-click flips the preview between the front and
 *  back screenshots (loaded from inside the .kkm; a shirt icon stands in while loading /
 *  missing). The flip resets nothing else — selection is untouched. */
function OutfitCard({ coord, modelPath, selected, worn, onSelect }: CardProps) {
  const front = coord.screenshots?.front;
  const back = coord.screenshots?.back;
  const [frontSrc, setFrontSrc] = useState<string | null>(null);
  const [backSrc, setBackSrc] = useState<string | null>(null);
  const [showBack, setShowBack] = useState(false);
  useEffect(() => {
    let cancelled = false;
    setFrontSrc(null);
    setBackSrc(null);
    setShowBack(false);
    if (front) {
      window.app.readModelScreenshot(modelPath, front)
        .then(url => { if (!cancelled) setFrontSrc(url); })
        .catch(() => {});
    }
    if (back) {
      window.app.readModelScreenshot(modelPath, back)
        .then(url => { if (!cancelled) setBackSrc(url); })
        .catch(() => {});
    }
    return () => { cancelled = true; };
  }, [modelPath, front, back]);
  const src = (showBack && backSrc) || frontSrc || backSrc;
  return (
    <button
      className={`outfit-card ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      onContextMenu={e => {
        e.preventDefault();
        if (backSrc) setShowBack(b => !b);
      }}
      title={coord.description || coord.name}
    >
      <span className="outfit-card-shot">
        {src ? <img src={src} alt="" /> : <Shirt size={28} />}
      </span>
      <span className="outfit-card-name">{coord.name || `Outfit ${coord.index}`}</span>
      {worn && <span className="outfit-card-current">worn</span>}
    </button>
  );
}
