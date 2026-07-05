import { useState } from 'react';
import { Maximize2, Minimize2, Minus, Pin, PinOff, Settings, X } from 'lucide-react';
import logoUrl from '../assets/waifu_code_logo.png';

/** The app name — when the titlebar shows this, it renders the logo image instead of text.
 *  Report/question windows pass their own title and keep the text. */
const APP_NAME = 'Waifu Code';

interface Props {
  /** Text shown on the left (also the drag handle). Defaults to the app name. */
  title?: string;
  /** When provided, a settings gear is shown (chat window only). */
  onOpenSettings?: () => void;
  /** Grey out the settings gear (e.g. while disconnected from Unity). */
  settingsDisabled?: boolean;
  /** When provided, the always-on-top (pin) toggle is shown. Controlled by the
   *  parent so the state stays shared across the window's titlebars (chat window). */
  alwaysOnTop?: boolean;
  onToggleAlwaysOnTop?: () => void;
  /** Show the minimize button. Default true. */
  showMinimize?: boolean;
  /** Show the maximize/restore toggle (report windows). Default false. */
  showMaximize?: boolean;
}

/** Custom titlebar for the frameless windows (app.py creates them with frameless=True).
 *  It's the window's real top bar: title on the left (the drag handle), controls on the right.
 *  Which controls appear depends on the window — the chat window gets settings + minimize +
 *  close; report windows get minimize + maximize + close; dialogs get close only. The drag
 *  region is the `.pywebview-drag-region` element (pywebview moves the window when it's dragged;
 *  easy_drag is off so the rest of the surface stays interactive). Buttons sit OUTSIDE the drag
 *  region so their clicks aren't swallowed as a drag. */
export function Titlebar({ title = APP_NAME, onOpenSettings, settingsDisabled = false, alwaysOnTop, onToggleAlwaysOnTop, showMinimize = true, showMaximize = false }: Props) {
  const [maximized, setMaximized] = useState(false);

  const toggleMaximize = () => {
    if (maximized) {
      window.app.restoreWindow();
      setMaximized(false);
    } else {
      window.app.maximizeWindow();
      setMaximized(true);
    }
  };

  return (
    <div className="titlebar">
      <div className="titlebar-drag pywebview-drag-region">
        {title === APP_NAME
          ? <img className="titlebar-logo" src={logoUrl} alt="Waifu Code" />
          : <span className="titlebar-title" title={title}>{title}</span>}
      </div>
      <div className="titlebar-controls">
        {onOpenSettings && (
          <button
            className="titlebar-btn"
            onClick={onOpenSettings}
            disabled={settingsDisabled}
            title={settingsDisabled ? 'Settings unavailable while disconnected' : 'Settings'}
            aria-label="Settings"
          >
            <Settings size={15} />
          </button>
        )}
        {onToggleAlwaysOnTop && (
          <button
            className={alwaysOnTop ? 'titlebar-btn active' : 'titlebar-btn'}
            onClick={onToggleAlwaysOnTop}
            title={alwaysOnTop ? 'Always on top: on' : 'Always on top: off'}
            aria-label="Toggle always on top"
            aria-pressed={alwaysOnTop}
          >
            {alwaysOnTop ? <Pin size={15} /> : <PinOff size={15} />}
          </button>
        )}
        {showMinimize && (
          <button
            className="titlebar-btn"
            onClick={() => window.app.minimizeWindow()}
            title="Minimize"
            aria-label="Minimize"
          >
            <Minus size={15} />
          </button>
        )}
        {showMaximize && (
          <button
            className="titlebar-btn"
            onClick={toggleMaximize}
            title={maximized ? 'Restore' : 'Maximize'}
            aria-label={maximized ? 'Restore' : 'Maximize'}
          >
            {maximized ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
          </button>
        )}
        <button
          className="titlebar-btn close"
          onClick={() => window.app.closeWindow()}
          title="Close"
          aria-label="Close"
        >
          <X size={15} />
        </button>
      </div>
    </div>
  );
}
