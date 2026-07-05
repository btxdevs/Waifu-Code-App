import { useState } from 'react';
import { LoaderCircle, Play, WifiOff } from 'lucide-react';

/** Full-window overlay shown whenever the Unity backend WS is down. The app
 *  auto-reconnects with backoff (see app.py `_ws_loop`), so this clears itself
 *  the moment the connection is re-established. The "Open Player" button asks the
 *  app to (re)launch the player — handy when it was closed or crashed. */
export function ConnectionOverlay() {
  const [launching, setLaunching] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  const openPlayer = async () => {
    setLaunching(true);
    setNote(null);
    try {
      const res = await window.app.openPlayer();
      if (!res?.ok) {
        setNote(res?.error || 'Could not start the player.');
        setLaunching(false);
      } else if (res.launched === false) {
        setNote('The player is already running.');
        setLaunching(false);
      } else {
        setNote('Launching the player…');
        // Keep the button busy; the overlay clears itself once the WS reconnects.
      }
    } catch {
      setNote('Could not start the player.');
      setLaunching(false);
    }
  };

  return (
    <div className="connection-overlay" role="alertdialog" aria-live="assertive">
      <div className="connection-overlay-card">
        <div className="connection-overlay-icon">
          <WifiOff size={28} />
        </div>
        <div className="connection-overlay-title">Disconnected from Player</div>
        <div className="connection-overlay-text">
          Waiting for the player to reconnect… This will clear automatically once the
          connection is back.
        </div>
        <button className="connection-overlay-button" onClick={openPlayer} disabled={launching}>
          {launching ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}
          <span>{launching ? 'Opening…' : 'Open Player'}</span>
        </button>
        {note ? <div className="connection-overlay-note">{note}</div> : null}
        <div className="connection-overlay-progress">
          <LoaderCircle size={14} className="spin" />
          <span>Reconnecting…</span>
        </div>
      </div>
    </div>
  );
}
