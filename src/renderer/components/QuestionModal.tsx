import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { AskQuestionOption, AskQuestionPayload, QuestionAnswerPayload } from '../payloads/task';

interface Props {
  payload: AskQuestionPayload;
  onAnswer: (a: QuestionAnswerPayload) => void;
}

export function QuestionModal({ payload, onAnswer }: Props) {
  const [typed, setTyped] = useState('');
  const [checked, setChecked] = useState<Set<string>>(new Set());
  // Which option's preview pane is currently displayed. Kept independent from
  // `checked`: in multi-select you may tick three options but only one preview
  // is shown at a time (focused/hovered).
  const [focused, setFocused] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const options = payload.options ?? [];
  // Preview pane only renders when ≥1 option has a preview AND the question is single-select.
  // Matches the reference: "previews are only supported for single-select questions".
  const hasAnyPreview = useMemo(
    () => !payload.multiSelect && options.some(o => !!o.preview),
    [payload.multiSelect, options]
  );

  // Reset state when a new question arrives (parent reuses this component for back-to-back tasks).
  useEffect(() => {
    setTyped('');
    setChecked(new Set());
    setFocused(hasAnyPreview ? options.find(o => !!o.preview)?.label ?? null : null);
    inputRef.current?.focus();
  }, [payload, hasAnyPreview, options]);

  const submitSingle = useCallback(
    (text: string) => onAnswer({ cancelled: false, text, wasMultiSelect: false }),
    [onAnswer]
  );

  const submitMulti = useCallback(() => {
    const parts: string[] = [];
    for (const opt of options) if (checked.has(opt.label)) parts.push(opt.label);
    const trimmed = typed.trim();

    let text = '';
    if (parts.length && trimmed) text = `${parts.join(', ')} (notes: ${trimmed})`;
    else if (parts.length) text = parts.join(', ');
    else if (trimmed) text = trimmed;

    onAnswer({ cancelled: false, text, wasMultiSelect: true });
  }, [options, checked, typed, onAnswer]);

  const cancel = useCallback(
    () => onAnswer({ cancelled: true, text: '', wasMultiSelect: payload.multiSelect }),
    [onAnswer, payload.multiSelect]
  );

  function onKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter') {
      e.preventDefault();
      if (payload.multiSelect) submitMulti();
      else                     submitSingle(typed);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      cancel();
    }
  }

  function toggle(label: string) {
    setChecked(prev => {
      const next = new Set(prev);
      if (next.has(label)) next.delete(label);
      else                  next.add(label);
      return next;
    });
  }

  const focusedPreview = useMemo(() => {
    if (!hasAnyPreview || !focused) return null;
    return options.find(o => o.label === focused)?.preview ?? null;
  }, [hasAnyPreview, focused, options]);

  return (
    <div className={`card question-card ${hasAnyPreview ? 'has-preview' : ''}`}>
      {payload.header && <div className="question-header-chip">{payload.header}</div>}
      <h1>{payload.question || '(no question)'}</h1>

      <div className={`question-body ${hasAnyPreview ? 'with-preview' : ''}`}>
        <div className="question-options-col">
          {options.length > 0 && !payload.multiSelect && (
            <div className="options single rich">
              {options.map(opt => (
                <OptionButton
                  key={opt.label}
                  opt={opt}
                  focused={focused === opt.label}
                  onFocus={() => setFocused(opt.label)}
                  onSelect={() => submitSingle(opt.label)}
                />
              ))}
            </div>
          )}

          {options.length > 0 && payload.multiSelect && (
            <div className="options multi rich">
              {options.map(opt => (
                <OptionCheckbox
                  key={opt.label}
                  opt={opt}
                  checked={checked.has(opt.label)}
                  onToggle={() => toggle(opt.label)}
                />
              ))}
            </div>
          )}
        </div>

        {hasAnyPreview && (
          <div className="question-preview-col">
            {focusedPreview
              ? <pre className="question-preview">{focusedPreview}</pre>
              : <div className="question-preview empty">Hover an option to preview.</div>}
          </div>
        )}
      </div>

      {payload.allowFreeText && (
        <input
          ref={inputRef}
          type="text"
          value={typed}
          onChange={e => setTyped(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={payload.multiSelect ? 'Optional notes…' : 'Or type your own answer (Enter to submit)…'}
        />
      )}

      <div className="actions">
        <button className="ghost" onClick={cancel}>Cancel</button>
        <button
          className="allow"
          onClick={() => (payload.multiSelect ? submitMulti() : submitSingle(typed))}
        >
          {payload.multiSelect ? 'Submit selection' : 'Submit'}
        </button>
      </div>
    </div>
  );
}

interface OptionButtonProps {
  opt: AskQuestionOption;
  focused: boolean;
  onFocus: () => void;
  onSelect: () => void;
}

function OptionButton({ opt, focused, onFocus, onSelect }: OptionButtonProps) {
  return (
    <button
      className={`question-option ${focused ? 'focused' : ''}`}
      onMouseEnter={onFocus}
      onFocus={onFocus}
      onClick={onSelect}
    >
      <span className="question-option-label">{opt.label}</span>
      {opt.description && <span className="question-option-desc">{opt.description}</span>}
    </button>
  );
}

interface OptionCheckboxProps {
  opt: AskQuestionOption;
  checked: boolean;
  onToggle: () => void;
}

function OptionCheckbox({ opt, checked, onToggle }: OptionCheckboxProps) {
  return (
    <label className="question-option checkbox">
      <input type="checkbox" checked={checked} onChange={onToggle} />
      <span className="question-option-text">
        <span className="question-option-label">{opt.label}</span>
        {opt.description && <span className="question-option-desc">{opt.description}</span>}
      </span>
    </label>
  );
}
