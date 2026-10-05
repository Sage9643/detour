import { useState } from "react";

import type { CatalogEntry } from "./api";

const MAX = 6;

interface Props {
  catalog: CatalogEntry[];
  selected: string[];
  onChange: (labels: string[]) => void;
  disabled?: boolean;
}

export function InterestPicker({ catalog, selected, onChange, disabled }: Props) {
  const [draft, setDraft] = useState("");
  const chosen = new Set(selected.map((s) => s.toLowerCase()));
  const full = selected.length >= MAX;

  function add(label: string) {
    const clean = label.trim().replace(/\s+/g, " ");
    if (!clean || chosen.has(clean.toLowerCase()) || full) return;
    onChange([...selected, clean]);
  }

  function remove(label: string) {
    onChange(selected.filter((s) => s !== label));
  }

  return (
    <div className="picker">
      <div className="picker-selected" aria-live="polite">
        {selected.length === 0 && <span className="picker-empty">Pick up to {MAX} interests.</span>}
        {selected.map((label) => (
          <button
            key={label}
            type="button"
            className="chip chip-on"
            onClick={() => remove(label)}
            disabled={disabled}
            aria-label={`Remove ${label}`}
          >
            {label}
            <span aria-hidden="true" className="chip-x">
              ×
            </span>
          </button>
        ))}
      </div>

      <form
        className="picker-custom"
        onSubmit={(e) => {
          e.preventDefault();
          add(draft);
          setDraft("");
        }}
      >
        <label htmlFor="custom-interest" className="visually-hidden">
          Add your own interest
        </label>
        <input
          id="custom-interest"
          value={draft}
          maxLength={60}
          placeholder="Add your own, e.g. bioinformatics"
          onChange={(e) => setDraft(e.target.value)}
          disabled={disabled || full}
        />
        <button type="submit" className="btn-quiet" disabled={disabled || full || !draft.trim()}>
          Add
        </button>
      </form>

      <div className="picker-catalog">
        {catalog
          .filter((c) => !chosen.has(c.label.toLowerCase()))
          .map((c) => (
            <button
              key={c.label}
              type="button"
              className="chip"
              onClick={() => add(c.label)}
              disabled={disabled || full}
              title={`GitHub topics: ${c.topics.slice(0, 4).join(", ")}`}
            >
              {c.label}
            </button>
          ))}
      </div>
    </div>
  );
}
