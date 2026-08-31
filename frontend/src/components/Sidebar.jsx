/**
 * The settings panel: what is being answered from, and the one knob.
 *
 * The document is display-only on purpose. It is pinned server side by a
 * Chroma metadata filter, so there is nothing here to change it with -- the
 * page cannot widen what it is allowed to read.
 */
export default function Sidebar({ health, topK, onTopK, onClear, canClear }) {
  const bounds = {
    min: health?.min_top_k ?? 1,
    max: health?.max_top_k ?? 10,
  }

  return (
    <aside className="sidebar">
      <h2>Settings</h2>

      <section className="field">
        <h3>Document</h3>
        <code>{health?.document ?? '—'}</code>
        <p className="hint">
          Fixed by design — this app has no upload, so the answers always come from this one PDF.
        </p>
      </section>

      <section className="field">
        <h3>Model</h3>
        <code>{health?.model ?? '—'}</code>
      </section>

      <section className="field">
        <label htmlFor="top-k">
          Excerpts per answer <span className="value">{topK}</span>
        </label>
        <input
          id="top-k"
          type="range"
          min={bounds.min}
          max={bounds.max}
          step={1}
          value={topK}
          onChange={(event) => onTopK(Number(event.target.value))}
        />
        <p className="hint">How many retrieved chunks are given to the model as context.</p>
      </section>

      <button type="button" className="secondary" onClick={onClear} disabled={!canClear}>
        Clear chat
      </button>

      {health?.chunks ? (
        <p className="hint stored">
          {health.chunks} stored chunk{health.chunks === 1 ? '' : 's'}
        </p>
      ) : null}
    </aside>
  )
}
