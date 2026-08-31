/**
 * The excerpts an answer was grounded on, collapsed until asked for.
 *
 * Scores and headings are shown here because this panel is for the reader.
 * The prompt still never carries them: `format_context` builds the model's
 * context block from the text and the citation label alone.
 */
export default function Sources({ chunks }) {
  if (!chunks?.length) return null

  const count = chunks.length
  return (
    <details className="sources">
      <summary>
        Sources ({count} excerpt{count === 1 ? '' : 's'})
      </summary>
      <ol className="sources-list">
        {chunks.map((chunk) => {
          const meta = chunk.metadata ?? {}
          const where = meta.pages ? `p.${meta.pages}` : 'page unknown'
          return (
            <li key={chunk.id} className="source">
              <p className="source-head">
                <strong>
                  {meta.source || 'unknown'} — {where}
                </strong>
                {meta.headings ? <em className="source-section">{meta.headings}</em> : null}
                <span className="source-score">match {chunk.score.toFixed(3)}</span>
              </p>
              {/* Preformatted, never markdown: this is the raw stored chunk,
                  and its line breaks and spacing are the evidence. */}
              <pre className="source-text">{chunk.text}</pre>
            </li>
          )
        })}
      </ol>
    </details>
  )
}
