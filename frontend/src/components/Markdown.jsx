import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

/**
 * Renders an answer as markdown.
 *
 * There is deliberately no math plugin here, and adding one would be a bug.
 * Equations reach the model flattened out of the PDF's text layer -- a
 * fraction bar and a subscript both arrive as a space -- so the system prompt
 * has the model quote them verbatim in a fenced code block rather than as
 * `$...$`. A typesetter would silently eat the braces and spacing that
 * quoting exists to preserve, and would render `min {γ, ...}` as something the
 * document never said.
 */
export default function Markdown({ children }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{children}</ReactMarkdown>
    </div>
  )
}
