import Markdown from './Markdown.jsx'
import Sources from './Sources.jsx'

/** One turn of the conversation, with its excerpts when it has any. */
export default function Message({ message }) {
  const isUser = message.role === 'user'
  return (
    <article className={`message message-${message.role}${message.error ? ' message-error' : ''}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? '🧑' : '📄'}
      </div>
      <div className="bubble">
        <span className="sr-only">{isUser ? 'You asked' : 'Answer'}:</span>
        <Markdown>{message.content}</Markdown>
        <Sources chunks={message.chunks} />
      </div>
    </article>
  )
}
