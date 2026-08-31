import { useRef, useState } from 'react'

/**
 * The question box, pinned to the bottom of the conversation.
 *
 * Enter sends and Shift+Enter starts a line, which is what a chat box is
 * expected to do; the textarea grows with the question up to a few lines.
 */
export default function Composer({ onSend, disabled, placeholder }) {
  const [text, setText] = useState('')
  const box = useRef(null)

  function resize(element) {
    if (!element) return
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 160)}px`
  }

  function submit(event) {
    event.preventDefault()
    const question = text.trim()
    if (!question || disabled) return
    setText('')
    resize(box.current)
    onSend(question)
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) submit(event)
  }

  return (
    <form className="composer" onSubmit={submit}>
      <textarea
        ref={box}
        rows={1}
        value={text}
        placeholder={placeholder}
        aria-label="Your question"
        disabled={disabled}
        onKeyDown={onKeyDown}
        onChange={(event) => {
          setText(event.target.value)
          resize(event.target)
        }}
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        Ask
      </button>
    </form>
  )
}
