import { useCallback, useEffect, useRef, useState } from 'react'

import { askQuestion, fetchHealth } from './api.js'
import Composer from './components/Composer.jsx'
import Message from './components/Message.jsx'
import Sidebar from './components/Sidebar.jsx'

// Kept until the server says otherwise, so the slider has a sane position on
// the very first paint; /api/health then reports the real default and bounds.
const FALLBACK_TOP_K = 5

// How often to re-ask an unhappy server. The store becomes answerable when an
// ingest finishes in another terminal, and the embedding model finishes
// loading a few seconds after startup -- both are worth waiting through
// rather than making the reader reload the page.
const POLL_MS = 3000

export default function App() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)
  const [checking, setChecking] = useState(true)
  const [messages, setMessages] = useState([])
  const [topK, setTopK] = useState(FALLBACK_TOP_K)
  const [pending, setPending] = useState(false)

  // Only the first health response may move the slider; after that the
  // reader owns it, and a poll must not drag it back to the default.
  const sliderTouched = useRef(false)
  const bottom = useRef(null)

  const checkHealth = useCallback(async () => {
    try {
      const data = await fetchHealth()
      setHealth(data)
      setHealthError(null)
      if (!sliderTouched.current) {
        setTopK(data.default_top_k ?? FALLBACK_TOP_K)
        sliderTouched.current = true
      }
    } catch (error) {
      setHealth(null)
      setHealthError(error.message)
    } finally {
      setChecking(false)
    }
  }, [])

  useEffect(() => {
    checkHealth()
  }, [checkHealth])

  // Poll only while something is still expected to change.
  const settled = health?.ready && health?.embedder_ready
  useEffect(() => {
    if (settled) return undefined
    const timer = setInterval(checkHealth, POLL_MS)
    return () => clearInterval(timer)
  }, [settled, checkHealth])

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, pending])

  async function send(question) {
    setMessages((current) => [...current, { role: 'user', content: question }])
    setPending(true)
    try {
      const { answer, chunks } = await askQuestion(question, topK)
      setMessages((current) => [...current, { role: 'assistant', content: answer, chunks }])
    } catch (error) {
      setMessages((current) => [
        ...current,
        { role: 'assistant', content: `⚠️ ${error.message}`, error: true },
      ])
      // The same failure usually means the server's state moved; re-read it
      // so the banner and the sidebar stop describing a world that is gone.
      checkHealth()
    } finally {
      setPending(false)
    }
  }

  const blocked = !checking && (healthError || (health && !health.ready))
  const blockedDetail = healthError ?? health?.detail ?? 'The API is not ready.'

  return (
    <div className="layout">
      <Sidebar
        health={health}
        topK={topK}
        onTopK={setTopK}
        onClear={() => setMessages([])}
        canClear={messages.length > 0}
      />

      <main className="chat">
        <header className="chat-header">
          <h1>📄 Document Q&amp;A</h1>
          <p className="caption">
            {health
              ? `Grounded on ${health.document} · ${health.model} via Groq`
              : 'Connecting to the API...'}
          </p>
        </header>

        <div className="transcript">
          {checking ? <p className="notice">Checking the vector store...</p> : null}

          {blocked ? (
            <div className="notice error" role="alert">
              <p>{blockedDetail}</p>
              <button type="button" className="secondary" onClick={checkHealth}>
                Try again
              </button>
            </div>
          ) : null}

          {health?.ready && !health.embedder_ready ? (
            <p className="notice">Loading the embedding model...</p>
          ) : null}

          {health?.ready && messages.length === 0 ? (
            <p className="notice greeting">
              Ask me anything about <strong>{health.document}</strong>. I answer only from that
              document, and I cite the page each claim came from.
            </p>
          ) : null}

          {messages.map((message, index) => (
            <Message key={index} message={message} />
          ))}

          {pending ? (
            <p className="notice pendingnotice" aria-live="polite">
              Searching the document<span className="dots" />
            </p>
          ) : null}

          <div ref={bottom} />
        </div>

        <Composer
          onSend={send}
          disabled={pending || !health?.ready}
          placeholder={
            health?.ready ? `Ask about ${health.document}...` : 'Waiting for the vector store...'
          }
        />
      </main>
    </div>
  )
}
