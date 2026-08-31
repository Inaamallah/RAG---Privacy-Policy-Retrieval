// The only module that talks to the backend.
//
// Every failing response carries `{"detail": "<message>"}` -- the FastAPI
// validation handler flattens its field errors into that same shape -- so one
// helper can turn any non-2xx into an Error whose message is already the
// sentence worth showing the reader.

/** Reads the server's message off a failed response. */
async function detailOf(response) {
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
  } catch {
    // A proxy or a crashed worker can answer with HTML; fall through.
  }
  return `The server answered ${response.status}.`
}

async function request(path, options) {
  let response
  try {
    response = await fetch(path, options)
  } catch {
    // fetch only rejects when the request never got an answer.
    throw new Error('Could not reach the API. Is `uv run rag-api` running?')
  }
  if (!response.ok) throw new Error(await detailOf(response))
  return response.json()
}

/**
 * Asks whether the store can answer, and what about.
 *
 * @returns {Promise<object>} The health payload: ready, document, model,
 *   chunks, embedder_ready, the top_k bounds, and detail when not ready.
 */
export function fetchHealth() {
  return request('/api/health')
}

/**
 * Asks a question of the pinned document.
 *
 * @param {string} question The user's question.
 * @param {number} topK How many excerpts to ground the answer on.
 * @returns {Promise<{answer: string, chunks: Array}>}
 */
export function askQuestion(question, topK) {
  return request('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, top_k: topK }),
  })
}
