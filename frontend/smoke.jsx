// Throwaway render check: does the page mount, and do its branches render?
import { renderToStaticMarkup } from 'react-dom/server'

import App from './src/App.jsx'
import Message from './src/components/Message.jsx'
import Sidebar from './src/components/Sidebar.jsx'

const READY = {
  ready: true,
  document: 'policy_removed_removed.pdf',
  model: 'openai/gpt-oss-120b',
  chunks: 6,
  embedder_ready: true,
  default_top_k: 5,
  min_top_k: 1,
  max_top_k: 10,
  detail: null,
}

const ANSWER = {
  role: 'assistant',
  content: 'The budget caps per-visitor risk [policy_removed_removed.pdf, p.1,2].\n\n```\nmin {γ, ξ t}\n```',
  chunks: [
    {
      id: 'policy_removed_removed.pdf:abc123:0',
      text: 'raw  chunk   text\nwith spacing',
      score: 0.6040477156639099,
      metadata: {
        source: 'policy_removed_removed.pdf',
        pages: '1,2',
        headings: '1. Introduction',
        chunk_index: 1,
      },
    },
  ],
}

let failures = 0

function check(label, html, expected, forbidden = []) {
  const missing = expected.filter((needle) => !html.includes(needle))
  const present = forbidden.filter((needle) => html.includes(needle))
  if (missing.length || present.length) {
    failures += 1
    console.log(`FAIL ${label}`)
    if (missing.length) console.log(`     missing: ${JSON.stringify(missing)}`)
    if (present.length) console.log(`     should not contain: ${JSON.stringify(present)}`)
    return
  }
  console.log(`ok   ${label}`)
}

// First paint, before /api/health has answered.
const initial = renderToStaticMarkup(<App />)
check('initial paint', initial, [
  'Document Q&amp;A',
  'Connecting to the API...',
  'Checking the vector store...',
  'Waiting for the vector store...',
  'Fixed by design',
  '<textarea',
  'disabled',
])

// An answer with its excerpts.
const answered = renderToStaticMarkup(<Message message={ANSWER} />)
check(
  'answer renders',
  answered,
  [
    'The budget caps per-visitor risk [policy_removed_removed.pdf, p.1,2].',
    '<pre><code>min {γ, ξ t}',       // fenced block survives verbatim
    'Sources (1 excerpt)',
    'policy_removed_removed.pdf — p.1,2',
    '1. Introduction',
    'match 0.604',
    'raw  chunk   text',              // preformatted, spacing intact
  ],
  ['<math', 'katex'],                 // equations must never be typeset
)

// An error turn.
const failed = renderToStaticMarkup(
  <Message message={{ role: 'assistant', content: '⚠️ GROQ_API_KEY is not set.', error: true }} />,
)
check('error turn renders', failed, ['message-error', 'GROQ_API_KEY is not set.'])

// The sidebar, with the server's bounds applied to the slider.
const sidebar = renderToStaticMarkup(
  <Sidebar health={READY} topK={5} onTopK={() => {}} onClear={() => {}} canClear />,
)
check('sidebar renders', sidebar, [
  'policy_removed_removed.pdf',
  'openai/gpt-oss-120b',
  'Excerpts per answer',
  'max="10"',
  'min="1"',
  '6 stored chunks',
])

console.log(failures ? `\n${failures} check(s) failed` : '\nall render checks passed')
process.exit(failures ? 1 : 0)
