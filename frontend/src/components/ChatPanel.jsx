import { useState } from 'react'
import { sendChat } from '../api'

export default function ChatPanel({ ticker }) {
  const [messages, setMessages] = useState([])
  const [input,    setInput]    = useState('')
  const [loading,  setLoading]  = useState(false)

  async function handleSend() {
    if (!input.trim() || !ticker) return

    const userMsg = { role: 'user', content: input }
    const history = messages.map(m => ({ role: m.role, content: m.content }))

    setMessages(prev => [...prev, { role: 'user', content: input }])
    setInput('')
    setLoading(true)

    try {
      const res = await sendChat(ticker, input, history)
      setMessages(prev => [...prev, { role: 'assistant', content: res.response }])
    } catch (err) {
      setMessages(prev => [...prev, { role: 'assistant', content: 'Error — is the backend running?' }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-full">
      {/* Message history */}
      <div className="flex-1 overflow-y-auto space-y-3 mb-4 pr-1">
        {messages.length === 0 && (
          <p className="text-gray-500 text-sm">
            Ask anything about {ticker || 'the selected stock'}...
          </p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-lg px-4 py-3 text-sm max-w-[85%] ${
              m.role === 'user'
                ? 'bg-blue-600 text-white ml-auto'
                : 'bg-gray-700 text-gray-100'
            }`}
          >
            {m.content}
          </div>
        ))}
        {loading && (
          <div className="bg-gray-700 text-gray-400 rounded-lg px-4 py-3 text-sm w-16">
            ...
          </div>
        )}
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          placeholder="Ask about volatility..."
          className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-3 py-2 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
        />
        <button
          onClick={handleSend}
          disabled={loading || !ticker}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
        >
          Send
        </button>
      </div>
    </div>
  )
}