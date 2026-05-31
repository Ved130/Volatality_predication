export default function TickerSelector({ tickers, selected, onChange }) {
  return (
    <div className="flex flex-col gap-1">
      <label className="text-sm font-medium text-gray-400">Ticker</label>
      <select
        value={selected}
        onChange={e => onChange(e.target.value)}
        className="bg-gray-800 text-white border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
      >
        {tickers.map(t => (
          <option key={t.ticker} value={t.ticker}>
            {t.ticker} — {t.company}
          </option>
        ))}
      </select>
    </div>
  )
}