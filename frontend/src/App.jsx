import { useState, useEffect } from 'react'
import { getTickers, forecastGarch, forecastTft } from './api'
import TickerSelector from './components/TickerSelector'
import ForecastChart  from './components/ForecastChart'
import ChatPanel      from './components/ChatPanel'

export default function App() {
  const [tickers,    setTickers]    = useState([])
  const [selected,   setSelected]   = useState('AAPL')
  const [horizon,    setHorizon]    = useState(5)
  const [garchData,  setGarchData]  = useState(null)
  const [tftData,    setTftData]    = useState(null)
  const [loading,    setLoading]    = useState(false)
  const [error,      setError]      = useState(null)
  const [activeTab,  setActiveTab]  = useState('forecast')

  // Load tickers on mount
  useEffect(() => {
    getTickers()
      .then(setTickers)
      .catch(() => setError('Could not load tickers — is the backend running?'))
  }, [])

  async function handleForecast() {
    if (!selected) return
    setLoading(true)
    setError(null)

    try {
      const [garch, tft] = await Promise.all([
        forecastGarch(selected, horizon),
        forecastTft(selected, horizon),
      ])
      setGarchData(garch)
      setTftData(tft)
    } catch (err) {
      setError('Forecast failed — ' + err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="border-b border-gray-700 px-6 py-4">
        <h1 className="text-xl font-bold text-white">
          Volatility Forecasting
        </h1>
        <p className="text-sm text-gray-400">
          TFT + GARCH across 50 S&P 500 stocks
        </p>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-6 space-y-6">

        {/* Controls */}
        <div className="bg-gray-800 rounded-xl p-5 flex flex-wrap gap-4 items-end">
          <div className="flex-1 min-w-48">
            <TickerSelector
              tickers={tickers}
              selected={selected}
              onChange={setSelected}
            />
          </div>

          {/* Horizon selector */}
          <div className="flex flex-col gap-1">
            <label className="text-sm font-medium text-gray-400">Horizon</label>
            <select
              value={horizon}
              onChange={e => setHorizon(Number(e.target.value))}
              className="bg-gray-700 text-white border border-gray-600 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500"
            >
              <option value={5}>5 days</option>
              <option value={10}>10 days</option>
            </select>
          </div>

          <button
            onClick={handleForecast}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? 'Forecasting...' : 'Forecast'}
          </button>
        </div>

        {/* Error banner */}
        {error && (
          <div className="bg-red-900/40 border border-red-500 rounded-lg px-4 py-3 text-red-300 text-sm">
            {error}
          </div>
        )}

        {/* Metrics cards */}
        {tftData && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Current Vol',   value: (tftData.current_vol * 100).toFixed(1) + '%' },
              { label: '30d Avg Vol',   value: (tftData.avg_vol_30d * 100).toFixed(1) + '%' },
              { label: '5d Return',     value: (tftData.return_5d  * 100).toFixed(2) + '%' },
              { label: 'Vol Regime',    value: tftData.vol_regime,
                highlight: tftData.vol_regime === 'high' },
            ].map(card => (
              <div key={card.label} className="bg-gray-800 rounded-xl p-4">
                <p className="text-xs text-gray-400 mb-1">{card.label}</p>
                <p className={`text-xl font-bold ${card.highlight ? 'text-red-400' : 'text-white'}`}>
                  {card.value}
                </p>
              </div>
            ))}
          </div>
        )}

        {/* Tabs */}
        <div className="flex gap-2 border-b border-gray-700 pb-0">
          {['forecast', 'chat'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-sm font-medium capitalize rounded-t-lg transition-colors ${
                activeTab === tab
                  ? 'bg-gray-800 text-white border border-b-0 border-gray-700'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              {tab === 'forecast' ? 'Forecast Chart' : 'AI Assistant'}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="bg-gray-800 rounded-xl rounded-tl-none p-5">
          {activeTab === 'forecast' ? (
            <div>
              <div className="flex justify-between items-center mb-4">
                <h2 className="font-semibold text-white">
                  {selected} — {horizon}-Day Volatility Forecast
                </h2>
                {tftData && (
                  <div className="flex gap-4 text-xs text-gray-400">
                    <span>TFT RMSE: 0.0602</span>
                    <span>GARCH RMSE: 0.0751</span>
                    <span>Coverage: 79.9%</span>
                  </div>
                )}
              </div>
              <ForecastChart garchData={garchData} tftData={tftData} />
            </div>
          ) : (
            <div className="h-96">
              <ChatPanel ticker={selected} />
            </div>
          )}
        </div>

      </div>
    </div>
  )
}