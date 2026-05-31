import {
  ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

export default function ForecastChart({ garchData, tftData }) {
  if (!garchData && !tftData) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">
        Select a ticker and click Forecast
      </div>
    )
  }

  // Build chart data from TFT forecast
  const data = tftData
    ? tftData.p50.map((val, i) => ({
        day:        `Day ${i + 1}`,
        tft_p50:    parseFloat(val.toFixed(4)),
        tft_lower:  parseFloat(tftData.conf_lower[i].toFixed(4)),
        tft_upper:  parseFloat(tftData.conf_upper[i].toFixed(4)),
        garch:      garchData ? parseFloat(garchData.forecast_vol[i].toFixed(4)) : null,
      }))
    : []

  return (
    <div className="w-full h-72">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis dataKey="day" stroke="#9CA3AF" tick={{ fontSize: 12 }} />
          <YAxis stroke="#9CA3AF" tick={{ fontSize: 12 }} tickFormatter={v => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
            labelStyle={{ color: '#F9FAFB' }}
            formatter={(value, name) => [value.toFixed(4), name]}
          />
          <Legend />

          {/* TFT uncertainty band */}
          <Area
            type="monotone"
            dataKey="tft_upper"
            fill="#3B82F6"
            stroke="none"
            fillOpacity={0.15}
            name="TFT Upper"
            legendType="none"
          />
          <Area
            type="monotone"
            dataKey="tft_lower"
            fill="#1F2937"
            stroke="none"
            fillOpacity={1}
            name="TFT Lower"
            legendType="none"
          />

          {/* TFT p50 */}
          <Line
            type="monotone"
            dataKey="tft_p50"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={{ r: 4 }}
            name="TFT p50"
          />

          {/* GARCH comparison */}
          {garchData && (
            <Line
              type="monotone"
              dataKey="garch"
              stroke="#F59E0B"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={{ r: 4 }}
              name="GARCH(1,1)"
            />
          )}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}