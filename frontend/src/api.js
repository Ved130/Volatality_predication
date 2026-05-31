import axios from 'axios'

const BASE = 'http://localhost:8000'

export const getTickers = () =>
  axios.get(`${BASE}/tickers`).then(r => r.data)

export const forecastGarch = (ticker, horizon = 5) =>
  axios.post(`${BASE}/forecast/garch`, { ticker, horizon }).then(r => r.data)

export const forecastTft = (ticker, horizon = 5) =>
  axios.post(`${BASE}/forecast/tft`, { ticker, horizon }).then(r => r.data)

export const sendChat = (ticker, message, history = []) =>
  axios.post(`${BASE}/chat`, { ticker, message, history }).then(r => r.data)