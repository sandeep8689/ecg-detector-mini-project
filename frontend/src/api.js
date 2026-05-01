/**
 * api.js — Centralised Axios instance
 *
 * LOCAL DEV:  VITE_API_URL is empty → Vite proxy forwards /api → localhost:8000
 * PRODUCTION: Set VITE_API_URL=https://your-backend.onrender.com in Vercel env vars
 */
import axios from 'axios'

const BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({ baseURL: BASE_URL })

export default api
