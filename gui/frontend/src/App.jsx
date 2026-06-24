import { useState, useEffect } from 'react'
import { NavLink, Routes, Route, Navigate } from 'react-router-dom'
import Training from './pages/Training'
import Inference from './pages/Inference'
import Preprocessing from './pages/Preprocessing'
import { checkHealth } from './api/client'
import suparcoLogo from '../assets/suparco logo.jpg'

const NAV_ITEMS = [
  {
    to: '/preprocessing',
    label: 'Preprocessing',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <rect x="2.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round"/>
        <rect x="8.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round"/>
        <rect x="8.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round"/>
        <rect x="2.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    to: '/training',
    label: 'Training',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <path d="M2.5 12.5L7.5 4.5L12.5 12.5" stroke="currentColor" strokeLinejoin="round"/>
        <path d="M4 10H11" stroke="currentColor" strokeLinejoin="round"/>
      </svg>
    ),
  },
  {
    to: '/inference',
    label: 'Inference',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <circle cx="6.5" cy="6.5" r="4" stroke="currentColor"/>
        <path d="M12.5 12.5L9.5 9.5" stroke="currentColor" strokeLinecap="round"/>
      </svg>
    ),
  },
]

export default function App() {
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const check = () => {
      checkHealth()
        .then(() => setConnected(true))
        .catch(() => setConnected(false))
    }
    check()
    const interval = setInterval(check, 4000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sb-brand">
          <img src={suparcoLogo} alt="SUPARCO" className="sb-logo" />
          <h1>SUPARCO SR</h1>
        </div>
        <nav className="sb-nav">
          <div className="sb-sec">Workflows</div>
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `sb-item ${isActive ? 'active' : ''}`}
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div style={{ marginTop: 'auto', padding: '18px', borderTop: '1px solid var(--line)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: '13.5px', color: connected ? 'var(--ok)' : 'var(--bad)', fontWeight: 500 }}>
            <div className={`pulse-dot ${connected ? '' : 'disconnected'}`} />
            {connected ? 'Connected' : 'Disconnected'}
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="main">
        <Routes>
          <Route path="/" element={<Navigate to="/training" replace />} />
          <Route path="/training" element={<Training />} />
          <Route path="/inference" element={<Inference />} />
          <Route path="/preprocessing" element={<Preprocessing />} />
        </Routes>
      </main>
    </div>
  )
}
