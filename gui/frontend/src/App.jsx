import { useState, useEffect, useRef } from 'react'
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
        <rect x="2.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="8.5" y="2.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="8.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
        <rect x="2.5" y="8.5" width="4" height="4" stroke="currentColor" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: '/training',
    label: 'Training',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <path d="M2.5 12.5L7.5 4.5L12.5 12.5" stroke="currentColor" strokeLinejoin="round" />
        <path d="M4 10H11" stroke="currentColor" strokeLinejoin="round" />
      </svg>
    ),
  },
  {
    to: '/inference',
    label: 'Inference',
    icon: (
      <svg width="15" height="15" viewBox="0 0 15 15" fill="none">
        <circle cx="6.5" cy="6.5" r="4" stroke="currentColor" />
        <path d="M12.5 12.5L9.5 9.5" stroke="currentColor" strokeLinecap="round" />
      </svg>
    ),
  },
]

export default function App() {
  const [connected, setConnected] = useState(false)
  const [dropdownOpen, setDropdownOpen] = useState(false)
  const [lastChecked, setLastChecked] = useState('')
  const [isChecking, setIsChecking] = useState(false)
  const dropdownRef = useRef(null)

  const check = (showChecking = false) => {
    if (showChecking) setIsChecking(true)
    return checkHealth()
      .then(() => {
        setConnected(true)
        setLastChecked(new Date().toLocaleTimeString())
      })
      .catch(() => {
        setConnected(false)
        setLastChecked(new Date().toLocaleTimeString())
      })
      .finally(() => {
        if (showChecking) setIsChecking(false)
      })
  }

  useEffect(() => {
    check(true)
    const interval = setInterval(() => check(false), 4000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    function handleClickOutside(event) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setDropdownOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  return (
    <div className="app">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sb-brand">
          <img src={suparcoLogo} alt="SUPARCO" className="sb-logo" />
          <h1>Super-Resolution</h1>
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
        <div className="sb-footer" ref={dropdownRef}>
          {dropdownOpen && (
            <div className="sb-status-dropdown" role="menu">
              <div className="sb-status-detail-item">
                <span>Endpoint</span>
                <span className="val">/api/health</span>
              </div>
              <div className="sb-status-detail-item">
                <span>Status</span>
                <span className="val" style={{ color: connected ? 'var(--ok)' : 'var(--bad)', fontWeight: 600 }}>
                  {connected ? 'Healthy' : 'Unhealthy'}
                </span>
              </div>
              <div className="sb-status-detail-item">
                <span>Last Checked</span>
                <span className="val">{lastChecked || 'Checking...'}</span>
              </div>
              <div className="sb-status-divider" />
              <button
                type="button"
                className="sb-status-refresh-btn"
                onClick={() => check(true)}
                disabled={isChecking}
              >
                <svg
                  className={isChecking ? 'spin' : ''}
                  width="13"
                  height="13"
                  viewBox="0 0 16 16"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M2.5 2v4h4" />
                  <path d="M13.5 14v-4h-4" />
                  <path d="M13.5 6A7 7 0 0 0 4.5 3.5L2.5 6" />
                  <path d="M2.5 10a7 7 0 0 0 9 2.5l2-2.5" />
                </svg>
                {isChecking ? 'Checking...' : 'Check Status'}
              </button>
            </div>
          )}
          <button
            type="button"
            className={`sb-status-trigger ${dropdownOpen ? 'open' : ''}`}
            onClick={() => setDropdownOpen(!dropdownOpen)}
            aria-haspopup="true"
            aria-expanded={dropdownOpen}
          >
            <div className="sb-status-info" style={{ color: 'var(--ink)' }}>
              API Status
            </div>
            <svg
              className="chevron"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="6 9 12 15 18 9" />
            </svg>
          </button>
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
