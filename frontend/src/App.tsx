import { useState, useEffect } from 'react'
import './App.css'

interface HealthStatus {
  status: string
  service: string
  version: string
}

function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8002'

  useEffect(() => {
    fetch(`${apiUrl}/health`)
      .then(res => res.json())
      .then(data => {
        setHealth(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [apiUrl])

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏗️ Infrastructure Intelligence Platform</h1>
        <p className="subtitle">Underground Utility Detection & BIM Integration</p>
      </header>

      <main className="main-content">
        <section className="status-card">
          <h2>🔌 Backend Status</h2>
          {loading && <p>Connecting to backend...</p>}
          {error && <p className="error">Error: {error}</p>}
          {health && (
            <div className="status-info">
              <div className="status-badge success">✓ {health.status.toUpperCase()}</div>
              <p><strong>Service:</strong> {health.service}</p>
              <p><strong>Version:</strong> {health.version}</p>
              <p><strong>API URL:</strong> {apiUrl}</p>
            </div>
          )}
        </section>

        <section className="features-grid">
          <div className="feature-card">
            <h3>📡 GPR Data Processing</h3>
            <p>Ground Penetrating Radar signal analysis and utility detection</p>
            <div className="badge">Phase 1A</div>
          </div>

          <div className="feature-card">
            <h3>🏛️ BIM Integration</h3>
            <p>Building Information Modeling file validation and processing</p>
            <div className="badge planned">Phase 1B</div>
          </div>

          <div className="feature-card">
            <h3>📊 LiDAR Processing</h3>
            <p>Point cloud analysis for construction progress monitoring</p>
            <div className="badge planned">Phase 1C</div>
          </div>

          <div className="feature-card">
            <h3>🤖 AI-Powered Reports</h3>
            <p>PAS 128 compliant reporting with LLM integration</p>
            <div className="badge planned">Phase 2</div>
          </div>
        </section>

        <section className="api-links">
          <h2>🔗 Quick Links</h2>
          <div className="links-grid">
            <a href={`${apiUrl}/docs`} target="_blank" rel="noopener noreferrer" className="link-button">
              📚 API Documentation
            </a>
            <a href="http://localhost:9001" target="_blank" rel="noopener noreferrer" className="link-button">
              🗄️ MinIO Console
            </a>
            <a href="/SETUP_COMPLETE.md" target="_blank" rel="noopener noreferrer" className="link-button">
              📖 Setup Guide
            </a>
          </div>
        </section>
      </main>

      <footer className="footer">
        <p>Built with FastAPI + React + TypeScript | Docker Compose Infrastructure</p>
        <p className="tech-stack">
          PostgreSQL · Redis · MinIO · PGVector · PostGIS
        </p>
      </footer>
    </div>
  )
}

export default App
