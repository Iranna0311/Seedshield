import React from 'react';

export default function Navbar({ theme, onToggleTheme, onLaunchScanner }) {
  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav className="site-navbar" role="navigation" aria-label="Main Navigation">
      <div className="navbar-container">
        <a href="#top" className="navbar-brand" onClick={(e) => { e.preventDefault(); scrollTo('top'); }}>
          <span className="navbar-logo-icon">🌱</span>
          <span className="navbar-logo-text">SeedShield</span>
        </a>

        <div className="navbar-links">
          <button className="nav-link-btn" onClick={() => scrollTo('how-it-works')}>How it works</button>
          <button className="nav-link-btn" onClick={() => scrollTo('features')}>Features</button>
          <button className="nav-link-btn" onClick={() => scrollTo('limitations')}>Limitations</button>
          <button className="nav-link-btn" onClick={() => scrollTo('roadmap')}>Roadmap</button>
        </div>

        <div className="navbar-actions">
          <button
            className="theme-toggle-btn"
            onClick={onToggleTheme}
            aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          <button
            className="btn-launch-scanner"
            onClick={onLaunchScanner}
          >
            Launch Scanner
          </button>
        </div>
      </div>
    </nav>
  );
}
