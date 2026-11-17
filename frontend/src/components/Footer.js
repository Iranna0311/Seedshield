import React from 'react';

export default function Footer() {
  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <footer className="site-footer">
      <div className="footer-container">
        <div className="footer-top">
          <div className="footer-brand">
            <span className="footer-logo">🌱 SeedShield</span>
            <p className="footer-tagline">
              AI-powered seed quality screening prototype. Built to help smallholder farmers spot adulterated seed batches early.
            </p>
          </div>

          <div className="footer-stack">
            <span className="footer-stack-title">Tech Stack</span>
            <div className="footer-tags">
              <span className="footer-tag">FastAPI</span>
              <span className="footer-tag">OpenCV</span>
              <span className="footer-tag">scikit-learn</span>
              <span className="footer-tag">React 18</span>
              <span className="footer-tag">Three.js</span>
            </div>
          </div>
        </div>

        <div className="footer-bottom">
          <p className="footer-copy">
            © {new Date().getFullYear()} SeedShield. Open-source prototype released under the MIT License.
          </p>
          <button className="footer-back-to-top" onClick={scrollToTop}>
            Back to top ↑
          </button>
        </div>
      </div>
    </footer>
  );
}
