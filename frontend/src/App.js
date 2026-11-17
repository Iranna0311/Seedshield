import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Hero3D from './components/Hero3D';
import HowItWorks from './components/HowItWorks';
import Features from './components/Features';
import Scanner from './components/Scanner';
import Limitations from './components/Limitations';
import Roadmap from './components/Roadmap';
import Footer from './components/Footer';
import './App.css';

function getInitialTheme() {
  try {
    const saved = localStorage.getItem('seedshield_theme');
    if (saved === 'dark' || saved === 'light') return saved;
  } catch (e) {
    // fallback
  }
  if (typeof window !== 'undefined' && window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    return 'dark';
  }
  return 'light';
}

export default function App() {
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('seedshield_theme', theme);
    } catch (e) {
      // localStorage fallback
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const scrollToScanner = () => {
    const el = document.getElementById('scanner');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="site-layout">
      <Navbar
        theme={theme}
        onToggleTheme={toggleTheme}
        onLaunchScanner={scrollToScanner}
      />

      <main className="site-main">
        <Hero3D
          theme={theme}
          onScanSample={scrollToScanner}
          onUploadOwn={scrollToScanner}
        />

        <HowItWorks />
        <Features />
        <Scanner />
        <Limitations />
        <Roadmap />
      </main>

      <Footer />
    </div>
  );
}
