import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, Tooltip as RechartTooltip, Legend as RechartLegend } from 'recharts';

const API_URL = process.env.REACT_APP_API_URL || 'https://seedshield-backend.onrender.com';

const DETECTION_COLORS = {
  REAL: '#22c55e',
  FAKE: '#ef4444',
  LOW_QUALITY: '#f59e0b',
  UNCERTAIN: '#9ca3af',
};

const LOADING_PHASES = [
  'Detecting seed contours...',
  'Extracting morphological features...',
  'Classifying with Random Forest...',
];

const SAMPLE_SEEDS = [
  { id: 'genuine', name: 'Genuine Seeds', file: 'sample_genuine.jpg', desc: 'Uniform oval seeds' },
  { id: 'mixed', name: 'Mixed Batch', file: 'sample_mixed.jpg', desc: 'Varied size & color' },
  { id: 'damaged', name: 'Damaged Batch', file: 'sample_damaged.jpg', desc: 'Shriveled contours' },
];

export default function Scanner() {
  const [screen, setScreen] = useState('upload'); // 'upload', 'loading', 'results'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [imageQuality, setImageQuality] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [hoveredDetection, setHoveredDetection] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [scanHistory, setScanHistory] = useState([]);
  const [filterClass, setFilterClass] = useState('ALL'); // 'ALL', 'FAKE', 'REAL', 'LOW_QUALITY'
  const [viewMode, setViewMode] = useState('interactive'); // 'interactive', 'annotated'
  const overlayContainerRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load scan history from localStorage
  useEffect(() => {
    try {
      const saved = localStorage.getItem('seedshield_scan_history');
      if (saved) {
        setScanHistory(JSON.parse(saved).slice(0, 5));
      }
    } catch (e) {
      console.warn('Could not read scan history from localStorage:', e);
    }
  }, []);

  // Save to history helper
  const saveToHistory = useCallback((res, thumb) => {
    try {
      const item = {
        id: res.scan_id || Date.now().toString(),
        verdict: res.verdict,
        verdict_message: res.verdict_message,
        total_seeds: res.total_seeds,
        fake_percentage: res.fake_percentage,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        thumb: thumb || null,
      };
      setScanHistory((prev) => {
        const updated = [item, ...prev.filter((p) => p.id !== item.id)].slice(0, 5);
        try {
          localStorage.setItem('seedshield_scan_history', JSON.stringify(updated));
        } catch (err) {
          // localStorage quota exceeded fallback
        }
        return updated;
      });
    } catch (e) {
      console.warn('Could not save history:', e);
    }
  }, []);

  // Cycle loading phase status text
  useEffect(() => {
    if (screen !== 'loading') return;
    const interval = setInterval(() => {
      setLoadingPhase((p) => (p + 1) % 3);
    }, 1100);
    return () => clearInterval(interval);
  }, [screen]);

  // Analyze image quality & seed content
  const analyzeImageQuality = useCallback((imageData) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
      const maxSize = 100;
      const ratio = Math.min(maxSize / img.width, maxSize / img.height);
      canvas.width = img.width * ratio;
      canvas.height = img.height * ratio;
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

      const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
      const data = imgData.data;

      let totalBrightness = 0;
      let minBrightness = 255;
      let maxBrightness = 0;
      let greenPixels = 0;
      let bluePixels = 0;
      let brownPixels = 0;
      let neutralPixels = 0;
      let skinPixels = 0;

      const pixelCount = data.length / 4;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];
        const brightness = (r + g + b) / 3;
        totalBrightness += brightness;
        minBrightness = Math.min(minBrightness, brightness);
        maxBrightness = Math.max(maxBrightness, brightness);

        if (g > r + 20 && g > b + 20 && g > 60) greenPixels++;
        else if (b > r + 30 && b > g + 10 && b > 80) bluePixels++;
        else if (r > g && r > b && r > 80 && r < 200 && Math.abs(r - g) < 80 && b < g) brownPixels++;
        else if (Math.abs(r - g) < 20 && Math.abs(g - b) < 20 && brightness > 150) neutralPixels++;
        else if (r > 95 && g > 40 && b > 20 && r > g && r > b && Math.abs(r - g) > 15 && brightness > 100 && brightness < 220) skinPixels++;
      }

      const avgBrightness = totalBrightness / pixelCount;
      const contrast = maxBrightness - minBrightness;
      const greenPercent = (greenPixels / pixelCount) * 100;
      const bluePercent = (bluePixels / pixelCount) * 100;
      const brownPercent = (brownPixels / pixelCount) * 100;
      const neutralPercent = (neutralPixels / pixelCount) * 100;
      const skinPercent = (skinPixels / pixelCount) * 100;

      let brightnessStatus = 'success';
      let brightnessMessage = 'Good lighting detected';
      let contrastStatus = 'success';
      let contrastMessage = 'Good image contrast';
      let seedStatus = 'success';
      let seedMessage = 'Seed-like content detected';
      let overallOk = true;

      if (avgBrightness < 30) { brightnessStatus = 'error'; brightnessMessage = 'Image too dark'; overallOk = false; }
      else if (avgBrightness < 60) { brightnessStatus = 'warning'; brightnessMessage = 'Image is quite dark'; }
      else if (avgBrightness > 230) { brightnessStatus = 'error'; brightnessMessage = 'Image overexposed'; overallOk = false; }
      else if (avgBrightness > 200) { brightnessStatus = 'warning'; brightnessMessage = 'Image is quite bright'; }

      if (contrast < 25) { contrastStatus = 'error'; contrastMessage = 'No seed objects detected - solid/blank'; overallOk = false; }
      else if (contrast < 45) { contrastStatus = 'warning'; contrastMessage = 'Low contrast'; }

      if (greenPercent > 35) { seedStatus = 'error'; seedMessage = 'Foliage/leaves detected - seed photo required'; overallOk = false; }
      else if (bluePercent > 30) { seedStatus = 'error'; seedMessage = 'Non-seed background (screen/water)'; overallOk = false; }
      else if (brownPercent < 4 && skinPercent < 15 && neutralPercent < 15) {
        seedStatus = 'error'; seedMessage = 'No seeds detected - seed photo required'; overallOk = false;
      } else if (brownPercent > 12 || (brownPercent > 4 && neutralPercent > 15)) {
        seedStatus = 'success'; seedMessage = 'Seed objects detected';
      } else if (skinPercent > 20 && brownPercent > 3) {
        seedStatus = 'success'; seedMessage = 'Seeds on hand background detected';
      } else {
        seedStatus = 'error'; seedMessage = 'No seeds identified - please upload seeds on plain surface'; overallOk = false;
      }

      setImageQuality({
        brightness: avgBrightness,
        contrast,
        brightnessStatus,
        brightnessMessage,
        contrastStatus,
        contrastMessage,
        seedStatus,
        seedMessage,
        overallOk,
        brightnessPercent: Math.min(100, (avgBrightness / 255) * 100),
        contrastPercent: Math.min(100, (contrast / 255) * 100),
        seedPercent: Math.min(100, brownPercent + (neutralPercent * 0.5)),
      });
    };
    img.src = imageData;
  }, []);

  // Handle local file selection
  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) {
      setError('Please upload a valid image (JPEG, PNG, or WebP)');
      return;
    }
    if (file.size > 8 * 1024 * 1024) {
      setError('File size must be less than 8MB');
      return;
    }
    setError(null);
    setSelectedFile(file);
    setImageQuality(null);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
      analyzeImageQuality(reader.result);
    };
    reader.readAsDataURL(file);
  }, [analyzeImageQuality]);

  // Load a pre-bundled sample image
  const loadSample = async (sample) => {
    try {
      setError(null);
      const url = `${process.env.PUBLIC_URL || ''}/samples/${sample.file}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(`Could not load ${sample.file}`);
      const blob = await res.blob();
      const file = new File([blob], sample.file, { type: 'image/jpeg' });
      handleFileSelect(file);
    } catch (err) {
      console.error('Failed to load sample:', err);
      setError(`Failed to load ${sample.name}. Please try uploading a file.`);
    }
  };

  const handleInputChange = (e) => handleFileSelect(e.target.files[0]);
  const handleDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    handleFileSelect(e.dataTransfer.files[0]);
  };

  const removeFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    setImageQuality(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Submit image to /predict
  const analyzeImage = async () => {
    if (!selectedFile) return;
    if (imageQuality && !imageQuality.overallOk) {
      setError(imageQuality.seedMessage || 'Only seed photos are accepted.');
      return;
    }

    setScreen('loading');
    setError(null);
    setLoadingPhase(0);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${API_URL}/predict?return_image=true`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });

      if (!response.data.total_seeds || response.data.total_seeds === 0) {
        setError('No seeds detected in this image. Only seed photos are accepted.');
        setScreen('upload');
        return;
      }

      setResults(response.data);
      setSelectedDetection(null);
      saveToHistory(response.data, previewUrl);
      setScreen('results');
    } catch (err) {
      console.error('Analysis error:', err);
      let msg;
      if (!err.response) {
        msg = 'Server unreachable. Please check your connection and try again.';
      } else if (err.response.status === 400) {
        msg = err.response?.data?.error || err.response?.data?.detail || 'No seeds detected. Only seed photos are accepted.';
      } else if (err.response.status === 413) {
        msg = 'Image too large. Maximum size is 8MB.';
      } else {
        msg = err.response?.data?.detail || err.response?.data?.error || 'Analysis failed. Please try again.';
      }
      setError(msg);
      setScreen('upload');
    }
  };

  const startOver = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResults(null);
    setError(null);
    setSelectedDetection(null);
    setFilterClass('ALL');
    setViewMode('interactive');
    setScreen('upload');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  // Download annotated image
  const downloadImage = () => {
    if (!results?.annotated_image_base64) return;
    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${results.annotated_image_base64}`;
    link.download = `seedshield_scan_${results.scan_id || Date.now()}.jpg`;
    link.click();
  };

  // Download JSON scan report
  const downloadReportJson = () => {
    if (!results) return;
    const report = {
      scan_id: results.scan_id,
      timestamp: new Date().toISOString(),
      model_version: results.model_version,
      verdict: results.verdict,
      verdict_message: results.verdict_message,
      summary: {
        total_seeds: results.total_seeds,
        real_count: results.real_count,
        fake_count: results.fake_count,
        low_quality_count: results.low_quality_count,
        uncertain_count: results.uncertain_count,
        fake_rate_95_ci: results.ci95,
      },
      detections: (results.detections || []).map((d) => ({
        id: d.id,
        class_name: d.class_name,
        confidence: d.confidence,
        bbox_norm: d.bbox_norm,
        probabilities: d.probs,
      })),
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `seedshield_report_${results.scan_id || Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const getVerdictIcon = (verdict) => {
    switch (verdict) {
      case 'genuine': return '✅';
      case 'suspected_fake': return '❌';
      case 'low_quality': return '⚠️';
      default: return '❓';
    }
  };

  const getFarmerRecommendation = (verdict) => {
    switch (verdict) {
      case 'genuine':
        return 'These seeds appear genuine based on morphological profiling. Safe for standard planting preparation.';
      case 'suspected_fake':
        return 'WARNING: Suspected counterfeit seeds detected! Avoid planting this batch. Contact your agricultural extension agent or distributor.';
      case 'low_quality':
        return 'Quality issues detected (damaged or shriveled grains). Manual sorting recommended before sowing.';
      default:
        return 'Review the detailed per-seed classification breakdown before planting.';
    }
  };

  // Handle mouse move over SVG overlay
  const handleRectMouseMove = (e, det) => {
    if (!overlayContainerRef.current) return;
    const rect = overlayContainerRef.current.getBoundingClientRect();
    setTooltipPos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
    setHoveredDetection(det);
  };

  const handleRectMouseLeave = () => setHoveredDetection(null);

  // Build pie chart data
  const buildPieData = (res) => {
    if (!res) return [];
    const data = [];
    if (res.real_count > 0) data.push({ name: 'Real', value: res.real_count, color: DETECTION_COLORS.REAL });
    if (res.fake_count > 0) data.push({ name: 'Fake', value: res.fake_count, color: DETECTION_COLORS.FAKE });
    if (res.low_quality_count > 0) data.push({ name: 'Low Quality', value: res.low_quality_count, color: DETECTION_COLORS.LOW_QUALITY });
    if (res.uncertain_count > 0) data.push({ name: 'Uncertain', value: res.uncertain_count, color: DETECTION_COLORS.UNCERTAIN });
    return data;
  };

  // Render Upload Screen
  const renderUpload = () => (
    <div className="scanner-card">
      <div className="scanner-header">
        <h2 className="scanner-title">Interactive Seed Scanner</h2>
        <p className="scanner-subtitle">
          Upload a high-contrast photo of your seeds or click a pre-loaded sample below for an instant demo.
        </p>
      </div>

      {/* Instant Demo Samples */}
      <div className="sample-picker-block">
        <span className="sample-picker-title">⚡ Instant Demo Samples:</span>
        <div className="sample-buttons">
          {SAMPLE_SEEDS.map((s) => (
            <button
              key={s.id}
              className="btn-sample"
              onClick={() => loadSample(s)}
              type="button"
            >
              <span className="sample-btn-icon">🌱</span>
              <span className="sample-btn-name">{s.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Upload Zone */}
      {!selectedFile ? (
        <div
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click(); }}
        >
          <div className="upload-zone-icon">📸</div>
          <h3 className="upload-zone-title">Click to upload or drag & drop</h3>
          <p className="upload-zone-desc">JPEG, PNG or WebP — Max 8 MB</p>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleInputChange}
            accept="image/jpeg,image/png,image/webp"
            style={{ display: 'none' }}
          />
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Selected seed preview" className="preview-image" />
          <div className="preview-actions">
            <span className="preview-name">{selectedFile.name}</span>
            <button className="preview-remove-btn" onClick={removeFile}>✕ Remove</button>
          </div>
        </div>
      )}

      {error && (
        <div className="scanner-alert error">
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Quality Check Section */}
      {selectedFile && (
        <div className="quality-check-box">
          <div className="quality-header">Pre-Analysis Image Validation</div>
          <div className="quality-metrics-list">
            <div className="quality-metric-row">
              <span className={`metric-tag ${imageQuality?.brightnessStatus || 'success'}`}>
                {imageQuality?.brightnessStatus === 'error' ? '✗' : '✓'}
              </span>
              <div className="metric-info">
                <span className="metric-name">Lighting</span>
                <span className="metric-desc">{imageQuality?.brightnessMessage || 'Analyzing...'}</span>
              </div>
            </div>

            <div className="quality-metric-row">
              <span className={`metric-tag ${imageQuality?.contrastStatus || 'success'}`}>
                {imageQuality?.contrastStatus === 'error' ? '✗' : '✓'}
              </span>
              <div className="metric-info">
                <span className="metric-name">Contrast</span>
                <span className="metric-desc">{imageQuality?.contrastMessage || 'Analyzing...'}</span>
              </div>
            </div>

            <div className="quality-metric-row">
              <span className={`metric-tag ${imageQuality?.seedStatus || 'success'}`}>
                {imageQuality?.seedStatus === 'error' ? '✗' : '✓'}
              </span>
              <div className="metric-info">
                <span className="metric-name">Seed Content</span>
                <span className="metric-desc">{imageQuality?.seedMessage || 'Analyzing...'}</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Action Buttons */}
      <div className="scanner-actions">
        {selectedFile ? (
          <>
            {imageQuality && !imageQuality.overallOk && (
              <div className="scanner-alert error">
                ⛔ <strong>Non-seed photo rejected:</strong> {imageQuality.seedMessage}
              </div>
            )}
            <button
              className="btn-analyze"
              onClick={analyzeImage}
              disabled={Boolean(imageQuality && !imageQuality.overallOk)}
            >
              {imageQuality && !imageQuality.overallOk ? '⛔ Seed Photo Required' : 'Analyze Seeds Now'}
            </button>
            <button className="btn-secondary-link" onClick={removeFile}>
              Choose Another Image
            </button>
          </>
        ) : null}
      </div>

      {/* Scan History (last 5 scans) */}
      {scanHistory.length > 0 && (
        <div className="scan-history-section">
          <span className="history-title">🕒 Recent Scans (Stored Locally)</span>
          <div className="history-chips">
            {scanHistory.map((item) => (
              <div key={item.id} className={`history-chip ${item.verdict}`}>
                <span className="history-chip-icon">{getVerdictIcon(item.verdict)}</span>
                <span className="history-chip-text">{item.total_seeds} seeds • {item.timestamp}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  // Render Loading Screen
  const renderLoading = () => (
    <div className="scanner-card loading-state">
      <div className="scan-line-animation">
        <div className="scan-sweep-beam"></div>
      </div>
      <div className="loading-spinner"></div>
      <h3 className="loading-title">{LOADING_PHASES[loadingPhase]}</h3>
      <p className="loading-subtitle">Computing 20 agronomic features and running Wilson confidence estimations</p>
      <div className="skeleton skeleton-chart"></div>
    </div>
  );

  // Render Results Screen
  const renderResults = () => {
    const pieData = buildPieData(results);
    const detections = results.detections || [];

    return (
      <div className="scanner-card results-layout">
        {/* Verdict Banner */}
        <div className={`verdict-banner ${results.verdict}`}>
          <div className="verdict-banner-icon">{getVerdictIcon(results.verdict)}</div>
          <div className="verdict-banner-content">
            <h3 className="verdict-banner-title">{results.verdict_message}</h3>
            <div className="verdict-banner-meta">
              <span>{results.total_seeds} total seed{results.total_seeds !== 1 ? 's' : ''}</span>
              {results.ci95 && (
                <span className="verdict-ci-tag">
                  Fake Rate 95% CI: {(results.ci95[0] * 100).toFixed(1)}%–{(results.ci95[1] * 100).toFixed(1)}%
                </span>
              )}
              {results.uncertain_count > 0 && (
                <span className="verdict-uncertain-tag">
                  {results.uncertain_count} Uncertain
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="results-stats-grid">
          <div className="result-stat-box real">
            <span className="stat-label">🟢 Real</span>
            <span className="stat-value">{results.real_count}</span>
            <span className="stat-pct">{results.real_percentage}%</span>
          </div>
          <div className="result-stat-box fake">
            <span className="stat-label">🔴 Fake</span>
            <span className="stat-value">{results.fake_count}</span>
            <span className="stat-pct">{results.fake_percentage}%</span>
          </div>
          <div className="result-stat-box low">
            <span className="stat-label">🟡 Low Quality</span>
            <span className="stat-value">{results.low_quality_count}</span>
            <span className="stat-pct">{results.low_quality_percentage}%</span>
          </div>
        </div>

        {/* Donut Chart */}
        {pieData.length > 0 && (
          <div className="results-chart-box">
            <h4 className="chart-box-title">Batch Class Distribution</h4>
            <div className="chart-canvas-wrapper">
              <PieChart width={300} height={200}>
                <Pie
                  data={pieData}
                  cx={150}
                  cy={90}
                  innerRadius={50}
                  outerRadius={80}
                  dataKey="value"
                >
                  {pieData.map((entry, idx) => (
                    <Cell key={idx} fill={entry.color} />
                  ))}
                </Pie>
                <RechartTooltip formatter={(val, name) => [`${val} seeds`, name]} />
                <RechartLegend />
              </PieChart>
            </div>
          </div>
        )}

        {/* Farmer Recommendation */}
        <div className={`recommendation-box ${results.verdict}`}>
          <span className="rec-icon">👨‍🌾</span>
          <div className="rec-text">
            <strong>Agronomic Advice: </strong>
            {getFarmerRecommendation(results.verdict)}
          </div>
        </div>

        {/* Interactive Visual Overlay & Effects */}
        {previewUrl && detections.length > 0 && (
          <div className="overlay-wrapper-card">
            <div className="overlay-header">
              <div className="overlay-header-info">
                <span className="overlay-title">🔍 Seed Quality Analysis Map</span>
                <span className="overlay-subtitle">Click any seed to inspect features</span>
              </div>
              <div className="overlay-view-controls">
                <button
                  type="button"
                  className={`overlay-mode-btn ${viewMode === 'interactive' ? 'active' : ''}`}
                  onClick={() => setViewMode('interactive')}
                >
                  ✨ Visual Effects
                </button>
                {results.annotated_image_base64 && (
                  <button
                    type="button"
                    className={`overlay-mode-btn ${viewMode === 'annotated' ? 'active' : ''}`}
                    onClick={() => setViewMode('annotated')}
                  >
                    🔬 Deep-Scan Photo
                  </button>
                )}
              </div>
            </div>

            {/* Quick Filter Bar */}
            <div className="overlay-filter-bar">
              <span className="filter-label">Filter:</span>
              <button
                type="button"
                className={`filter-chip ${filterClass === 'ALL' ? 'active' : ''}`}
                onClick={() => setFilterClass('ALL')}
              >
                All ({detections.length})
              </button>
              <button
                type="button"
                className={`filter-chip fake ${filterClass === 'FAKE' ? 'active' : ''}`}
                onClick={() => setFilterClass('FAKE')}
              >
                🚨 Fake Seeds ({results.fake_count})
              </button>
              <button
                type="button"
                className={`filter-chip real ${filterClass === 'REAL' ? 'active' : ''}`}
                onClick={() => setFilterClass('REAL')}
              >
                ✓ Genuine Seeds ({results.real_count})
              </button>
              {results.low_quality_count > 0 && (
                <button
                  type="button"
                  className={`filter-chip low ${filterClass === 'LOW_QUALITY' ? 'active' : ''}`}
                  onClick={() => setFilterClass('LOW_QUALITY')}
                >
                  ⚠️ Low Quality ({results.low_quality_count})
                </button>
              )}
            </div>

            {/* Main Image Display Area */}
            {viewMode === 'annotated' && results.annotated_image_base64 ? (
              <div className="overlay-image-area">
                <img
                  src={`data:image/jpeg;base64,${results.annotated_image_base64}`}
                  alt="Deep scan annotated analysis"
                  className="overlay-base-img"
                />
              </div>
            ) : (
              <div
                className="overlay-image-area"
                ref={overlayContainerRef}
              >
                <img
                  src={previewUrl}
                  alt="Analyzed seed sample"
                  className="overlay-base-img"
                />
                <svg
                  className="overlay-svg-layer"
                  viewBox="0 0 1 1"
                  preserveAspectRatio="none"
                >
                  <defs>
                    {/* Glowing drop shadow filter for fake seeds */}
                    <filter id="fakeGlowFilter" x="-30%" y="-30%" width="160%" height="160%">
                      <feDropShadow dx="0" dy="0" stdDeviation="0.007" floodColor="#ef4444" floodOpacity="0.9" />
                      <feDropShadow dx="0" dy="0" stdDeviation="0.014" floodColor="#dc2626" floodOpacity="0.7" />
                    </filter>

                    {/* Verification glow filter for real seeds */}
                    <filter id="realGlowFilter" x="-20%" y="-20%" width="140%" height="140%">
                      <feDropShadow dx="0" dy="0" stdDeviation="0.004" floodColor="#22c55e" floodOpacity="0.8" />
                    </filter>

                    {/* Warning diagonal hatch pattern for fake seeds */}
                    <pattern id="fakeHazardPattern" width="0.03" height="0.03" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">
                      <line x1="0" y1="0" x2="0" y2="0.03" stroke="rgba(239, 68, 68, 0.45)" strokeWidth="0.006" />
                    </pattern>
                  </defs>

                  <AnimatePresence>
                    {detections.map((det, index) => {
                      const bn = det.bbox_norm;
                      if (!bn || bn.length < 4) return null;
                      const [x, y, w, h] = bn;
                      const isFake = det.class_name === 'FAKE';
                      const isReal = det.class_name === 'REAL';
                      const isLow = det.class_name === 'LOW_QUALITY';
                      const isDimmed = filterClass !== 'ALL' && filterClass !== det.class_name;
                      const cLen = Math.min(w, h) * 0.35;
                      const badgeW = Math.max(0.09, Math.min(w * 0.95, 0.14));
                      const badgeH = 0.026;
                      const badgeY = y > 0.035 ? y - badgeH - 0.004 : y + h + 0.004;

                      return (
                        <g
                          key={det.id || index}
                          className={`seed-detection-group ${isFake ? 'fake-seed-active' : ''} ${isDimmed ? 'dimmed' : ''}`}
                          style={{
                            cursor: 'pointer',
                            opacity: isDimmed ? 0.18 : 1,
                            transition: 'opacity 0.25s ease',
                          }}
                          onClick={() => setSelectedDetection(det)}
                          onMouseMove={(e) => handleRectMouseMove(e, det)}
                          onMouseLeave={handleRectMouseLeave}
                        >
                          {/* FAKE SEED SPECIAL EFFECTS */}
                          {isFake && (
                            <>
                              {/* Pulsing outer aura ring */}
                              <rect
                                x={x - 0.006}
                                y={y - 0.006}
                                width={w + 0.012}
                                height={h + 0.012}
                                rx={0.006}
                                fill="none"
                                stroke="#ef4444"
                                strokeWidth={0.004}
                                strokeDasharray="0.02 0.01"
                                className="fake-pulse-ring"
                              />
                              {/* Hazard striped fill */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="url(#fakeHazardPattern)"
                              />
                              {/* Alert red tint */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="rgba(239, 68, 68, 0.22)"
                              />
                              {/* Bold flashing main border */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="none"
                                stroke="#ef4444"
                                strokeWidth={0.008}
                                filter="url(#fakeGlowFilter)"
                                className="seed-fake-effect"
                              />
                              {/* Tactical Danger Corner Reticles */}
                              <path
                                d={`M ${x} ${y + cLen} L ${x} ${y} L ${x + cLen} ${y}
                                    M ${x + w - cLen} ${y} L ${x + w} ${y} L ${x + w} ${y + cLen}
                                    M ${x} ${y + h - cLen} L ${x} ${y + h} L ${x + cLen} ${y + h}
                                    M ${x + w - cLen} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - cLen}`}
                                fill="none"
                                stroke="#ffffff"
                                strokeWidth={0.005}
                              />
                              {/* Floating Danger Tag */}
                              <g transform={`translate(${x}, ${badgeY})`}>
                                <rect
                                  width={badgeW}
                                  height={badgeH}
                                  rx={0.004}
                                  fill="#dc2626"
                                  stroke="#ffffff"
                                  strokeWidth={0.002}
                                />
                                <text
                                  x={badgeW / 2}
                                  y={badgeH * 0.72}
                                  fill="#ffffff"
                                  fontSize={0.016}
                                  fontWeight="900"
                                  textAnchor="middle"
                                  fontFamily="system-ui, sans-serif"
                                >
                                  🚨 FAKE
                                </text>
                              </g>
                            </>
                          )}

                          {/* REAL SEED EFFECTS */}
                          {isReal && (
                            <>
                              {/* Subtle authentic green tint */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="rgba(34, 197, 94, 0.16)"
                              />
                              {/* Clean emerald verification border with soft glow */}
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="none"
                                stroke="#22c55e"
                                strokeWidth={0.005}
                                filter="url(#realGlowFilter)"
                              />
                              {/* Subtle green corner check accents */}
                              <path
                                d={`M ${x} ${y + cLen} L ${x} ${y} L ${x + cLen} ${y}
                                    M ${x + w - cLen} ${y} L ${x + w} ${y} L ${x + w} ${y + cLen}
                                    M ${x} ${y + h - cLen} L ${x} ${y + h} L ${x + cLen} ${y + h}
                                    M ${x + w - cLen} ${y + h} L ${x + w} ${y + h} L ${x + w} ${y + h - cLen}`}
                                fill="none"
                                stroke="#86efac"
                                strokeWidth={0.004}
                              />
                              {/* Verification Tag */}
                              <g transform={`translate(${x}, ${badgeY})`}>
                                <rect
                                  width={badgeW}
                                  height={badgeH}
                                  rx={0.004}
                                  fill="#15803d"
                                  stroke="#86efac"
                                  strokeWidth={0.002}
                                />
                                <text
                                  x={badgeW / 2}
                                  y={badgeH * 0.72}
                                  fill="#ffffff"
                                  fontSize={0.016}
                                  fontWeight="bold"
                                  textAnchor="middle"
                                  fontFamily="system-ui, sans-serif"
                                >
                                  ✓ REAL
                                </text>
                              </g>
                            </>
                          )}

                          {/* LOW QUALITY SEED EFFECTS */}
                          {isLow && (
                            <>
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="rgba(245, 158, 11, 0.14)"
                              />
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="none"
                                stroke="#f59e0b"
                                strokeWidth={0.005}
                                strokeDasharray="0.02 0.01"
                              />
                              <g transform={`translate(${x}, ${badgeY})`}>
                                <rect
                                  width={badgeW}
                                  height={badgeH}
                                  rx={0.004}
                                  fill="#b45309"
                                  stroke="#fde68a"
                                  strokeWidth={0.002}
                                />
                                <text
                                  x={badgeW / 2}
                                  y={badgeH * 0.72}
                                  fill="#ffffff"
                                  fontSize={0.015}
                                  fontWeight="bold"
                                  textAnchor="middle"
                                  fontFamily="system-ui, sans-serif"
                                >
                                  ⚠️ LOW
                                </text>
                              </g>
                            </>
                          )}

                          {/* UNCERTAIN SEED EFFECTS */}
                          {!isFake && !isReal && !isLow && (
                            <>
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="rgba(156, 163, 175, 0.12)"
                              />
                              <rect
                                x={x}
                                y={y}
                                width={w}
                                height={h}
                                fill="none"
                                stroke="#9ca3af"
                                strokeWidth={0.004}
                                strokeDasharray="0.015 0.01"
                              />
                              <g transform={`translate(${x}, ${badgeY})`}>
                                <rect
                                  width={badgeW}
                                  height={badgeH}
                                  rx={0.004}
                                  fill="#4b5563"
                                  stroke="#d1d5db"
                                  strokeWidth={0.002}
                                />
                                <text
                                  x={badgeW / 2}
                                  y={badgeH * 0.72}
                                  fill="#ffffff"
                                  fontSize={0.014}
                                  fontWeight="bold"
                                  textAnchor="middle"
                                  fontFamily="system-ui, sans-serif"
                                >
                                  ? UNCERTAIN
                                </text>
                              </g>
                            </>
                          )}
                        </g>
                      );
                    })}
                  </AnimatePresence>
                </svg>

                {/* Hover Tooltip */}
                {hoveredDetection && (
                  <div
                    className="interactive-tooltip"
                    style={{
                      left: tooltipPos.x + 10,
                      top: tooltipPos.y - 36,
                    }}
                  >
                    <strong>{hoveredDetection.class_name}</strong>{' '}
                    {(hoveredDetection.confidence * 100).toFixed(1)}%
                  </div>
                )}
              </div>
            )}

            <div className="overlay-legend-row">
              <div className="legend-item"><span className="legend-dot real"></span><span>✓ Real Seed (Authentic)</span></div>
              <div className="legend-item"><span className="legend-dot fake"></span><span>🚨 Fake Seed (Suspected Counterfeit)</span></div>
              <div className="legend-item"><span className="legend-dot low"></span><span>⚠️ Low Quality / Damaged</span></div>
              <div className="legend-item"><span className="legend-dot uncertain"></span><span>? Uncertain</span></div>
            </div>
          </div>
        )}

        {/* Actions Row */}
        <div className="results-actions-grid">
          <button className="btn-analyze" onClick={startOver}>
            📸 Scan Another Batch
          </button>
          <button className="btn-action-outline" onClick={downloadReportJson}>
            📄 Download JSON Report
          </button>
          {results.annotated_image_base64 && (
            <button className="btn-action-outline" onClick={downloadImage}>
              💾 Save Annotated Photo
            </button>
          )}
        </div>

        {/* Per-Seed Inspection Panel (Side Panel / Bottom Sheet) */}
        <AnimatePresence>
          {selectedDetection && (
            <motion.div
              className="seed-detail-drawer"
              initial={{ opacity: 0, y: 40 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 40 }}
              transition={{ type: 'spring', stiffness: 300, damping: 28 }}
            >
              <div className="drawer-header">
                <span className="drawer-title">Seed #{selectedDetection.id + 1} Inspector</span>
                <button className="drawer-close-btn" onClick={() => setSelectedDetection(null)}>✕</button>
              </div>

              <div className="drawer-body">
                <div
                  className="drawer-class-tag"
                  style={{ color: DETECTION_COLORS[selectedDetection.class_name] || '#9ca3af' }}
                >
                  {selectedDetection.class_name}
                </div>
                <div className="drawer-conf-row">
                  <span>Confidence:</span>
                  <strong>{(selectedDetection.confidence * 100).toFixed(1)}%</strong>
                </div>

                {selectedDetection.probs && (
                  <div className="drawer-probs-box">
                    <span className="probs-box-title">Class Probabilities:</span>
                    {['REAL', 'FAKE', 'LOW_QUALITY'].map((cls) => {
                      const val = selectedDetection.probs[cls] || 0;
                      return (
                        <div key={cls} className="prob-meter-row">
                          <span className="prob-meter-label">{cls.replace('_', ' ')}</span>
                          <div className="prob-meter-track">
                            <div
                              className="prob-meter-fill"
                              style={{ width: `${(val * 100).toFixed(1)}%`, background: DETECTION_COLORS[cls] }}
                            ></div>
                          </div>
                          <span className="prob-meter-val">{(val * 100).toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                <div className="drawer-note-box">
                  <p className="drawer-note-text">
                    Classified via Random Forest over 20 hand-crafted shape, color, and gradient features.
                  </p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <section className="section-container scanner-wrapper" id="scanner">
      <div className="section-header-block">
        <span className="section-eyebrow">Field Testing</span>
        <h2 className="section-title">Batch Quality Scanner</h2>
        <p className="section-subtitle">
          Test any sample seed photo directly in your browser. All processing is validated and executed via the live backend.
        </p>
      </div>

      <div className="scanner-center-stage">
        {screen === 'upload' && renderUpload()}
        {screen === 'loading' && renderLoading()}
        {screen === 'results' && renderResults()}
      </div>
    </section>
  );
}
