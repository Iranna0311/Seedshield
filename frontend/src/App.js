import React, { useState, useRef, useCallback, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { PieChart, Pie, Cell, Tooltip as RechartTooltip, Legend as RechartLegend } from 'recharts';
import './App.css';

// API Configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Detection colors
const DETECTION_COLORS = {
  REAL: '#22c55e',
  FAKE: '#ef4444',
  LOW_QUALITY: '#f59e0b',
  UNCERTAIN: '#9ca3af',
};

const LOADING_PHASES = ['Detecting seeds...', 'Classifying...', 'Finalizing...'];

function getInitialTheme() {
  try {
    const stored = localStorage.getItem('seedshield-theme');
    if (stored === 'dark' || stored === 'light') return stored;
  } catch (_) {}
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

function App() {
  // Theme
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('seedshield-theme', theme); } catch (_) {}
  }, [theme]);

  const toggleTheme = () => setTheme(t => t === 'dark' ? 'light' : 'dark');

  // Screen state
  const [screen, setScreen] = useState('instructions'); // 'instructions', 'upload', 'loading', 'results'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [imageQuality, setImageQuality] = useState(null);

  // New state for P4/P5
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [loadingPhase, setLoadingPhase] = useState(0);
  const [hoveredDetection, setHoveredDetection] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const overlayContainerRef = useRef(null);

  const fileInputRef = useRef(null);

  // Loading phase cycling
  useEffect(() => {
    if (screen !== 'loading') return;
    const interval = setInterval(() => {
      setLoadingPhase(p => (p + 1) % 3);
    }, 1200);
    return () => clearInterval(interval);
  }, [screen]);

  // Analyze image quality
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
      let contrastMessage = 'Good image quality';
      let seedStatus = 'success';
      let seedMessage = 'Seed-like content detected';
      let overallOk = true;

      if (avgBrightness < 30) { brightnessStatus = 'error'; brightnessMessage = 'Image too dark - no seeds visible'; overallOk = false; }
      else if (avgBrightness < 60) { brightnessStatus = 'warning'; brightnessMessage = 'Image is quite dark'; }
      else if (avgBrightness > 230) { brightnessStatus = 'error'; brightnessMessage = 'Image too bright - overexposed'; overallOk = false; }
      else if (avgBrightness > 200) { brightnessStatus = 'warning'; brightnessMessage = 'Image is quite bright'; }

      if (contrast < 25) { contrastStatus = 'error'; contrastMessage = 'No seed objects detected - solid/blank image'; overallOk = false; }
      else if (contrast < 45) { contrastStatus = 'warning'; contrastMessage = 'Low contrast - seeds may be hard to detect'; }

      if (greenPercent > 35) { seedStatus = 'error'; seedMessage = 'Foliage/leaves detected - only seed photos are accepted'; overallOk = false; }
      else if (bluePercent > 30) { seedStatus = 'error'; seedMessage = 'Sky/screen/water detected - only seed photos are accepted'; overallOk = false; }
      else if (brownPercent < 4 && skinPercent < 15 && neutralPercent < 15) {
        seedStatus = 'error'; seedMessage = 'No seeds detected - only seed photos are accepted'; overallOk = false;
      } else if (brownPercent > 12 || (brownPercent > 4 && neutralPercent > 15)) {
        seedStatus = 'success'; seedMessage = 'Seed objects detected';
      } else if (skinPercent > 20 && brownPercent > 3) {
        seedStatus = 'success'; seedMessage = 'Seeds on hand background detected';
      } else {
        seedStatus = 'error'; seedMessage = 'No seeds identified - please upload a photo of seeds'; overallOk = false;
      }

      setImageQuality({
        brightness: avgBrightness, contrast,
        brightnessStatus, brightnessMessage, contrastStatus, contrastMessage,
        seedStatus, seedMessage, overallOk,
        brightnessPercent: Math.min(100, (avgBrightness / 255) * 100),
        contrastPercent: Math.min(100, (contrast / 255) * 100),
        seedPercent: Math.min(100, brownPercent + (neutralPercent * 0.5))
      });
    };
    img.src = imageData;
  }, []);

  const handleFileSelect = useCallback((file) => {
    if (!file) return;
    const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
    if (!validTypes.includes(file.type)) { setError('Please upload a valid image (JPEG, PNG, or WebP)'); return; }
    if (file.size > 10 * 1024 * 1024) { setError('File size must be less than 10MB'); return; }
    setError(null); setSelectedFile(file); setImageQuality(null);
    const reader = new FileReader();
    reader.onloadend = () => { setPreviewUrl(reader.result); analyzeImageQuality(reader.result); };
    reader.readAsDataURL(file);
  }, [analyzeImageQuality]);

  const handleInputChange = (e) => handleFileSelect(e.target.files[0]);
  const handleDragEnter = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(true); };
  const handleDragLeave = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); };
  const handleDragOver = (e) => { e.preventDefault(); e.stopPropagation(); };
  const handleDrop = (e) => { e.preventDefault(); e.stopPropagation(); setIsDragging(false); handleFileSelect(e.dataTransfer.files[0]); };

  const startCapturing = () => setScreen('upload');

  const removeFile = () => {
    setSelectedFile(null); setPreviewUrl(null); setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    if (imageQuality && !imageQuality.overallOk) {
      setError(imageQuality.seedMessage || 'Only seed photos are accepted. Please upload an image with visible seeds.');
      return;
    }
    setScreen('loading'); setError(null); setLoadingPhase(0);

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
      setScreen('results');
    } catch (err) {
      console.error('Analysis error:', err);
      let msg;
      if (!err.response) {
        msg = 'Server unreachable. Please check your connection and try again.';
      } else if (err.response.status === 400) {
        msg = err.response?.data?.detail || err.response?.data?.error || 'No seeds detected. Only seed photos are accepted.';
      } else if (err.response.status === 413) {
        msg = 'Image too large. Please upload an image smaller than 8MB.';
      } else {
        msg = err.response?.data?.detail || err.response?.data?.error || 'Failed to analyze image. Please try again.';
      }
      setError(msg);
      setScreen('upload');
    }
  };

  const startOver = () => {
    setSelectedFile(null); setPreviewUrl(null); setResults(null); setError(null);
    setSelectedDetection(null);
    setScreen('instructions');
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const downloadImage = () => {
    if (!results?.annotated_image_base64) return;
    const link = document.createElement('a');
    link.href = `data:image/jpeg;base64,${results.annotated_image_base64}`;
    link.download = `seedshield_analysis_${Date.now()}.jpg`;
    link.click();
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
      case 'genuine': return '✅ These seeds appear to be genuine. They are safe to use for planting. Store in a cool, dry place.';
      case 'suspected_fake': return '❌ WARNING: Fake seeds detected! Do NOT use these seeds for planting. Contact your supplier and report this to the nearest Agriculture Department.';
      case 'low_quality': return '⚠️ These seeds have quality issues. Some may be damaged or shriveled. Select only healthy-looking seeds before planting. Consider buying a fresh batch.';
      default: return '❓ Analysis complete. Please review the results carefully before using these seeds.';
    }
  };

  // Build pie chart data from results
  const buildPieData = (r) => {
    const data = [];
    if (r.real_count > 0) data.push({ name: 'Real', value: r.real_count, color: DETECTION_COLORS.REAL });
    if (r.fake_count > 0) data.push({ name: 'Fake', value: r.fake_count, color: DETECTION_COLORS.FAKE });
    if (r.low_quality_count > 0) data.push({ name: 'Low Quality', value: r.low_quality_count, color: DETECTION_COLORS.LOW_QUALITY });
    if (r.uncertain_count > 0) data.push({ name: 'Uncertain', value: r.uncertain_count, color: DETECTION_COLORS.UNCERTAIN });
    return data;
  };

  // SVG overlay handlers
  const handleRectMouseMove = (e, det) => {
    const container = overlayContainerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    setTooltipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
    setHoveredDetection(det);
  };

  const handleRectMouseLeave = () => setHoveredDetection(null);

  // ========== RENDER INSTRUCTIONS ==========
  const renderInstructions = () => (
    <div className="instructions-section">
      <div>
        <h1 className="instructions-title">How to capture seeds</h1>
        <p className="instructions-subtitle">Follow these steps for the best results</p>
      </div>
      <div className="instruction-list">
        <div className="instruction-item">
          <div className="instruction-icon orange">📋</div>
          <div className="instruction-content">
            <h3>Use plain background <span className="checkmark">✓</span></h3>
            <p>Place seeds on a white or neutral surface for best results</p>
          </div>
        </div>
        <div className="instruction-item">
          <div className="instruction-icon orange">🌱</div>
          <div className="instruction-content">
            <h3>Add 3-5 seeds <span className="checkmark">✓</span></h3>
            <p>Multiple seeds help improve detection accuracy</p>
          </div>
        </div>
        <div className="instruction-item">
          <div className="instruction-icon yellow">💡</div>
          <div className="instruction-content">
            <h3>Ensure good lighting <span className="checkmark">✓</span></h3>
            <p>Natural daylight or bright indoor lighting works best</p>
          </div>
        </div>
        <div className="instruction-item">
          <div className="instruction-icon gray">📷</div>
          <div className="instruction-content">
            <h3>Keep camera steady <span className="checkmark">✓</span></h3>
            <p>Hold still and tap capture when seeds are in focus</p>
          </div>
        </div>
      </div>
      <div className="example-box">
        <div className="example-seeds">
          <div className="seed-dot brown"></div>
          <div className="seed-dot dark-brown"></div>
          <div className="seed-dot light-brown"></div>
        </div>
        <p className="example-text">Example: Seeds on plain background</p>
      </div>
      <button className="btn-primary" onClick={startCapturing}>Got it, Start Capturing</button>
    </div>
  );

  // ========== RENDER UPLOAD ==========
  const renderUpload = () => (
    <div className="instructions-section">
      <h1 className="instructions-title">Upload Seed Image</h1>
      {!selectedFile ? (
        <div
          className={`upload-area ${isDragging ? 'dragging' : ''}`}
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <div className="upload-icon">📸</div>
          <h2 className="upload-title">Tap to upload or take photo</h2>
          <p className="upload-subtitle">JPEG, PNG or WebP - Max 10MB</p>
          <input type="file" ref={fileInputRef} onChange={handleInputChange} accept="image/jpeg,image/png,image/webp" style={{ display: 'none' }} />
        </div>
      ) : (
        <div className="preview-container">
          <img src={previewUrl} alt="Preview" className="preview-image" />
          <div className="preview-actions">
            <span className="preview-name">{selectedFile.name}</span>
            <span className="preview-remove" onClick={removeFile}>✕ Remove</span>
          </div>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}

      {selectedFile && (
        <div className="quality-check-section">
          <div className="quality-header">Image Quality Check</div>
          <div className="quality-metrics">
            {/* Brightness */}
            <div className="quality-metric">
              <div className="metric-header">
                <span className={`metric-icon ${imageQuality?.brightnessStatus || 'success'}`}>
                  {imageQuality?.brightnessStatus === 'error' ? '✗' : imageQuality?.brightnessStatus === 'warning' ? '!' : '✓'}
                </span>
                <span className="metric-title">Brightness</span>
              </div>
              <div className="metric-subtitle">{imageQuality?.brightnessMessage || 'Analyzing...'}</div>
              <div className="progress-bar">
                <div className={`progress-fill ${imageQuality?.brightnessStatus || 'success'}`} style={{ width: `${imageQuality?.brightnessPercent || 0}%` }}></div>
              </div>
            </div>
            {/* Contrast */}
            <div className="quality-metric">
              <div className="metric-header">
                <span className={`metric-icon ${imageQuality?.contrastStatus || 'success'}`}>
                  {imageQuality?.contrastStatus === 'error' ? '✗' : imageQuality?.contrastStatus === 'warning' ? '!' : '✓'}
                </span>
                <span className="metric-title">Object Detection</span>
              </div>
              <div className="metric-subtitle">{imageQuality?.contrastMessage || 'Analyzing...'}</div>
              <div className="progress-bar">
                <div className={`progress-fill ${imageQuality?.contrastStatus || 'success'}`} style={{ width: `${imageQuality?.contrastPercent || 0}%` }}></div>
              </div>
            </div>
            {/* Seed Content */}
            <div className="quality-metric">
              <div className="metric-header">
                <span className={`metric-icon ${imageQuality?.seedStatus || 'success'}`}>
                  {imageQuality?.seedStatus === 'error' ? '✗' : imageQuality?.seedStatus === 'warning' ? '!' : '✓'}
                </span>
                <span className="metric-title">Seed Content</span>
              </div>
              <div className="metric-subtitle">{imageQuality?.seedMessage || 'Analyzing...'}</div>
              <div className="progress-bar">
                <div className={`progress-fill ${imageQuality?.seedStatus || 'success'}`} style={{ width: `${imageQuality?.seedPercent || 0}%` }}></div>
              </div>
            </div>
          </div>
          {imageQuality && !imageQuality.overallOk ? (
            <div className="tip-box" style={{ background: '#fee2e2', borderColor: '#ef4444' }}>
              <span className="tip-icon">⚠️</span>
              <span className="tip-text" style={{ color: '#b91c1c' }}>
                <strong>Warning:</strong> This image may not produce accurate results. Please use a photo with visible seeds on a plain background.
              </span>
            </div>
          ) : (
            <div className="tip-box">
              <span className="tip-icon">💡</span>
              <span className="tip-text">
                <strong>Tip:</strong> For best results, ensure seeds are clearly visible and image is in focus.
              </span>
            </div>
          )}
        </div>
      )}

      <div className="actions">
        {selectedFile ? (
          <>
            {imageQuality && !imageQuality.overallOk && (
              <div className="error-message" style={{ margin: '0.25rem 0', textAlign: 'center' }}>
                ⛔ <strong>Image Rejected:</strong> {imageQuality.seedMessage || 'Only seed photos are accepted.'}
              </div>
            )}
            <button
              className="btn-blue"
              onClick={analyzeImage}
              disabled={Boolean(imageQuality && !imageQuality.overallOk)}
              style={{
                opacity: imageQuality && !imageQuality.overallOk ? 0.45 : 1,
                cursor: imageQuality && !imageQuality.overallOk ? 'not-allowed' : 'pointer',
              }}
            >
              {imageQuality && !imageQuality.overallOk ? '⛔ Seed Photo Required' : 'Analyze Seeds'}
            </button>
            <button className="btn-secondary" onClick={removeFile}>Retake / Choose Another</button>
          </>
        ) : (
          <button className="btn-secondary" onClick={() => setScreen('instructions')}>← Back to Instructions</button>
        )}
      </div>
    </div>
  );

  // ========== RENDER LOADING ==========
  const renderLoading = () => (
    <div className="loading-section">
      {/* Skeleton loaders */}
      <div className="skeleton skeleton-chart"></div>
      <div className="loading-spinner"></div>
      <p className="loading-text">{LOADING_PHASES[loadingPhase]}</p>
      <p className="loading-subtext">Detecting and classifying each seed</p>
      <div className="skeleton skeleton-verdict"></div>
    </div>
  );

  // ========== RENDER RESULTS ==========
  const renderResults = () => {
    const pieData = buildPieData(results);
    const detections = results.detections || [];

    return (
      <div className="results-section">
        {/* Verdict Card */}
        <div className={`verdict-card ${results.verdict}`}>
          <div className="verdict-icon">{getVerdictIcon(results.verdict)}</div>
          <div className="verdict-message">{results.verdict_message}</div>
          <div className="verdict-total">
            {results.total_seeds} seed{results.total_seeds !== 1 ? 's' : ''} analysed
          </div>
          {results.ci95 && (
            <div className="verdict-ci">
              Fake rate CI: {(results.ci95[0] * 100).toFixed(1)}%–{(results.ci95[1] * 100).toFixed(1)}%
            </div>
          )}
          {results.uncertain_count > 0 && (
            <div className="verdict-uncertain">
              ⚠️ {results.uncertain_count} uncertain seed{results.uncertain_count !== 1 ? 's' : ''}
            </div>
          )}
        </div>

        {/* Stats Grid */}
        <div className="stats-grid">
          <div className="stat-card real">
            <div className="stat-label">🟢 Real</div>
            <div className="stat-value">{results.real_count}</div>
            <div className="stat-percent">{results.real_percentage}%</div>
          </div>
          <div className="stat-card fake">
            <div className="stat-label">🔴 Fake</div>
            <div className="stat-value">{results.fake_count}</div>
            <div className="stat-percent">{results.fake_percentage}%</div>
          </div>
          <div className="stat-card low-quality">
            <div className="stat-label">🟡 Low</div>
            <div className="stat-value">{results.low_quality_count}</div>
            <div className="stat-percent">{results.low_quality_percentage}%</div>
          </div>
        </div>

        {/* Donut Chart */}
        {pieData.length > 0 && (
          <div className="chart-section">
            <div className="section-header">📊 Seed Distribution</div>
            <div className="chart-container">
              <PieChart width={280} height={200}>
                <Pie
                  data={pieData}
                  cx={140}
                  cy={90}
                  innerRadius={50}
                  outerRadius={80}
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <RechartTooltip formatter={(value, name) => [`${value} seeds`, name]} />
                <RechartLegend />
              </PieChart>
            </div>
          </div>
        )}

        {/* Farmer Recommendation */}
        <div className={`tip-box farmer-recommendation ${results.verdict}`}>
          <span className="tip-icon">👨‍🌾</span>
          <span className="tip-text">
            <strong>Farmer Advice: </strong>{getFarmerRecommendation(results.verdict)}
          </span>
        </div>

        {/* SVG Overlay Image with detections */}
        {previewUrl && detections.length > 0 && (
          <div className="annotated-image-section">
            <div className="section-header">🖼️ Analyzed Image (click a box for details)</div>
            <div
              className="annotated-image-container"
              style={{ position: 'relative', display: 'inline-block', width: '100%' }}
              ref={overlayContainerRef}
            >
              <img
                src={previewUrl}
                alt="Analyzed seeds"
                className="annotated-image"
                style={{ display: 'block', width: '100%' }}
              />
              <svg
                style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
                viewBox="0 0 1 1"
                preserveAspectRatio="none"
              >
                <AnimatePresence>
                  {detections.map((det, index) => {
                    const bn = det.bbox_norm;
                    if (!bn || bn.length < 4) return null;
                    const [x, y, w, h] = bn;
                    const color = DETECTION_COLORS[det.class_name] || DETECTION_COLORS.UNCERTAIN;
                    return (
                      <motion.rect
                        key={det.id || index}
                        x={x} y={y} width={w} height={h}
                        fill="none"
                        stroke={color}
                        strokeWidth={0.005}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: index * 0.04 }}
                        style={{ cursor: 'pointer' }}
                        onClick={() => setSelectedDetection(det)}
                        onMouseMove={(e) => handleRectMouseMove(e, det)}
                        onMouseLeave={handleRectMouseLeave}
                      />
                    );
                  })}
                </AnimatePresence>
              </svg>
              {/* Hover tooltip */}
              {hoveredDetection && (
                <div
                  className="detection-tooltip"
                  style={{
                    position: 'absolute',
                    left: tooltipPos.x + 8,
                    top: tooltipPos.y - 36,
                    pointerEvents: 'none',
                  }}
                >
                  <strong>{hoveredDetection.class_name}</strong>
                  {' '}{(hoveredDetection.confidence * 100).toFixed(1)}%
                </div>
              )}
            </div>
            <div className="legend">
              <div className="legend-item"><span className="legend-color real"></span><span>Real</span></div>
              <div className="legend-item"><span className="legend-color fake"></span><span>Fake</span></div>
              <div className="legend-item"><span className="legend-color low-quality"></span><span>Low Quality</span></div>
              <div className="legend-item"><span className="legend-color uncertain"></span><span>Uncertain</span></div>
            </div>
          </div>
        )}

        {/* Fallback: show annotated image from backend if no detections array */}
        {results.annotated_image_base64 && (!detections || detections.length === 0) && (
          <div className="annotated-image-section">
            <div className="section-header">🖼️ Analyzed Image</div>
            <div className="annotated-image-container">
              <img
                src={`data:image/jpeg;base64,${results.annotated_image_base64}`}
                alt="Analyzed seeds with annotations"
                className="annotated-image"
              />
            </div>
          </div>
        )}

        {/* Actions */}
        <div className="actions">
          <button className="btn-primary" onClick={startOver}>📸 Analyze Another Image</button>
          {results.annotated_image_base64 && (
            <button className="btn-secondary" onClick={downloadImage}>💾 Download Result</button>
          )}
        </div>

        {/* Detection detail side panel */}
        <AnimatePresence>
          {selectedDetection && (
            <motion.div
              className="detection-panel"
              initial={{ opacity: 0, x: 60 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: 60 }}
              transition={{ type: 'spring', stiffness: 300, damping: 30 }}
            >
              <div className="detection-panel-header">
                <span>Seed Detail</span>
                <button className="detection-panel-close" onClick={() => setSelectedDetection(null)}>✕</button>
              </div>
              <div className="detection-panel-body">
                <div className="detection-label" style={{ color: DETECTION_COLORS[selectedDetection.class_name] || '#6b7280' }}>
                  {selectedDetection.class_name}
                </div>
                <div className="detection-confidence">
                  Confidence: <strong>{(selectedDetection.confidence * 100).toFixed(1)}%</strong>
                </div>

                {selectedDetection.probs && (
                  <div className="prob-bars">
                    <div className="prob-bar-title">Class Probabilities</div>
                    {['REAL', 'FAKE', 'LOW_QUALITY'].map(cls => {
                      const val = selectedDetection.probs[cls] || 0;
                      return (
                        <div key={cls} className="prob-bar-row">
                          <span className="prob-bar-label">{cls.replace('_', ' ')}</span>
                          <div className="prob-bar-track">
                            <div
                              className="prob-bar-fill"
                              style={{ width: `${(val * 100).toFixed(1)}%`, background: DETECTION_COLORS[cls] }}
                            ></div>
                          </div>
                          <span className="prob-bar-pct">{(val * 100).toFixed(1)}%</span>
                        </div>
                      );
                    })}
                  </div>
                )}

                <div className="gradcam-section">
                  <button className="btn-gradcam" disabled>Show Grad-CAM</button>
                  <p className="gradcam-note">Grad-CAM not available (roadmap) — backend uses sklearn, not Keras.</p>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    );
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1 className="header-title">🌱 SeedShield</h1>
        <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
          {theme === 'dark' ? '☀️' : '🌙'}
        </button>
      </header>

      {/* Main Content */}
      <main className="main">
        {screen === 'instructions' && renderInstructions()}
        {screen === 'upload' && renderUpload()}
        {screen === 'loading' && renderLoading()}
        {screen === 'results' && renderResults()}
      </main>
    </div>
  );
}

export default App;
