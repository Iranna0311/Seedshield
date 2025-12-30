import React, { useState, useRef, useCallback } from 'react';
import axios from 'axios';
import './App.css';

// API Configuration
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
    // State
    const [screen, setScreen] = useState('instructions'); // 'instructions', 'upload', 'loading', 'results'
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [error, setError] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [imageQuality, setImageQuality] = useState(null);

    const fileInputRef = useRef(null);
    const canvasRef = useRef(null);

    // Analyze image quality (brightness, contrast, and seed content detection)
    const analyzeImageQuality = useCallback((imageData) => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        const img = new Image();

        img.onload = () => {
            // Resize for faster analysis
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

            // Color category counters for seed detection
            let greenPixels = 0;    // Leaves, plants
            let bluePixels = 0;     // Sky, water
            let brownPixels = 0;    // Seeds, grains
            let neutralPixels = 0;  // White/gray background
            let skinPixels = 0;     // Skin tones (hands)

            // Calculate brightness and color distribution for each pixel
            const pixelCount = data.length / 4;
            for (let i = 0; i < data.length; i += 4) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                const brightness = (r + g + b) / 3;
                totalBrightness += brightness;
                minBrightness = Math.min(minBrightness, brightness);
                maxBrightness = Math.max(maxBrightness, brightness);

                // Classify pixel by dominant color
                // Green detection (leaves, plants)
                if (g > r + 20 && g > b + 20 && g > 60) {
                    greenPixels++;
                }
                // Blue detection (sky, water)
                else if (b > r + 30 && b > g + 10 && b > 80) {
                    bluePixels++;
                }
                // Brown/tan detection (typical seed colors)
                else if (r > g && r > b && r > 80 && r < 200 &&
                    Math.abs(r - g) < 80 && b < g) {
                    brownPixels++;
                }
                // Neutral (white/gray - good background)
                else if (Math.abs(r - g) < 20 && Math.abs(g - b) < 20 && brightness > 150) {
                    neutralPixels++;
                }
                // Skin tone detection
                else if (r > 95 && g > 40 && b > 20 && r > g && r > b &&
                    Math.abs(r - g) > 15 && brightness > 100 && brightness < 220) {
                    skinPixels++;
                }
            }

            const avgBrightness = totalBrightness / pixelCount;
            const contrast = maxBrightness - minBrightness;

            // Calculate percentages
            const greenPercent = (greenPixels / pixelCount) * 100;
            const bluePercent = (bluePixels / pixelCount) * 100;
            const brownPercent = (brownPixels / pixelCount) * 100;
            const neutralPercent = (neutralPixels / pixelCount) * 100;
            const skinPercent = (skinPixels / pixelCount) * 100;

            // Determine quality status
            let brightnessStatus = 'success';
            let brightnessMessage = 'Good lighting detected';
            let contrastStatus = 'success';
            let contrastMessage = 'Good image quality';
            let seedStatus = 'success';
            let seedMessage = 'Seed-like content detected';
            let overallOk = true;

            // Brightness checks
            if (avgBrightness < 30) {
                brightnessStatus = 'error';
                brightnessMessage = 'Image too dark - no seeds visible';
                overallOk = false;
            } else if (avgBrightness < 60) {
                brightnessStatus = 'warning';
                brightnessMessage = 'Image is quite dark';
            } else if (avgBrightness > 230) {
                brightnessStatus = 'error';
                brightnessMessage = 'Image too bright - overexposed';
                overallOk = false;
            } else if (avgBrightness > 200) {
                brightnessStatus = 'warning';
                brightnessMessage = 'Image is quite bright';
            }

            // Contrast checks
            if (contrast < 20) {
                contrastStatus = 'error';
                contrastMessage = 'No objects detected - solid color image';
                overallOk = false;
            } else if (contrast < 50) {
                contrastStatus = 'warning';
                contrastMessage = 'Low contrast - seeds may be hard to detect';
            }

            // Seed content detection
            // Reject images with too much green (leaves/plants)
            if (greenPercent > 40) {
                seedStatus = 'error';
                seedMessage = 'Leaves/plants detected - please upload seed image';
                overallOk = false;
            }
            // Reject images with too much blue (sky/water/screenshots)
            else if (bluePercent > 35) {
                seedStatus = 'error';
                seedMessage = 'Non-seed image detected (sky/water/screen)';
                overallOk = false;
            }
            // Check if image has seed-appropriate colors
            else if (brownPercent < 5 && neutralPercent < 30) {
                // No brown (seeds) and no neutral background
                if (greenPercent > 20) {
                    seedStatus = 'error';
                    seedMessage = 'Plant detected - not a seed image';
                    overallOk = false;
                } else {
                    seedStatus = 'warning';
                    seedMessage = 'May not contain seeds - check image';
                }
            }
            // Good seed image indicators
            else if (brownPercent > 15 || (brownPercent > 5 && neutralPercent > 20)) {
                seedStatus = 'success';
                seedMessage = 'Seed-like objects detected';
            }
            // Acceptable with warning
            else if (skinPercent > 20 && brownPercent > 3) {
                seedStatus = 'success';
                seedMessage = 'Seeds with hand background detected';
            }
            else {
                seedStatus = 'warning';
                seedMessage = 'Uncertain - verify image contains seeds';
            }

            setImageQuality({
                brightness: avgBrightness,
                contrast: contrast,
                brightnessStatus,
                brightnessMessage,
                contrastStatus,
                contrastMessage,
                seedStatus,
                seedMessage,
                overallOk,
                brightnessPercent: Math.min(100, (avgBrightness / 255) * 100),
                contrastPercent: Math.min(100, (contrast / 255) * 100),
                seedPercent: Math.min(100, brownPercent + (neutralPercent * 0.5))
            });
        };

        img.src = imageData;
    }, []);

    // Handle file selection
    const handleFileSelect = useCallback((file) => {
        if (!file) return;

        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp'];
        if (!validTypes.includes(file.type)) {
            setError('Please upload a valid image (JPEG, PNG, or WebP)');
            return;
        }

        // Validate file size (10MB max)
        if (file.size > 10 * 1024 * 1024) {
            setError('File size must be less than 10MB');
            return;
        }

        setError(null);
        setSelectedFile(file);
        setImageQuality(null); // Reset quality

        // Create preview and analyze quality
        const reader = new FileReader();
        reader.onloadend = () => {
            setPreviewUrl(reader.result);
            analyzeImageQuality(reader.result);
        };
        reader.readAsDataURL(file);
    }, [analyzeImageQuality]);

    // Handle file input change
    const handleInputChange = (e) => {
        const file = e.target.files[0];
        handleFileSelect(file);
    };

    // Handle drag events
    const handleDragEnter = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    };

    const handleDragLeave = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    };

    const handleDragOver = (e) => {
        e.preventDefault();
        e.stopPropagation();
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);

        const file = e.dataTransfer.files[0];
        handleFileSelect(file);
    };

    // Start capturing (move to upload screen)
    const startCapturing = () => {
        setScreen('upload');
    };

    // Remove selected file
    const removeFile = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setError(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Analyze image
    const analyzeImage = async () => {
        if (!selectedFile) return;

        setIsLoading(true);
        setScreen('loading');
        setError(null);

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await axios.post(`${API_URL}/predict`, formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            setResults(response.data);
            setScreen('results');
        } catch (err) {
            console.error('Analysis error:', err);
            setError(
                err.response?.data?.detail ||
                err.response?.data?.error ||
                'Failed to analyze image. Please try again.'
            );
            setScreen('upload');
        } finally {
            setIsLoading(false);
        }
    };

    // Reset and start over
    const startOver = () => {
        setSelectedFile(null);
        setPreviewUrl(null);
        setResults(null);
        setError(null);
        setScreen('instructions');
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    // Download annotated image
    const downloadImage = () => {
        if (!results?.annotated_image_base64) return;

        const link = document.createElement('a');
        link.href = `data:image/jpeg;base64,${results.annotated_image_base64}`;
        link.download = `seedshield_analysis_${Date.now()}.jpg`;
        link.click();
    };

    // Get verdict styling
    const getVerdictIcon = (verdict) => {
        switch (verdict) {
            case 'genuine': return '✅';
            case 'suspected_fake': return '❌';
            case 'low_quality': return '⚠️';
            default: return '❓';
        }
    };

    // Render Instructions Screen
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

            <button className="btn-primary" onClick={startCapturing}>
                Got it, Start Capturing
            </button>
        </div>
    );

    // Render Upload Screen
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
                        {/* Brightness Check */}
                        <div className="quality-metric">
                            <div className="metric-header">
                                <span className={`metric-icon ${imageQuality?.brightnessStatus || 'success'}`}>
                                    {imageQuality?.brightnessStatus === 'error' ? '✗' :
                                        imageQuality?.brightnessStatus === 'warning' ? '!' : '✓'}
                                </span>
                                <span className="metric-title">Brightness</span>
                            </div>
                            <div className="metric-subtitle">
                                {imageQuality?.brightnessMessage || 'Analyzing...'}
                            </div>
                            <div className="progress-bar">
                                <div
                                    className={`progress-fill ${imageQuality?.brightnessStatus || 'success'}`}
                                    style={{ width: `${imageQuality?.brightnessPercent || 0}%` }}
                                ></div>
                            </div>
                        </div>

                        {/* Contrast/Objects Check */}
                        <div className="quality-metric">
                            <div className="metric-header">
                                <span className={`metric-icon ${imageQuality?.contrastStatus || 'success'}`}>
                                    {imageQuality?.contrastStatus === 'error' ? '✗' :
                                        imageQuality?.contrastStatus === 'warning' ? '!' : '✓'}
                                </span>
                                <span className="metric-title">Object Detection</span>
                            </div>
                            <div className="metric-subtitle">
                                {imageQuality?.contrastMessage || 'Analyzing...'}
                            </div>
                            <div className="progress-bar">
                                <div
                                    className={`progress-fill ${imageQuality?.contrastStatus || 'success'}`}
                                    style={{ width: `${imageQuality?.contrastPercent || 0}%` }}
                                ></div>
                            </div>
                        </div>

                        {/* Seed Content Check */}
                        <div className="quality-metric">
                            <div className="metric-header">
                                <span className={`metric-icon ${imageQuality?.seedStatus || 'success'}`}>
                                    {imageQuality?.seedStatus === 'error' ? '✗' :
                                        imageQuality?.seedStatus === 'warning' ? '!' : '✓'}
                                </span>
                                <span className="metric-title">Seed Content</span>
                            </div>
                            <div className="metric-subtitle">
                                {imageQuality?.seedMessage || 'Analyzing...'}
                            </div>
                            <div className="progress-bar">
                                <div
                                    className={`progress-fill ${imageQuality?.seedStatus || 'success'}`}
                                    style={{ width: `${imageQuality?.seedPercent || 0}%` }}
                                ></div>
                            </div>
                        </div>
                    </div>

                    {/* Warning tip for bad images */}
                    {imageQuality && !imageQuality.overallOk ? (
                        <div className="tip-box" style={{ background: '#fee2e2', borderColor: '#ef4444' }}>
                            <span className="tip-icon">⚠️</span>
                            <span className="tip-text" style={{ color: '#b91c1c' }}>
                                <strong>Warning:</strong> This image may not produce accurate results.
                                Please use a photo with visible seeds on a plain background.
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
                        <button
                            className={`btn-blue ${imageQuality && !imageQuality.overallOk ? 'btn-disabled' : ''}`}
                            onClick={analyzeImage}
                            disabled={imageQuality && !imageQuality.overallOk}
                            style={imageQuality && !imageQuality.overallOk ? { opacity: 0.5, cursor: 'not-allowed' } : {}}
                        >
                            {imageQuality && !imageQuality.overallOk ? 'Invalid Image' : 'Analyze Seeds'}
                        </button>
                        <button className="btn-secondary" onClick={removeFile}>
                            Retake Photo
                        </button>
                    </>
                ) : (
                    <button className="btn-secondary" onClick={() => setScreen('instructions')}>
                        ← Back to Instructions
                    </button>
                )}
            </div>
        </div>
    );

    // Render Loading Screen
    const renderLoading = () => (
        <div className="loading-section">
            <div className="loading-spinner"></div>
            <p className="loading-text">Analyzing your seeds...</p>
            <p className="loading-subtext">Detecting and classifying each seed</p>
        </div>
    );

    // Render Results Screen
    const renderResults = () => (
        <div className="results-section">
            {/* Verdict Card */}
            <div className={`verdict-card ${results.verdict}`}>
                <div className="verdict-icon">{getVerdictIcon(results.verdict)}</div>
                <div className="verdict-message">{results.verdict_message}</div>
            </div>

            {/* Statistics Grid */}
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

            {/* Annotated Image */}
            {results.annotated_image_base64 && (
                <div className="annotated-image-section">
                    <div className="section-header">
                        🖼️ Analyzed Image
                    </div>
                    <div className="annotated-image-container">
                        <img
                            src={`data:image/jpeg;base64,${results.annotated_image_base64}`}
                            alt="Analyzed seeds with annotations"
                            className="annotated-image"
                        />
                    </div>
                    <div className="legend">
                        <div className="legend-item">
                            <span className="legend-color real"></span>
                            <span>Real</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-color fake"></span>
                            <span>Fake</span>
                        </div>
                        <div className="legend-item">
                            <span className="legend-color low-quality"></span>
                            <span>Low Quality</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Tip Box */}
            <div className="tip-box">
                <span className="tip-icon">💡</span>
                <span className="tip-text">
                    <strong>Tip:</strong> Seeds with unusual color, texture, or shape may be flagged as fake or low quality.
                </span>
            </div>

            {/* Actions */}
            <div className="actions">
                <button className="btn-primary" onClick={startOver}>
                    📸 Analyze Another Image
                </button>
                {results.annotated_image_base64 && (
                    <button className="btn-secondary" onClick={downloadImage}>
                        💾 Download Result
                    </button>
                )}
            </div>
        </div>
    );

    return (
        <div className="app">
            {/* Header */}
            <header className="header">
                <h1 className="header-title">SeedShield</h1>
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
