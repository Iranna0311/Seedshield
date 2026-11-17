import React from 'react';
import { motion } from 'framer-motion';

export default function HowItWorks() {
  const steps = [
    {
      num: '01',
      icon: '📸',
      title: 'Capture & Content Validation',
      desc: 'Photograph 3-10 seeds on a plain, neutral background. The client and backend run instant content checks (brightness, contrast, and color-space analysis) to reject non-seed photos before analysis begins.',
    },
    {
      num: '02',
      icon: '🔬',
      title: 'OpenCV Contour Segmentation',
      desc: 'Bilateral filtering, Otsu thresholding, and morphological operations isolate individual grain contours. 20 agronomic features (circularity, aspect ratio, solidity, color uniformity) are extracted from each seed crop.',
    },
    {
      num: '03',
      icon: '⚖️',
      title: 'Classification & Wilson CI',
      desc: 'A Random Forest model classifies each seed as REAL, FAKE, or LOW QUALITY. Ambiguous seeds (max probability < 0.6) are flagged UNCERTAIN. A 95% Wilson score confidence interval is computed over the batch.',
    },
  ];

  return (
    <section className="section-container" id="how-it-works">
      <div className="section-header-block">
        <span className="section-eyebrow">Pipeline Architecture</span>
        <h2 className="section-title">How SeedShield Works</h2>
        <p className="section-subtitle">
          A robust two-stage screening system designed to run on resource-constrained devices without cloud GPU dependencies.
        </p>
      </div>

      <div className="steps-grid">
        {steps.map((s, idx) => (
          <motion.div
            key={s.num}
            className="step-card"
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.28, delay: idx * 0.08 }}
          >
            <div className="step-header">
              <span className="step-badge">{s.num}</span>
              <span className="step-icon">{s.icon}</span>
            </div>
            <h3 className="step-title">{s.title}</h3>
            <p className="step-desc">{s.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
