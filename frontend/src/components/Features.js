import React from 'react';
import { motion } from 'framer-motion';

export default function Features() {
  const features = [
    {
      icon: '🛡️',
      title: 'Uncertainty Abstention',
      desc: 'Instead of forcing noisy predictions into arbitrary labels, any crop with maximum confidence below 0.6 is categorized as UNCERTAIN and excluded from the fake-rate numerator.',
    },
    {
      icon: '📊',
      title: 'Wilson 95% Confidence Interval',
      desc: 'Small sample batches (e.g. 5–15 seeds) carry statistical variance. SeedShield computes the Wilson score interval [low, high] to provide farmers an honest interval for batch risk.',
    },
    {
      icon: '🔍',
      title: 'Per-Seed Interactive Inspection',
      desc: 'Every detection is normalized and outlined with SVG bounding boxes. Click any seed in the result to view individual crop coordinates, confidence ratings, and class probability distributions.',
    },
    {
      icon: '🔒',
      title: 'Upload Hardening & Hygiene',
      desc: 'Strict 8 MB payload caps, PIL decompression bomb guards (25M pixel max), EXIF orientation transposition, and automated content validation to reject non-seed uploads.',
    },
    {
      icon: '⚡',
      title: 'Zero GPU Requirement',
      desc: 'Classical CV segmentation paired with scikit-learn Random Forests means the entire inference engine executes in milliseconds on standard CPU hardware or low-power servers.',
    },
    {
      icon: '📱',
      title: 'Field-Ready & Mobile First',
      desc: 'Engineered for farmers in the field. Responsive touch targets (≥ 44px), mobile bottom sheet inspector, high-contrast dark/light mode, and full offline report exports.',
    },
  ];

  return (
    <section className="section-container" id="features">
      <div className="section-header-block">
        <span className="section-eyebrow">Key Capabilities</span>
        <h2 className="section-title">Designed for Agronomic Reality</h2>
        <p className="section-subtitle">
          Engineered to prevent false alarms, protect against counterfeit batches, and provide transparent statistical estimates.
        </p>
      </div>

      <div className="features-grid">
        {features.map((f, idx) => (
          <motion.div
            key={f.title}
            className="feature-card"
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.28, delay: idx * 0.05 }}
          >
            <div className="feature-icon">{f.icon}</div>
            <h3 className="feature-title">{f.title}</h3>
            <p className="feature-desc">{f.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
