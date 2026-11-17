import React from 'react';
import { motion } from 'framer-motion';

export default function Limitations() {
  const points = [
    {
      label: 'Synthetic Training Distribution',
      text: 'The FAKE seed class was generated synthetically by perturbing morphological distributions (hue shifts, extreme aspect ratios, artificial surface roughness). While grounded in agronomic observation, these models have not yet been evaluated on genuine commercial counterfeits.',
    },
    {
      label: 'Lighting & Surface Dependency',
      text: 'Contour detection requires reasonable contrast against a plain background (white paper, neutral tabletop, or cupped hand). Heavy glare, deep shadows, or patterned backgrounds can interfere with seed boundary separation.',
    },
    {
      label: 'Morphology Profile Specificity',
      text: 'The current feature extractor is calibrated primarily on oval/grain seed morphology (soybeans, legumes, pulses). Highly irregular seeds (e.g. winged carrot seeds or tiny sesame) may require dedicated contour calibrations.',
    },
    {
      label: 'Screening Tool, Not Germination Test',
      text: 'SeedShield is a rapid non-destructive physical screening tool to spot obvious adulteration or damage. It does not replace certified biochemical germination or genetic purity testing.',
    },
  ];

  return (
    <section className="section-container" id="limitations">
      <div className="section-header-block">
        <span className="section-eyebrow">Honest Engineering</span>
        <h2 className="section-title">Known Limitations</h2>
        <p className="section-subtitle">
          Transparent appraisal of current boundaries — what this prototype can and cannot do in practical deployment.
        </p>
      </div>

      <div className="limitations-grid">
        {points.map((p, idx) => (
          <motion.div
            key={p.label}
            className="limitation-card"
            initial={{ opacity: 0, y: 15 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.28, delay: idx * 0.06 }}
          >
            <div className="limitation-header">
              <span className="limitation-badge">Notice</span>
              <h3 className="limitation-title">{p.label}</h3>
            </div>
            <p className="limitation-text">{p.text}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
