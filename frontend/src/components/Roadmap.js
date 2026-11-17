import React from 'react';
import { motion } from 'framer-motion';

export default function Roadmap() {
  const roadmapItems = [
    {
      quarter: 'Near Term',
      title: 'Real-Photo Evaluation Set',
      desc: 'Collaborate with local agricultural extension offices to collect verified genuine and seized counterfeit seed samples for an empirical validation benchmark.',
    },
    {
      quarter: 'Near Term',
      title: 'SHAP & Feature Importance Explainability',
      desc: 'Expose local TreeSHAP explanations for the Random Forest classifier directly in the per-seed inspection panel to show farmers exactly which features triggered a fake flag.',
    },
    {
      quarter: 'Mid Term',
      title: 'One-Class Anomaly Scoring',
      desc: 'Train autoencoders or isolation forests strictly on certified genuine seeds, framing counterfeit detection as out-of-distribution anomaly detection rather than multi-class classification.',
    },
    {
      quarter: 'Mid Term',
      title: 'Fully Offline PWA via ONNX Runtime Web',
      desc: 'Port the OpenCV contour extraction to WebAssembly and run scikit-learn models client-side with ONNX Runtime Web so scans run with zero internet connection.',
    },
    {
      quarter: 'Long Term',
      title: 'Vision-Language Explanations',
      desc: 'Incorporate compact on-device vision-language models to produce localized natural-language summaries for agricultural extension workers.',
    },
  ];

  return (
    <section className="section-container" id="roadmap">
      <div className="section-header-block">
        <span className="section-eyebrow">Future Trajectory</span>
        <h2 className="section-title">Development Roadmap</h2>
        <p className="section-subtitle">
          Planned enhancements moving SeedShield from a working prototype toward certified field deployment.
        </p>
      </div>

      <div className="roadmap-timeline">
        {roadmapItems.map((item, idx) => (
          <motion.div
            key={item.title}
            className="roadmap-card"
            initial={{ opacity: 0, x: -15 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.28, delay: idx * 0.06 }}
          >
            <div className="roadmap-meta">
              <span className="roadmap-phase">{item.quarter}</span>
            </div>
            <h3 className="roadmap-title">{item.title}</h3>
            <p className="roadmap-desc">{item.desc}</p>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
