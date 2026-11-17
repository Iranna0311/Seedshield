import React, { useRef, useEffect, useState, useMemo } from 'react';
import * as THREE from 'three';

export default function Hero3D({ theme = 'light', onScanSample, onUploadOwn }) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [webGLFailed, setWebGLFailed] = useState(false);

  // Check prefers-reduced-motion
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }, []);

  useEffect(() => {
    if (prefersReducedMotion) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    let renderer, scene, camera, animationFrameId;
    let isVisible = true;

    try {
      // 1. Setup Renderer
      renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: true,
        antialias: true,
        powerPreference: 'high-performance',
      });
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      renderer.setPixelRatio(dpr);
      const width = canvas.parentElement?.clientWidth || window.innerWidth;
      const height = canvas.parentElement?.clientHeight || window.innerHeight;
      renderer.setSize(width, height);

      // 2. Setup Scene & Camera
      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 100);
      camera.position.set(0, 0, 11);

      // 3. Lighting
      const ambient = new THREE.AmbientLight(0xffffff, theme === 'dark' ? 0.9 : 1.3);
      scene.add(ambient);

      const dirLight = new THREE.DirectionalLight(0xffffff, 1.4);
      dirLight.position.set(8, 12, 10);
      scene.add(dirLight);

      const greenGlow = new THREE.PointLight(0x10b981, 1.2, 25);
      greenGlow.position.set(-6, -4, 4);
      scene.add(greenGlow);

      // 4. Instanced Seeds
      const isMobile = window.innerWidth < 768;
      const count = isMobile ? 45 : 100;
      const geo = new THREE.SphereGeometry(1, 20, 20);
      const mat = new THREE.MeshStandardMaterial({
        roughness: 0.35,
        metalness: 0.15,
      });

      const instancedMesh = new THREE.InstancedMesh(geo, mat, count);

      const colors = [
        new THREE.Color('#10b981'), // Real (Green) ~60%
        new THREE.Color('#10b981'),
        new THREE.Color('#10b981'),
        new THREE.Color('#f59e0b'), // Low Quality (Amber) ~25%
        new THREE.Color('#ef4444'), // Fake (Red) ~15%
      ];

      const seedData = [];
      const dummy = new THREE.Object3D();

      for (let i = 0; i < count; i++) {
        const x = (Math.random() - 0.5) * 16;
        const y = (Math.random() - 0.5) * 10;
        const z = (Math.random() - 0.5) * 10;

        const scaleX = 0.35 + Math.random() * 0.25;
        const scaleY = 0.6 + Math.random() * 0.45;
        const scaleZ = 0.35 + Math.random() * 0.25;

        dummy.position.set(x, y, z);
        dummy.scale.set(scaleX, scaleY, scaleZ);
        dummy.rotation.set(Math.random() * Math.PI, Math.random() * Math.PI, Math.random() * Math.PI);
        dummy.updateMatrix();

        instancedMesh.setMatrixAt(i, dummy.matrix);
        const col = colors[i % colors.length];
        instancedMesh.setColorAt(i, col);

        seedData.push({
          pos: new THREE.Vector3(x, y, z),
          origY: y,
          scale: new THREE.Vector3(scaleX, scaleY, scaleZ),
          rot: new THREE.Euler(dummy.rotation.x, dummy.rotation.y, dummy.rotation.z),
          rotSpeed: {
            x: (Math.random() - 0.5) * 0.015,
            y: (Math.random() - 0.5) * 0.02,
            z: (Math.random() - 0.5) * 0.015,
          },
          baseColor: col.clone(),
          phase: Math.random() * Math.PI * 2,
        });
      }
      instancedMesh.instanceMatrix.needsUpdate = true;
      if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;
      scene.add(instancedMesh);

      // 5. Glowing Scan Plane
      const planeGeo = new THREE.PlaneGeometry(22, 16);
      const planeMat = new THREE.MeshBasicMaterial({
        color: 0x10b981,
        transparent: true,
        opacity: 0.1,
        side: THREE.DoubleSide,
      });
      const scanPlane = new THREE.Mesh(planeGeo, planeMat);
      scene.add(scanPlane);

      // 6. Mouse Parallax
      let mouseX = 0;
      let mouseY = 0;
      const onMouseMove = (e) => {
        const x = (e.clientX / window.innerWidth) * 2 - 1;
        const y = -(e.clientY / window.innerHeight) * 2 + 1;
        mouseX = x * 1.2;
        mouseY = y * 0.8;
      };
      window.addEventListener('mousemove', onMouseMove, { passive: true });

      // 7. Resize Handler
      const onResize = () => {
        if (!canvas.parentElement) return;
        const w = canvas.parentElement.clientWidth;
        const h = canvas.parentElement.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
      };
      window.addEventListener('resize', onResize);

      // 8. Intersection Observer (pause when off-screen)
      let observer;
      if (typeof IntersectionObserver !== 'undefined' && containerRef.current) {
        observer = new IntersectionObserver(([entry]) => {
          isVisible = entry.isIntersecting;
        }, { threshold: 0.05 });
        observer.observe(containerRef.current);
      }

      // 9. Animation Loop
      let clock = new THREE.Clock();
      const highlightColor = new THREE.Color('#ffffff');
      const tempColor = new THREE.Color();

      const animate = () => {
        animationFrameId = requestAnimationFrame(animate);
        if (!isVisible) return;

        const t = clock.getElapsedTime();

        // Scan plane sweeps
        const planeZ = Math.sin(t * 0.75) * 5.5;
        scanPlane.position.z = planeZ;

        // Animate seeds
        for (let i = 0; i < count; i++) {
          const s = seedData[i];
          s.rot.x += s.rotSpeed.x;
          s.rot.y += s.rotSpeed.y;
          s.rot.z += s.rotSpeed.z;
          s.pos.y = s.origY + Math.sin(t * 1.1 + s.phase) * 0.3;

          dummy.position.copy(s.pos);
          dummy.rotation.copy(s.rot);
          dummy.scale.copy(s.scale);
          dummy.updateMatrix();
          instancedMesh.setMatrixAt(i, dummy.matrix);

          const dist = Math.abs(s.pos.z - planeZ);
          if (dist < 1.1) {
            tempColor.copy(s.baseColor).lerp(highlightColor, (1 - dist / 1.1) * 0.65);
            instancedMesh.setColorAt(i, tempColor);
          } else {
            instancedMesh.setColorAt(i, s.baseColor);
          }
        }
        instancedMesh.instanceMatrix.needsUpdate = true;
        if (instancedMesh.instanceColor) instancedMesh.instanceColor.needsUpdate = true;

        // Camera lerp
        camera.position.x += (mouseX - camera.position.x) * 0.05;
        camera.position.y += (mouseY - camera.position.y) * 0.05;
        camera.lookAt(0, 0, 0);

        renderer.render(scene, camera);
      };

      animate();

      // Cleanup
      return () => {
        window.removeEventListener('mousemove', onMouseMove);
        window.removeEventListener('resize', onResize);
        if (observer) observer.disconnect();
        cancelAnimationFrame(animationFrameId);
        geo.dispose();
        mat.dispose();
        planeGeo.dispose();
        planeMat.dispose();
        renderer.dispose();
      };
    } catch (err) {
      console.warn('WebGL init error, using fallback:', err);
      setWebGLFailed(true);
    }
  }, [theme, prefersReducedMotion]);

  const fallbackGradient = (
    <div className="hero-fallback-bg" aria-hidden="true">
      <div className="gradient-orb orb-1"></div>
      <div className="gradient-orb orb-2"></div>
      <div className="scan-line-sweep"></div>
    </div>
  );

  return (
    <section className="hero-section" id="top" ref={containerRef}>
      {/* 3D Background or Fallback */}
      <div className="hero-canvas-container" aria-hidden="true">
        {webGLFailed || prefersReducedMotion ? (
          fallbackGradient
        ) : (
          <canvas ref={canvasRef} style={{ width: '100%', height: '100%', display: 'block' }} />
        )}
      </div>

      {/* Hero Content Overlay */}
      <div className="hero-content">
        <div className="hero-badge">
          <span className="badge-pulse"></span>
          <span>OpenCV + scikit-learn Hybrid Screening</span>
        </div>

        <h1 className="hero-headline">
          Visual Seed Quality Screening for Farmers
        </h1>

        <p className="hero-description">
          Detect authentic, damaged, and suspected counterfeit seeds directly from a smartphone photo.
          Zero GPU required, 95% Wilson confidence intervals, and automated quality checks.
        </p>

        <div className="hero-actions">
          <button className="btn-hero-primary" onClick={onScanSample}>
            🌱 Try a Sample Scan
          </button>
          <button className="btn-hero-secondary" onClick={onUploadOwn}>
            📸 Upload Seed Photo
          </button>
        </div>

        {/* Key construction stats */}
        <div className="hero-stats-row">
          <div className="hero-stat-card">
            <span className="hero-stat-num">4</span>
            <span className="hero-stat-desc">Classes (Real, Fake, Low, Uncertain)</span>
          </div>
          <div className="hero-stat-card">
            <span className="hero-stat-num">20</span>
            <span className="hero-stat-desc">Agronomic Morphological Features</span>
          </div>
          <div className="hero-stat-card">
            <span className="hero-stat-num">95%</span>
            <span className="hero-stat-desc">Wilson Score Confidence Interval</span>
          </div>
          <div className="hero-stat-card">
            <span className="hero-stat-num">8 MB</span>
            <span className="hero-stat-desc">Hardened Upload Payload Limit</span>
          </div>
        </div>
      </div>
    </section>
  );
}
