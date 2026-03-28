/* ═══════════════════════════════════════════════════════════
   LatentMesh Documentation — Client‑Side Logic
   Tab switching, scrollspy, smooth scroll, code copy, canvas, theme
   ═══════════════════════════════════════════════════════════ */

/* ── Theme Toggle ───────────────────────────────────────── */
(function initTheme() {
  const saved = localStorage.getItem('ds-theme');
  if (saved) {
    document.documentElement.setAttribute('data-theme', saved);
  } else if (window.matchMedia('(prefers-color-scheme: light)').matches) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();

document.addEventListener('DOMContentLoaded', () => {
  const toggle = document.getElementById('theme-toggle');
  if (toggle) {
    toggle.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      const next = current === 'light' ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('ds-theme', next);
    });
  }
});

/* ── Tab System ─────────────────────────────────────────── */
function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll('.header-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === tabName);
  });

  // Update tab panels
  document.querySelectorAll('.tab-panel').forEach(panel => {
    panel.classList.toggle('active', panel.dataset.tab === tabName);
  });

  // Show/hide docs-specific elements
  const sidebar = document.getElementById('sidebar');
  const tocPanel = document.getElementById('toc-panel');
  const aurora = document.getElementById('home-aurora');
  if (tabName === 'docs') {
    if (sidebar) sidebar.style.display = '';
    if (tocPanel) tocPanel.style.display = '';
    if (aurora) aurora.classList.remove('visible');
    // Re-trigger section visibility in docs
    revealObserver && observeSections();
    window.scrollTo(0, 0);
  } else {
    if (sidebar) sidebar.style.display = 'none';
    if (tocPanel) tocPanel.style.display = 'none';
    if (aurora) aurora.classList.add('visible');
    window.scrollTo(0, 0);
  }
}

// Wire header tab buttons
document.querySelectorAll('.header-tab').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// Initially hide sidebar/TOC since home is default, show aurora
document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('sidebar');
  const tocPanel = document.getElementById('toc-panel');
  const aurora = document.getElementById('home-aurora');
  if (sidebar) sidebar.style.display = 'none';
  if (tocPanel) tocPanel.style.display = 'none';
  // Fade in aurora after a brief delay
  setTimeout(() => { if (aurora) aurora.classList.add('visible'); }, 300);
});

/* ── Background Grid ────────────────────────────────────── */
(function initGrid() {
  const c = document.getElementById('bg-canvas');
  if (!c) return;
  const ctx = c.getContext('2d');
  let w, h, cols, rows;
  const gap = 48;

  function resize() {
    w = c.width  = window.innerWidth;
    h = c.height = window.innerHeight;
    cols = Math.ceil(w / gap) + 1;
    rows = Math.ceil(h / gap) + 1;
  }
  resize();
  window.addEventListener('resize', resize);

  function draw(t) {
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(108,138,255,0.04)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i < cols; i++) {
      const x = i * gap;
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let j = 0; j < rows; j++) {
      const y = j * gap + Math.sin(t / 4000 + j * 0.3) * 1.5;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    requestAnimationFrame(draw);
  }
  requestAnimationFrame(draw);
  requestAnimationFrame(() => c.classList.add('visible'));
})();

/* ── Section Reveal ─────────────────────────────────────── */
let revealObserver;
function observeSections() {
  if (revealObserver) revealObserver.disconnect();
  revealObserver = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
  }, { threshold: 0.08 });
  document.querySelectorAll('.doc-section').forEach(s => revealObserver.observe(s));
}
observeSections();

/* ── Scrollspy ──────────────────────────────────────────── */
const spyObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const id = entry.target.id;
    document.querySelectorAll('.nav-link').forEach(link => {
      link.classList.toggle('active', link.dataset.section === id);
    });
    document.querySelectorAll('.toc-link').forEach(link => {
      link.classList.toggle('active', link.dataset.section === id);
    });
  });
}, { rootMargin: '-20% 0px -75% 0px' });

document.querySelectorAll('.doc-section[id]').forEach(s => spyObserver.observe(s));

/* ── Smooth Scroll (sidebar & TOC links) ────────────────── */
function scrollToSection(id) {
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.querySelectorAll('.nav-link[data-section], .toc-link[data-section]').forEach(link => {
  link.addEventListener('click', () => {
    // If not on docs tab, switch first
    const docsPanel = document.getElementById('tab-docs');
    if (!docsPanel.classList.contains('active')) {
      switchTab('docs');
    }
    scrollToSection(link.dataset.section);
    // Close mobile sidebar
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('active');
  });
});

/* ── Code Copy ──────────────────────────────────────────── */
function copyCode(btn) {
  const block = btn.closest('.code-block');
  const code = block.querySelector('pre code') || block.querySelector('pre');
  navigator.clipboard.writeText(code.textContent).then(() => {
    btn.classList.add('copied');
    const orig = btn.innerHTML;
    btn.innerHTML = `<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 8.5l2.5 2.5L12 5"/></svg> Copied!`;
    setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = orig; }, 1800);
  });
}

/* ── Mobile Menu ────────────────────────────────────────── */
const menuBtn = document.getElementById('mobile-menu-toggle');
const sidebar = document.getElementById('sidebar');
const overlay = document.getElementById('sidebar-overlay');

if (menuBtn && sidebar) {
  menuBtn.addEventListener('click', () => {
    // If on home tab, switch to docs first
    const docsPanel = document.getElementById('tab-docs');
    if (!docsPanel.classList.contains('active')) {
      switchTab('docs');
    }
    sidebar.style.display = '';
    sidebar.classList.toggle('open');
    overlay && overlay.classList.toggle('active');
  });
}
if (overlay) {
  overlay.addEventListener('click', () => {
    sidebar && sidebar.classList.remove('open');
    overlay.classList.remove('active');
  });
}

/* ── Benchmark Bar Animation ────────────────────────────── */
const barObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      const fill = e.target;
      fill.style.width = fill.dataset.target + '%';
      fill.classList.add('animated');
      barObserver.unobserve(fill);
    }
  });
}, { threshold: 0.3 });

document.querySelectorAll('.benchmark-fill[data-target]').forEach(f => barObserver.observe(f));
