import fs from 'fs';

const html = fs.readFileSync('legacy_index.html', 'utf8');
const start = html.indexOf('<div class="tab-panel docs-panel" id="tab-docs" data-tab="docs">') + '<div class="tab-panel docs-panel" id="tab-docs" data-tab="docs">'.length;
const end = html.indexOf('<!-- Right TOC -->') + html.substring(html.indexOf('<!-- Right TOC -->')).indexOf('</div>') + 6;

let docsHtml = html.substring(start, end);

// Escape { and } before replacing JSX attributes
docsHtml = docsHtml.replace(/\{/g, '{"{"}').replace(/\}/g, '{"}"}');

// Conversion to JSX
docsHtml = docsHtml
  .replace(/class=/g, 'className=')
  .replace(/stroke-width=/g, 'strokeWidth=')
  .replace(/mix-blend-mode=/g, 'mixBlendMode=')
  .replace(/onclick="copyCode\(this\)"/g, 'onClick={copyCode}')
  .replace(/<hr className="section-divider">/g, '<hr className="section-divider" />')
  .replace(/<br>/g, '<br />')
  .replace(/<input(.*?)>/g, '<input$1 />')
  .replace(/<img(.*?)>/g, '<img$1 />');

// Function for sidebar links
docsHtml = docsHtml.replace(/<button className="nav-link(.*?)" data-section="(.*?)"(.*?)>/g, (match, p1, p2, p3) => {
  return `<button className={\`nav-link \${activeSection === '${p2}' ? 'active' : ''}\`} onClick={() => scrollToSection('${p2}')}${p3}>`;
});

// Function for TOC links
docsHtml = docsHtml.replace(/<button className="toc-link(.*?)" data-section="(.*?)"(.*?)>/g, (match, p1, p2, p3) => {
  return `<button className={\`toc-link \${activeSection === '${p2}' ? 'active' : ''}\`} onClick={() => scrollToSection('${p2}')}${p3}>`;
});


const component = `import React, { useEffect, useRef, useState } from 'react';
import '../styles/api-styles.css';

export default function ApiDocs() {
  const [activeSection, setActiveSection] = useState('introduction');
  const docsRef = useRef(null);

  useEffect(() => {
    if (!docsRef.current) return;
    const sections = docsRef.current.querySelectorAll('.doc-section');
    const revealObserver = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
        }
      });
    }, { threshold: 0.08 });

    const spyObserver = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          setActiveSection(entry.target.id);
        }
      });
    }, { rootMargin: '-20% 0px -75% 0px' });

    sections.forEach(s => {
      revealObserver.observe(s);
      if (s.id) spyObserver.observe(s);
    });

    const fills = docsRef.current.querySelectorAll('.benchmark-fill');
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

    fills.forEach(f => barObserver.observe(f));

    return () => {
      revealObserver.disconnect();
      spyObserver.disconnect();
      barObserver.disconnect();
    };
  }, []);

  const scrollToSection = (id) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const copyCode = (e) => {
    const btn = e.currentTarget;
    const block = btn.closest('.code-block');
    const code = block.querySelector('pre code') || block.querySelector('pre');
    navigator.clipboard.writeText(code.textContent).then(() => {
      btn.classList.add('copied');
      const orig = btn.innerHTML;
      btn.innerHTML = \`<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M4 8.5l2.5 2.5L12 5"/></svg> Copied!\`;
      setTimeout(() => { btn.classList.remove('copied'); btn.innerHTML = orig; }, 1800);
    });
  };

  return (
    <div className="docs-panel active" ref={docsRef} style={{ paddingTop: '60px', minHeight: '100vh', display: 'flex' }}>
      \${docsHtml}
    </div>
  );
}
`;

fs.writeFileSync('src/pages/ApiDocs.jsx', component);
console.log('Successfully generated ApiDocs.jsx');
