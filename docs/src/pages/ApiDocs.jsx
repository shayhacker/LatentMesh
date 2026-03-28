import React, { useEffect, useRef, useState } from 'react';
import '../styles/api-styles.css';

/* ── SVG nav icons ─────────────────────────────────────────── */
const I = {
  intro: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="6" /><path d="M8 5v3l2 2" /></svg>,
  install: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 2v8m0 0l-3-3m3 3l3-3M3 13h10" /></svg>,
  quickstart: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M5 3l7 5-7 5V3z" /></svg>,
  latentllm: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="2" y="2" width="12" height="12" rx="2" /><path d="M5 6h6M5 10h4" /></svg>,
  primitives: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="5" cy="8" r="2.5" /><circle cx="11" cy="5" r="2.5" /><circle cx="11" cy="11" r="2.5" /></svg>,
  graphstate: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="4" cy="4" r="2" /><circle cx="12" cy="4" r="2" /><circle cx="8" cy="12" r="2" /><path d="M5.5 5.5L7 10.5M10.5 5.5L9 10.5" /></svg>,
  sequential: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M3 4h10M3 8h7M3 12h9" /></svg>,
  voting: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M8 2l2 4 4 1-3 3 1 4-4-2-4 2 1-4-3-3 4-1z" /></svg>,
  hierarchical: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="3" r="2" /><circle cx="4" cy="11" r="2" /><circle cx="12" cy="11" r="2" /><path d="M8 5v2M6 9l-1 1M10 9l1 1" /></svg>,
  deployment: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="8" cy="8" r="6" /><path d="M8 5v6M5 8h6" /></svg>,
  debug: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M4 2v4M12 2v4M2 8h12M4 12v2M12 12v2M6 8v4M10 8v4" /></svg>,
  benchmarks: <svg className="nav-icon" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5"><rect x="1" y="9" width="3" height="5" /><rect x="6.5" y="5" width="3" height="9" /><rect x="12" y="2" width="3" height="12" /></svg>,
};

/* ── Copy button ───────────────────────────────────────────── */
function CopyBtn() {
  const [copied, setCopied] = useState(false);
  return (
    <button className={`copy-btn${copied ? ' copied' : ''}`}
      onClick={(e) => {
        const block = e.currentTarget.closest('.code-block');
        const code = block.querySelector('pre code') || block.querySelector('pre');
        navigator.clipboard.writeText(code.textContent).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1800);
        });
      }}>
      <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" width={11} height={11}>
        {copied
          ? <path d="M4 8.5l2.5 2.5L12 5" />
          : <><rect x="5" y="5" width="9" height="9" rx="1.5" /><path d="M5 11H3.5A1.5 1.5 0 012 9.5v-7A1.5 1.5 0 013.5 1h7A1.5 1.5 0 0112 2.5V5" /></>}
      </svg>
      {copied ? 'Copied!' : 'Copy'}
    </button>
  );
}

/* ── Syntax-highlighted Python block ─────────────────────── */
function Py({ children }) {
  return <pre><code>{children}</code></pre>;
}

/* ── Main component ─────────────────────────────────────────── */
export default function ApiDocs() {
  const [activeSection, setActiveSection] = useState('introduction');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const docsRef = useRef(null);

  useEffect(() => {
    if (!docsRef.current) return;

    const revealObs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) e.target.classList.add('visible'); });
    }, { threshold: 0.06 });

    const spyObs = new IntersectionObserver(entries => {
      entries.forEach(e => { if (e.isIntersecting) setActiveSection(e.target.id); });
    }, { rootMargin: '-15% 0px -75% 0px' });

    docsRef.current.querySelectorAll('.doc-section').forEach(s => {
      revealObs.observe(s);
      if (s.id) spyObs.observe(s);
    });

    const barObs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          const fill = e.target;
          fill.style.width = fill.dataset.target + '%';
          barObs.unobserve(fill);
        }
      });
    }, { threshold: 0.3 });

    docsRef.current.querySelectorAll('.benchmark-fill[data-target]').forEach(f => barObs.observe(f));

    return () => { revealObs.disconnect(); spyObs.disconnect(); barObs.disconnect(); };
  }, []);

  const scrollTo = (id) => {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    setSidebarOpen(false);
  };

  const nl = (section) => `nav-link${activeSection === section ? ' active' : ''}`;

  return (
    <div className="docs-root" ref={docsRef}>

      {/* ── Sidebar ─────────────────────────────────────────── */}
      <nav className={`docs-sidebar${sidebarOpen ? ' open' : ''}`}>
        <div className="nav-section">
          <div className="nav-section-title">Getting Started</div>
          <button className={nl('introduction')} onClick={() => scrollTo('introduction')}>{I.intro} Introduction</button>
          <button className={nl('installation')} onClick={() => scrollTo('installation')}>{I.install} Installation</button>
          <button className={nl('quick-start')} onClick={() => scrollTo('quick-start')}>{I.quickstart} Quick Start</button>
        </div>
        <div className="nav-section">
          <div className="nav-section-title">API Reference</div>
          <button className={nl('latentllm')} onClick={() => scrollTo('latentllm')}>{I.latentllm} LatentLLM</button>
          <button className={nl('primitives')} onClick={() => scrollTo('primitives')}>{I.primitives} Primitives</button>
          <button className={nl('graph-state')} onClick={() => scrollTo('graph-state')}>{I.graphstate} Graph State</button>
        </div>
        <div className="nav-section">
          <div className="nav-section-title">Examples</div>
          <button className={nl('sequential')} onClick={() => scrollTo('sequential')}>{I.sequential} Sequential Pipeline</button>
          <button className={nl('voting')} onClick={() => scrollTo('voting')}>{I.voting} Consensus Voting</button>
          <button className={nl('hierarchical')} onClick={() => scrollTo('hierarchical')}>{I.hierarchical} Hierarchical Routing</button>
          <button className={nl('deployment')} onClick={() => scrollTo('deployment')}>{I.deployment} FastAPI Deployment</button>
          <button className={nl('debug-logging')} onClick={() => scrollTo('debug-logging')}>{I.debug} Debug &amp; Logging</button>
        </div>
        <div className="nav-section">
          <div className="nav-section-title">Performance</div>
          <button className={nl('benchmarks')} onClick={() => scrollTo('benchmarks')}>{I.benchmarks} Benchmarks</button>
        </div>
      </nav>

      {/* ── Overlay (mobile) ────────────────────────────────── */}
      <div className={`sidebar-overlay${sidebarOpen ? ' active' : ''}`} onClick={() => setSidebarOpen(false)} />

      {/* ── Main content ────────────────────────────────────── */}
      <main className="docs-main">

        {/* Introduction */}
        <section className="doc-section visible" id="introduction">
          <h1>LatentMesh</h1>
          <p className="section-subtitle">
            Multi-agent KV-cache communication for LLMs. Wire multiple agents
            together so downstream agents <strong>skip re-encoding context</strong> that
            upstream agents already processed.
          </p>
        </section>

        {/* Installation */}
        <section className="doc-section" id="installation">
          <h2>Installation</h2>
          <p>Install from PyPI:</p>
          <div className="code-block">
            <div className="code-block-header"><span>bash</span><CopyBtn /></div>
            <Py>pip install latentmesh</Py>
          </div>
          <p>For persistent disk-backed caching:</p>
          <div className="code-block">
            <div className="code-block-header"><span>bash</span><CopyBtn /></div>
            <Py>pip install latentmesh[disk]</Py>
          </div>
        </section>

        {/* Quick Start */}
        <section className="doc-section" id="quick-start">
          <h2>Quick Start</h2>
          <p>Build a three-agent pipeline (Plan → Reason → Review):</p>
          <div className="code-block">
            <div className="code-block-header"><span>python</span><CopyBtn /></div>
            <Py>
<span className="kw">from</span> latentmesh <span className="kw">import</span> ({"\n"}
{"    "}LatentLLM, LatentState,{"\n"}
{"    "}PlanPrimitive, ReasonPrimitive, ReviewPrimitive,{"\n"}
){"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> StateGraph, START, END{"\n"}
{"\n"}
<span className="cm"># Set up the KV cache store</span>{"\n"}
store = <span className="fn">MemoryKVStore</span>(){"\n"}
cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
llm   = <span className="fn">LatentLLM</span>({"\n"}
{"    "}<span className="str">"Qwen/Qwen3-0.6B"</span>,{"\n"}
{"    "}device=<span className="str">"cuda"</span>,{"\n"}
{"    "}global_cache=cache,{"\n"}
){"\n"}
{"\n"}
<span className="cm"># Build a LangGraph pipeline</span>{"\n"}
builder = <span className="fn">StateGraph</span>(LatentState){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"plan"</span>,   <span className="fn">PlanPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"reason"</span>, <span className="fn">ReasonPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"review"</span>, <span className="fn">ReviewPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_edge</span>(START, <span className="str">"plan"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"plan"</span>,   <span className="str">"reason"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"reason"</span>, <span className="str">"review"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"review"</span>, END){"\n"}
{"\n"}
graph  = builder.<span className="fn">compile</span>(){"\n"}
result = graph.<span className="fn">invoke</span>({"{"}{"\n"}
{"    "}<span className="str">"messages"</span>: [{"\n"}
{"        "}{"{"}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"         "}<span className="str">"content"</span>: <span className="str">"Prove √2 is irrational"</span>{"}"},
{"\n"}{"    "}],{"\n"}
{"    "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"}"}){"\n"}
{"\n"}
<span className="fn">print</span>(result[<span className="str">"latent"</span>].text)
            </Py>
          </div>
        </section>

        {/* LatentLLM */}
        <section className="doc-section" id="latentllm">
          <h2>LatentLLM</h2>
          <p>Core model wrapper. Loads any HuggingFace causal LM and integrates with <code>GlobalPrefixCache</code> for automatic KV-cache reuse.</p>
          <div className="api-signature">
            <span className="fn">LatentLLM</span>(model_name, device=<span className="str">"cuda"</span>, dtype=<span className="str">"auto"</span>, global_cache=<span className="kw">None</span>, debug=<span className="kw">False</span>)
          </div>
          <dl className="param-list">
            <dt>model_name</dt><dd>HuggingFace model identifier (e.g., <code>"Qwen/Qwen3-0.6B"</code>)</dd>
            <dt>device</dt><dd>Target device (<code>"cuda"</code>, <code>"cpu"</code>)</dd>
            <dt>dtype</dt><dd>Torch dtype. <code>"auto"</code> defaults to <code>float16</code>.</dd>
            <dt>global_cache</dt><dd>A <code>GlobalPrefixCache</code> instance for cross-agent KV reuse.</dd>
            <dt>debug</dt><dd>When <code>True</code>, logs cache hits/misses and token counts.</dd>
          </dl>

          <h3>generate()</h3>
          <div className="api-signature">
            <span className="fn">generate</span>(messages, max_new_tokens=<span className="num">128</span>, temperature=<span className="num">0.6</span>, output_scores=<span className="kw">False</span>) → <span className="cls">AgentOutput</span>
          </div>
          <p>Generates text from a message history. Automatically queries the global prefix cache for KV-cache reuse.</p>
          <dl className="param-list">
            <dt>messages</dt><dd>Chat messages list. Each dict has <code>"role"</code> and <code>"content"</code>.</dd>
            <dt>max_new_tokens</dt><dd>Maximum tokens to generate.</dd>
            <dt>output_scores</dt><dd>If <code>True</code>, computes mean logprob in <code>debug_text</code>.</dd>
          </dl>

          <h3>AgentOutput</h3>
          <table className="doc-table">
            <thead><tr><th>Field</th><th>Type</th><th>Description</th></tr></thead>
            <tbody>
              <tr><td><code>text</code></td><td><code>str</code></td><td>Generated text</td></tr>
              <tr><td><code>tokens</code></td><td><code>int</code></td><td>Number of generated tokens</td></tr>
              <tr><td><code>cached_tokens</code></td><td><code>int</code></td><td>Tokens loaded from prefix cache</td></tr>
              <tr><td><code>input_tokens_uncached</code></td><td><code>int</code></td><td>Tokens freshly encoded (cache miss)</td></tr>
              <tr><td><code>output_tokens</code></td><td><code>int</code></td><td>Tokens produced during generation</td></tr>
              <tr><td><code>debug_text</code></td><td><code>list[str]</code></td><td>Diagnostics (e.g. <code>mean_logprob:-0.5</code>)</td></tr>
            </tbody>
          </table>
        </section>

        {/* Primitives */}
        <section className="doc-section" id="primitives">
          <h2>Primitives</h2>
          <p>All primitives are callable LangGraph nodes (<code>{'state → {"latent": AgentOutput}'}</code>). Each uses a <strong>trigger text</strong> to steer the model's generation.</p>

          <h3>AgentPrimitive (Base)</h3>
          <div className="api-signature"><span className="fn">AgentPrimitive</span>(name, llm, trigger_text=<span className="str">""</span>, max_new_tokens=<span className="num">128</span>)</div>
          <p>Base class. Appends trigger text as a user message, then calls <code>llm.generate()</code>.</p>

          <h3>Built-in Primitives</h3>
          <table className="doc-table">
            <thead><tr><th>Primitive</th><th>Default Trigger</th><th>Purpose</th></tr></thead>
            <tbody>
              <tr><td><code>PlanPrimitive</code></td><td><code>"Break the problem into clear steps…"</code></td><td>Structural decomposition</td></tr>
              <tr><td><code>ReasonPrimitive</code></td><td><code>"Reason through each step carefully…"</code></td><td>Core computation</td></tr>
              <tr><td><code>ReviewPrimitive</code></td><td><code>"Review the reasoning for errors…"</code></td><td>Verification &amp; final answer</td></tr>
            </tbody>
          </table>

          <h3>VotingPrimitive</h3>
          <div className="api-signature"><span className="fn">VotingPrimitive</span>(name, candidates: List[AgentPrimitive])</div>
          <p>Runs multiple candidates on the same input. Selects the winner by <strong>highest mean generation log-probability</strong>.</p>
        </section>

        {/* Graph State */}
        <section className="doc-section" id="graph-state">
          <h2>Graph State</h2>
          <div className="api-signature" style={{ flexDirection: 'column', lineHeight: 1.8 }}>
            <span><span className="kw">class</span> <span className="cls">LatentState</span>(TypedDict):</span>
            <span>&nbsp;&nbsp;messages: Annotated[List[dict], add]</span>
            <span>&nbsp;&nbsp;latent: Annotated[Optional[AgentOutput], latent_reducer]</span>
            <span>&nbsp;&nbsp;tokens_so_far: Annotated[int, add]</span>
          </div>
          <p>LangGraph state schema for LatentMesh workflows. Compatible with <code>StateGraph</code>.</p>

          <h3>latent_reducer</h3>
          <p>Merges <code>AgentOutput</code> across nodes: text is concatenated, token counts are summed, and <code>debug_text</code> lists are combined.</p>
        </section>

        <hr className="section-divider" />

        {/* Sequential Pipeline */}
        <section className="doc-section" id="sequential">
          <h2>Example: Sequential Pipeline</h2>
          <p>Linear pipeline: Plan → Reason → Review. Each agent's KV cache is automatically reused by the next.</p>
          <div className="code-block">
            <div className="code-block-header"><span>python · sequential.py</span><CopyBtn /></div>
            <Py>
<span className="kw">from</span> latentmesh <span className="kw">import</span> ({"\n"}
{"    "}LatentLLM, LatentState, PlanPrimitive,{"\n"}
{"    "}ReasonPrimitive, ReviewPrimitive,{"\n"}
){"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> StateGraph, START, END{"\n"}
{"\n"}
store = <span className="fn">MemoryKVStore</span>(){"\n"}
cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
llm   = <span className="fn">LatentLLM</span>({"\n"}
{"    "}<span className="str">"Qwen/Qwen3-0.6B"</span>,{"\n"}
{"    "}device=<span className="str">"cuda"</span>,{"\n"}
{"    "}global_cache=cache, debug=<span className="kw">True</span>,{"\n"}
){"\n"}
{"\n"}
builder = <span className="fn">StateGraph</span>(LatentState){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"plan"</span>,   <span className="fn">PlanPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"reason"</span>, <span className="fn">ReasonPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"review"</span>, <span className="fn">ReviewPrimitive</span>(llm)){"\n"}
builder.<span className="fn">add_edge</span>(START, <span className="str">"plan"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"plan"</span>,   <span className="str">"reason"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"reason"</span>, <span className="str">"review"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"review"</span>, END){"\n"}
{"\n"}
graph  = builder.<span className="fn">compile</span>(){"\n"}
result = graph.<span className="fn">invoke</span>({"{"}{"\n"}
{"    "}<span className="str">"messages"</span>: [{"{"}{"\n"}
{"        "}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"        "}<span className="str">"content"</span>: <span className="str">"Prove √2 is irrational"</span>,{"\n"}
{"    "}{"}"}],{"\n"}
{"    "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"}"}){"\n"}
{"\n"}
<span className="fn">print</span>(result[<span className="str">"latent"</span>].text){"\n"}
<span className="fn">print</span>(<span className="str">f"Tokens: </span>{"{"}<span className="str">result['tokens_so_far']</span>{"}"}<span className="str">"</span>)
            </Py>
          </div>
        </section>

        {/* Consensus Voting */}
        <section className="doc-section" id="voting">
          <h2>Example: Consensus Voting</h2>
          <p>Run multiple reasoning paths and select the best by generation log-probability.</p>
          <div className="code-block">
            <div className="code-block-header"><span>python · complex.py</span><CopyBtn /></div>
            <Py>
<span className="kw">from</span> latentmesh <span className="kw">import</span> ({"\n"}
{"    "}LatentLLM, LatentState,{"\n"}
{"    "}ReasonPrimitive, VotingPrimitive,{"\n"}
){"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> StateGraph, START, END{"\n"}
{"\n"}
store = <span className="fn">MemoryKVStore</span>(){"\n"}
cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
llm   = <span className="fn">LatentLLM</span>({"\n"}
{"    "}<span className="str">"HuggingFaceTB/SmolLM-135M"</span>,{"\n"}
{"    "}device=<span className="str">"cpu"</span>, global_cache=cache,{"\n"}
){"\n"}
{"\n"}
candidates = [{"\n"}
{"    "}<span className="fn">ReasonPrimitive</span>({"\n"}
{"        "}llm,{"\n"}
{"        "}trigger_text=<span className="str">"Analyze step by step:"</span>,{"\n"}
{"        "}max_new_tokens=<span className="num">24</span>,{"\n"}
{"    "}),{"\n"}
{"    "}<span className="fn">ReasonPrimitive</span>({"\n"}
{"        "}llm,{"\n"}
{"        "}trigger_text=<span className="str">"Think creatively:"</span>,{"\n"}
{"        "}max_new_tokens=<span className="num">24</span>,{"\n"}
{"    "}),{"\n"}
{"    "}<span className="fn">ReasonPrimitive</span>({"\n"}
{"        "}llm,{"\n"}
{"        "}trigger_text=<span className="str">"Take a careful approach:"</span>,{"\n"}
{"        "}max_new_tokens=<span className="num">24</span>,{"\n"}
{"    "}),{"\n"}
]{"\n"}
{"\n"}
consensus = <span className="fn">VotingPrimitive</span>({"\n"}
{"    "}<span className="str">"BestOfThree"</span>, candidates,{"\n"}
){"\n"}
{"\n"}
builder = <span className="fn">StateGraph</span>(LatentState){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"vote"</span>, consensus){"\n"}
builder.<span className="fn">add_edge</span>(START, <span className="str">"vote"</span>){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"vote"</span>, END){"\n"}
{"\n"}
graph  = builder.<span className="fn">compile</span>(){"\n"}
result = graph.<span className="fn">invoke</span>({"{"}{"\n"}
{"    "}<span className="str">"messages"</span>: [{"{"}{"\n"}
{"        "}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"        "}<span className="str">"content"</span>: <span className="str">"Explain quantum entanglement"</span>,{"\n"}
{"    "}{"}"}],{"\n"}
{"    "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"}"}){"\n"}
<span className="fn">print</span>(result[<span className="str">"latent"</span>].text)
            </Py>
          </div>
        </section>

        {/* Hierarchical */}
        <section className="doc-section" id="hierarchical">
          <h2>Example: Hierarchical Routing</h2>
          <p>A supervisor agent routes to a specialist based on the content of its generated text.</p>
          <div className="code-block">
            <div className="code-block-header"><span>python · hierarchical.py</span><CopyBtn /></div>
            <Py>
<span className="kw">from</span> latentmesh <span className="kw">import</span> ({"\n"}
{"    "}LatentState, LatentLLM, AgentPrimitive,{"\n"}
{"    "}ReasonPrimitive,{"\n"}
){"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> END, START, StateGraph{"\n"}
{"\n"}
<span className="cm"># Route based on keywords in generated text</span>{"\n"}
<span className="kw">def</span> <span className="fn">route</span>(state):{"\n"}
{"    "}latent = state.get(<span className="str">"latent"</span>){"\n"}
{"    "}<span className="kw">if</span> latent <span className="kw">is</span> <span className="kw">None</span> <span className="kw">or</span> latent.text <span className="kw">is</span> <span className="kw">None</span>:{"\n"}
{"        "}<span className="kw">return</span> <span className="str">"creative"</span>{"\n"}
{"    "}text = latent.text.lower(){"\n"}
{"    "}<span className="kw">if</span> <span className="fn">any</span>(k <span className="kw">in</span> text <span className="kw">for</span> k <span className="kw">in</span> [{"\n"}
{"        "}<span className="str">"math"</span>, <span className="str">"equation"</span>, <span className="str">"solve"</span>,{"\n"}
{"    "}]):{"\n"}
{"        "}<span className="kw">return</span> <span className="str">"math"</span>{"\n"}
{"    "}<span className="kw">return</span> <span className="str">"creative"</span>{"\n"}
{"\n"}
store = <span className="fn">MemoryKVStore</span>(){"\n"}
cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
llm   = <span className="fn">LatentLLM</span>({"\n"}
{"    "}<span className="str">"HuggingFaceTB/SmolLM-135M"</span>,{"\n"}
{"    "}device=<span className="str">"cpu"</span>, global_cache=cache,{"\n"}
){"\n"}
{"\n"}
builder = <span className="fn">StateGraph</span>(LatentState){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"supervisor"</span>, <span className="fn">AgentPrimitive</span>({"\n"}
{"    "}<span className="str">"Supervisor"</span>, llm,{"\n"}
{"    "}trigger_text=<span className="str">"Classify this problem:"</span>,{"\n"}
{"    "}max_new_tokens=<span className="num">16</span>,{"\n"}
)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"math"</span>, <span className="fn">ReasonPrimitive</span>({"\n"}
{"    "}llm, max_new_tokens=<span className="num">32</span>,{"\n"}
{"    "}trigger_text=<span className="str">"Solve rigorously:"</span>,{"\n"}
)){"\n"}
builder.<span className="fn">add_node</span>(<span className="str">"creative"</span>, <span className="fn">ReasonPrimitive</span>({"\n"}
{"    "}llm, max_new_tokens=<span className="num">32</span>,{"\n"}
{"    "}trigger_text=<span className="str">"Brainstorm creatively:"</span>,{"\n"}
)){"\n"}
{"\n"}
builder.<span className="fn">add_edge</span>(START, <span className="str">"supervisor"</span>){"\n"}
builder.<span className="fn">add_conditional_edges</span>({"\n"}
{"    "}<span className="str">"supervisor"</span>, route,{"\n"}
){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"math"</span>, END){"\n"}
builder.<span className="fn">add_edge</span>(<span className="str">"creative"</span>, END){"\n"}
{"\n"}
graph  = builder.<span className="fn">compile</span>(){"\n"}
result = graph.<span className="fn">invoke</span>({"{"}{"\n"}
{"    "}<span className="str">"messages"</span>: [{"{"}{"\n"}
{"        "}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"        "}<span className="str">"content"</span>: <span className="str">"Calculate rocket trajectory"</span>,{"\n"}
{"    "}{"}"}],{"\n"}
{"    "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"}"}){"\n"}
<span className="fn">print</span>(result[<span className="str">"latent"</span>].text)
            </Py>
          </div>
        </section>

        {/* FastAPI */}
        <section className="doc-section" id="deployment">
          <h2>Example: FastAPI Deployment</h2>
          <p>Serve a LatentMesh pipeline as an HTTP endpoint.</p>
          <div className="code-block">
            <div className="code-block-header"><span>python · server.py</span><CopyBtn /></div>
            <Py>
<span className="kw">from</span> contextlib <span className="kw">import</span> asynccontextmanager{"\n"}
<span className="kw">from</span> fastapi <span className="kw">import</span> FastAPI{"\n"}
<span className="kw">import</span> uvicorn{"\n"}
{"\n"}
<span className="kw">from</span> latentmesh <span className="kw">import</span> LatentLLM, LatentState{"\n"}
<span className="kw">from</span> latentmesh.primitives <span className="kw">import</span> ReasonPrimitive{"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> StateGraph, START, END{"\n"}
{"\n"}
<span className="op">@</span>asynccontextmanager{"\n"}
<span className="kw">async def</span> <span className="fn">lifespan</span>(app):{"\n"}
{"    "}store = <span className="fn">MemoryKVStore</span>(){"\n"}
{"    "}cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
{"    "}app.state.llm = <span className="fn">LatentLLM</span>({"\n"}
{"        "}<span className="str">"HuggingFaceTB/SmolLM-135M"</span>,{"\n"}
{"        "}device=<span className="str">"cpu"</span>, global_cache=cache,{"\n"}
{"    "}){"\n"}
{"    "}b = <span className="fn">StateGraph</span>(LatentState){"\n"}
{"    "}b.<span className="fn">add_node</span>({"\n"}
{"        "}<span className="str">"reason"</span>,{"\n"}
{"        "}<span className="fn">ReasonPrimitive</span>(app.state.llm),{"\n"}
{"    "}){"\n"}
{"    "}b.<span className="fn">add_edge</span>(START, <span className="str">"reason"</span>){"\n"}
{"    "}b.<span className="fn">add_edge</span>(<span className="str">"reason"</span>, END){"\n"}
{"    "}app.state.graph = b.<span className="fn">compile</span>(){"\n"}
{"    "}<span className="kw">yield</span>{"\n"}
{"\n"}
app = <span className="fn">FastAPI</span>(lifespan=lifespan){"\n"}
{"\n"}
<span className="op">@</span>app.<span className="fn">post</span>(<span className="str">"/generate"</span>){"\n"}
<span className="kw">async def</span> <span className="fn">generate</span>(prompt: <span className="cls">str</span>):{"\n"}
{"    "}result = app.state.graph.<span className="fn">invoke</span>({"{"}{"\n"}
{"        "}<span className="str">"messages"</span>: [{"\n"}
{"            "}{"{"}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"             "}<span className="str">"content"</span>: prompt{"}"},
{"\n"}{"        "}],{"\n"}
{"        "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"    "}{"}"})
{"\n"}{"    "}<span className="kw">return</span> {"{"}{"\n"}
{"        "}<span className="str">"response"</span>: result[<span className="str">"latent"</span>].text,{"\n"}
{"    "}{"}"}{"\n"}
            </Py>
          </div>
        </section>

        {/* Debug & Logging */}
        <section className="doc-section" id="debug-logging">
          <h2>Example: Debug &amp; Token Tracking</h2>
          <p>Enable <code>debug=True</code> to see cache hit/miss details and per-agent token accounting.</p>
          <div className="code-block">
            <div className="code-block-header"><span>python</span><CopyBtn /></div>
            <Py>
<span className="kw">import</span> logging{"\n"}
logging.<span className="fn">basicConfig</span>(level=logging.INFO){"\n"}
{"\n"}
llm = <span className="fn">LatentLLM</span>({"\n"}
{"    "}<span className="str">"Qwen/Qwen3-0.6B"</span>,{"\n"}
{"    "}device=<span className="str">"cuda"</span>,{"\n"}
{"    "}global_cache=cache,{"\n"}
{"    "}debug=<span className="kw">True</span>,  <span className="cm"># enables logging</span>{"\n"}
){"\n"}
{"\n"}
result = llm.<span className="fn">generate</span>({"\n"}
{"    "}messages=[{"{"}{"\n"}
{"        "}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"        "}<span className="str">"content"</span>: <span className="str">"Hello"</span>,{"\n"}
{"    "}{"}"}],{"\n"}
{"    "}max_new_tokens=<span className="num">10</span>,{"\n"}
){"\n"}
{"\n"}
<span className="cm"># Token accounting</span>{"\n"}
<span className="fn">print</span>(<span className="str">f"Cached: </span>{"{"}<span className="str">result.cached_tokens</span>{"}"}<span className="str">"</span>){"\n"}
<span className="fn">print</span>(<span className="str">f"Uncached: </span>{"{"}<span className="str">result.input_tokens_uncached</span>{"}"}<span className="str">"</span>){"\n"}
<span className="fn">print</span>(<span className="str">f"Output: </span>{"{"}<span className="str">result.output_tokens</span>{"}"}<span className="str">"</span>)
            </Py>
          </div>
        </section>

        <hr className="section-divider" />

        {/* Benchmarks */}
        <section className="doc-section" id="benchmarks">
          <h2>Benchmarks</h2>
          <p>Performance comparison across three modes: single-model baseline, text-based multi-agent, and LatentMesh KV-cache sharing.</p>

          <h3>GPQA</h3>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Single Model</span><span>32.4%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill warn" data-target="32.4"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Text MAS</span><span>38.1%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill" data-target="38.1"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>LatentMesh</span><span>41.7%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill alt" data-target="41.7"></div></div>
          </div>

          <h3 style={{ marginTop: 28 }}>AIME 2025</h3>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Single Model</span><span>12.0%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill warn" data-target="12.0"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Text MAS</span><span>16.5%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill" data-target="16.5"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>LatentMesh</span><span>20.3%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill alt" data-target="20.3"></div></div>
          </div>

          <h3 style={{ marginTop: 28 }}>LiveCodeBench</h3>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Single Model</span><span>18.7%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill warn" data-target="18.7"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>Text MAS</span><span>22.4%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill" data-target="22.4"></div></div>
          </div>
          <div className="benchmark-bar-container">
            <div className="benchmark-label"><span>LatentMesh</span><span>25.1%</span></div>
            <div className="benchmark-track"><div className="benchmark-fill alt" data-target="25.1"></div></div>
          </div>
        </section>

      </main>
    </div>
  );
}
