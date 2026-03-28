import React, { useEffect, useRef, useState } from 'react';
import { motion, useInView, useScroll, useTransform, animate } from 'framer-motion';
import { Terminal, Github, ArrowRight, ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';

/* ────────────────────────────────────────────────────────────────
   Animation variants
   ──────────────────────────────────────────────────────────────── */
const fade = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { duration: 0.8, ease: [0.25, 0.1, 0.25, 1] } },
};
const fadeUp = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0, transition: { duration: 0.7, ease: [0.25, 0.1, 0.25, 1] } },
};
const stagger = {
    hidden: {},
    show: { transition: { staggerChildren: 0.12, delayChildren: 0.1 } },
};
const sectionReveal = {
    hidden: { opacity: 0, y: 40 },
    show: { opacity: 1, y: 0, transition: { duration: 0.8, ease: [0.25, 0.1, 0.25, 1] } },
};

/* ────────────────────────────────────────────────────────────────
   Helpers
   ──────────────────────────────────────────────────────────────── */
function CountUp({ to, suffix = '', decimals = 0 }) {
    const ref = useRef(null);
    const inView = useInView(ref, { once: true });
    useEffect(() => {
        if (!inView) return;
        const ctrl = animate(0, to, {
            duration: 1.6,
            ease: [0.25, 0.1, 0.25, 1],
            onUpdate(v) { if (ref.current) ref.current.textContent = v.toFixed(decimals) + suffix; },
        });
        return ctrl.stop;
    }, [inView]);
    return <span ref={ref}>0{suffix}</span>;
}

function CopyBtn({ text }) {
    const [copied, set] = useState(false);
    return (
        <button
            className={`copy-btn${copied ? ' copied' : ''}`}
            onClick={() => { navigator.clipboard.writeText(text); set(true); setTimeout(() => set(false), 1800); }}
        >
            <svg viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5" width={11} height={11}>
                {copied
                    ? <path d="M4 8.5l2.5 2.5L12 5" />
                    : <><rect x="5" y="5" width="9" height="9" rx="1.5" /><path d="M5 11H3.5A1.5 1.5 0 012 9.5v-7A1.5 1.5 0 013.5 1h7A1.5 1.5 0 0112 2.5V5" /></>
                }
            </svg>
            {copied ? 'Copied' : 'Copy'}
        </button>
    );
}

/* beam line component – thin, ambient animated line with tint */
function BeamLine() {
    return (
        <div style={{ position: 'relative', width: '100%', height: 1, overflow: 'hidden' }}>
            <div style={{
                position: 'absolute', inset: 0,
                background: 'var(--b-default)',
            }} />
            <motion.div
                animate={{ left: ['-40%', '100%'] }}
                transition={{ duration: 4, repeat: Infinity, ease: 'linear', repeatDelay: 1 }}
                style={{
                    position: 'absolute', top: 0,
                    width: '40%', height: '100%',
                    background: 'linear-gradient(90deg, transparent, var(--c-tint), transparent)',
                    opacity: 0.8,
                }}
            />
        </div>
    );
}

/* ────────────────────────────────────────────────────────────────
   Data
   ──────────────────────────────────────────────────────────────── */
const STATS = [
    { n: 41.7, s: '%', d: 1, label: 'GPQA' },
    { n: 20.3, s: '%', d: 1, label: 'AIME 2025' },
    { n: 25.1, s: '%', d: 1, label: 'LiveCodeBench' },
    { n: 9, s: '%', d: 0, label: 'vs Text MAS' },
];

const FEATURES = [
    { title: 'KV cache sharing', desc: 'Share transformer KV caches directly between agents. Downstream agents skip re-encoding upstream context.' },
    { title: 'Prefix-matched reuse', desc: 'Trie-indexed global cache finds the longest matching prefix in O(L) — only delta tokens are encoded.' },
    { title: 'Token accounting', desc: 'Track cached, uncached, and output tokens per agent. Full visibility into what was reused.' },
    { title: 'LangGraph native', desc: 'Drop-in primitives for StateGraph. Build complex topologies with the tools you already use.' },
];

const installText = `pip install latentmesh`;
const quickRaw = `from latentmesh import (
    LatentLLM, LatentState,
    PlanPrimitive, ReasonPrimitive, ReviewPrimitive,
)
from latentmesh.persistent_cache import (
    MemoryKVStore, GlobalPrefixCache,
)
from langgraph.graph import StateGraph, START, END

store = MemoryKVStore()
cache = GlobalPrefixCache(store)
llm   = LatentLLM("Qwen/Qwen3-0.6B",
                   device="cuda", global_cache=cache)

builder = StateGraph(LatentState)
builder.add_node("plan",   PlanPrimitive(llm))
builder.add_node("reason", ReasonPrimitive(llm))
builder.add_node("review", ReviewPrimitive(llm))
builder.add_edge(START, "plan")
builder.add_edge("plan",   "reason")
builder.add_edge("reason", "review")
builder.add_edge("review", END)

graph  = builder.compile()
result = graph.invoke({
    "messages": [{"role": "user",
                  "content": "Prove √2 is irrational"}],
    "tokens_so_far": 0,
})
print(result["latent"].text)`;

const TEAM = [
    { i: 'YR', name: 'Yash Ranjith', role: 'Stanford' },
    { i: 'WP', name: 'William Peng', role: 'Stanford' },
    { i: 'HK', name: 'Hiroki Kimiwada', role: 'Stanford' },
    { i: 'AR', name: 'Atharva Rao', role: 'Stanford' },
];


/* ────────────────────────────────────────────────────────────────
   Pipeline visualisation — clean, Vercel-style
   ──────────────────────────────────────────────────────────────── */
function PipelineDiagram() {
    const nodes = ['Query', 'Plan', 'Reason', 'Answer'];
    return (
        <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            gap: 0, padding: '48px 0', width: '100%',
        }}>
            {nodes.map((label, i) => (
                <React.Fragment key={label}>
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        whileInView={{ opacity: 1, scale: 1 }}
                        viewport={{ once: true }}
                        transition={{ delay: i * 0.1, duration: 0.5, ease: [0.25, 0.1, 0.25, 1] }}
                        style={{
                            display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8,
                            position: 'relative', zIndex: 2
                        }}
                    >
                        <div style={{
                            width: 56, height: 56,
                            borderRadius: '50%',
                            border: `1px solid ${i === nodes.length - 1 ? 'var(--c-tint)' : 'var(--b-default)'}`,
                            background: i === nodes.length - 1 ? 'var(--c-tint)' : 'var(--c-bg-elevated)',
                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                            fontSize: '0.6875rem',
                            fontWeight: 500,
                            fontFamily: "'JetBrains Mono', monospace",
                            color: i === nodes.length - 1 ? '#fff' : 'var(--t-secondary)',
                            letterSpacing: '-0.01em',
                            boxShadow: i === nodes.length - 1 ? '0 4px 20px rgba(139,139,255,0.25)' : 'none',
                        }}>
                            {label.charAt(0)}
                        </div>
                        <span style={{
                            fontSize: '0.6875rem',
                            color: i === nodes.length - 1 ? 'var(--c-tint)' : 'var(--t-tertiary)',
                            fontWeight: i === nodes.length - 1 ? 500 : 450,
                            letterSpacing: '-0.01em',
                        }}>
                            {label}
                        </span>
                    </motion.div>
                    {i < nodes.length - 1 && (
                        <div style={{
                            position: 'relative', width: 80, height: 1,
                            margin: '0 -8px', marginBottom: 24,
                            zIndex: -1
                        }}>
                            <div style={{
                                position: 'absolute', inset: 0,
                                background: 'var(--b-default)',
                            }} />
                            <motion.div
                                animate={{ left: ['-40%', '100%'] }}
                                transition={{ duration: 2, repeat: Infinity, ease: 'linear', repeatDelay: 1 + i * 0.4 }}
                                style={{
                                    position: 'absolute', top: 0,
                                    width: '40%', height: '100%',
                                    background: 'linear-gradient(90deg, transparent, var(--c-tint), transparent)',
                                    opacity: 0.5,
                                }}
                            />
                        </div>
                    )}
                </React.Fragment>
            ))}
        </div>
    );
}


/* ────────────────────────────────────────────────────────────────
   Home
   ──────────────────────────────────────────────────────────────── */
export default function Home() {
    const heroRef = useRef(null);
    const { scrollYProgress } = useScroll({ target: heroRef, offset: ['start start', 'end start'] });
    const heroOpacity = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

    return (
        <div style={{ position: 'relative', minHeight: '100vh', paddingTop: 'var(--header-h)' }}>

            {/* ─── Hero ───────────────────────────────────────── */}
            <section ref={heroRef} style={{
                position: 'relative',
                minHeight: 'calc(100vh - var(--header-h))',
                display: 'flex', flexDirection: 'column',
                alignItems: 'center', justifyContent: 'center',
                textAlign: 'center',
                padding: '80px 24px 120px',
            }}>
                <motion.div style={{ opacity: heroOpacity, maxWidth: 720, position: 'relative', zIndex: 1 }}>
                    <motion.div variants={stagger} initial="hidden" animate="show">

                        {/* badge */}
                        <motion.div variants={fadeUp} style={{ marginBottom: 40 }}>
                            <span style={{
                                display: 'inline-flex', alignItems: 'center', gap: 8,
                                padding: '4px 12px 4px 4px', borderRadius: 40,
                                border: '1px solid var(--b-tint)',
                                fontSize: '0.75rem', color: 'var(--t-secondary)',
                                fontWeight: 450,
                                background: 'var(--c-tint-subtle)',
                            }}>
                                <span style={{
                                    background: 'var(--c-tint)', color: '#fff',
                                    padding: '2px 8px', borderRadius: 20,
                                    fontSize: '0.625rem', fontWeight: 600, letterSpacing: '0.04em',
                                }}>v0.5.2</span>
                                Now available
                            </span>
                        </motion.div>

                        {/* title */}
                        <motion.h1 variants={fadeUp} style={{
                            fontSize: 'clamp(3rem, 8vw, 5.5rem)',
                            fontWeight: 600,
                            letterSpacing: '-0.04em',
                            lineHeight: 0.95,
                            marginBottom: 28,
                            color: 'var(--t-primary)',
                        }}>
                            Latent<em style={{ fontStyle: 'italic', fontWeight: 300 }}>Mesh</em>
                        </motion.h1>

                        {/* subtitle */}
                        <motion.p variants={fadeUp} style={{
                            fontSize: 'clamp(1rem, 1.5vw, 1.15rem)',
                            color: 'var(--t-secondary)',
                            lineHeight: 1.65,
                            maxWidth: 460, margin: '0 auto 48px',
                            fontWeight: 400,
                        }}>
                            Multi-agent KV-cache communication for LLMs.{' '}
                            Zero redundant re-encoding.
                        </motion.p>

                        {/* cta */}
                        <motion.div variants={fadeUp} style={{ display: 'flex', justifyContent: 'center', gap: 10, flexWrap: 'wrap' }}>
                            <Link to="/docs" className="btn btn-primary btn-lg">
                                <Terminal size={14} /> Documentation
                            </Link>
                            <a href="https://github.com/shayhacker/LatentMesh" target="_blank" rel="noopener noreferrer" className="btn btn-ghost btn-lg">
                                <Github size={14} /> GitHub <ArrowUpRight size={12} style={{ opacity: 0.4 }} />
                            </a>
                        </motion.div>
                    </motion.div>
                </motion.div>

                {/* animated ambient orbs — very subtle drifting gradients */}
                <div style={{ position: 'absolute', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
                    <motion.div
                        animate={{ x: [0, 40, -20, 0], y: [0, -30, 20, 0] }}
                        transition={{ duration: 20, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', top: '20%', left: '30%',
                            width: 500, height: 500, borderRadius: '50%',
                            background: 'radial-gradient(circle, var(--c-tint) 0%, transparent 70%)',
                            opacity: 0.04, filter: 'blur(60px)',
                        }}
                    />
                    <motion.div
                        animate={{ x: [0, -30, 25, 0], y: [0, 25, -35, 0] }}
                        transition={{ duration: 25, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', top: '40%', right: '20%',
                            width: 400, height: 400, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(139,139,255,0.8) 0%, transparent 70%)',
                            opacity: 0.03, filter: 'blur(50px)',
                        }}
                    />
                    <motion.div
                        animate={{ x: [0, 20, -15, 0], y: [0, -20, 15, 0] }}
                        transition={{ duration: 18, repeat: Infinity, ease: 'easeInOut' }}
                        style={{
                            position: 'absolute', top: '60%', left: '50%',
                            width: 350, height: 350, borderRadius: '50%',
                            background: 'radial-gradient(circle, rgba(99,98,231,0.6) 0%, transparent 70%)',
                            opacity: 0.03, filter: 'blur(40px)',
                        }}
                    />
                </div>

                {/* scroll indicator */}
                <motion.div
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                    transition={{ delay: 1.4, duration: 0.6 }}
                    style={{ position: 'absolute', bottom: 40, left: '50%', transform: 'translateX(-50%)' }}
                >
                    <motion.div
                        animate={{ y: [0, 6, 0] }}
                        transition={{ repeat: Infinity, duration: 2.5, ease: 'easeInOut' }}
                        style={{
                            width: 1, height: 40,
                            background: 'linear-gradient(to bottom, var(--c-tint), transparent)',
                            opacity: 0.4,
                        }}
                    />
                </motion.div>
            </section>

            {/* ─── Pipeline ──────────────────────────────────── */}
            <motion.section
                variants={sectionReveal} initial="hidden" whileInView="show"
                viewport={{ once: true, margin: '-80px' }}
                style={{ position: 'relative', maxWidth: 800, margin: '0 auto', padding: '0 24px 120px' }}
            >
                <BeamLine />
                <div style={{ paddingTop: 80 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 16, flexWrap: 'wrap', gap: 16 }}>
                        <div>
                            <p style={{
                                fontSize: '0.6875rem', fontWeight: 500, letterSpacing: '0.08em',
                                textTransform: 'uppercase', color: 'var(--t-tertiary)', marginBottom: 8,
                            }}>Architecture</p>
                            <h2 style={{
                                fontSize: 'clamp(1.5rem, 3vw, 2.25rem)',
                                fontWeight: 600, letterSpacing: '-0.035em',
                                lineHeight: 1.1,
                                color: 'var(--t-primary)',
                            }}>
                                Agents share<br />KV caches
                            </h2>
                        </div>
                        <p style={{
                            fontSize: '0.875rem', color: 'var(--t-secondary)',
                            maxWidth: 300, textAlign: 'right', lineHeight: 1.6, fontWeight: 400,
                        }}>
                            Each agent's KV cache is stored in a global prefix cache.
                            Downstream agents skip re-encoding the overlapping context.
                        </p>
                    </div>
                    <div style={{
                        border: '1px solid var(--b-default)',
                        borderRadius: 'var(--r)',
                        background: 'var(--c-surface)',
                        overflow: 'hidden',
                    }}>
                        <PipelineDiagram />
                        <div style={{
                            display: 'flex', justifyContent: 'center', gap: 24, padding: '12px 0 16px',
                            borderTop: '1px solid var(--b-default)',
                        }}>
                            {[
                                ['var(--t-tertiary)', 'KV cache shared'],
                                ['var(--c-tint)', 'text generated'],
                            ].map(([c, l]) => (
                                <span key={l} style={{
                                    display: 'flex', alignItems: 'center', gap: 6,
                                    fontSize: '0.6875rem', color: 'var(--t-tertiary)',
                                    fontFamily: "'JetBrains Mono', monospace",
                                }}>
                                    <span style={{ width: 5, height: 5, borderRadius: '50%', background: c, display: 'inline-block' }} />
                                    {l}
                                </span>
                            ))}
                        </div>
                    </div>
                </div>
            </motion.section>

            {/* ─── Stats ─────────────────────────────────────── */}
            <motion.section
                variants={sectionReveal} initial="hidden" whileInView="show"
                viewport={{ once: true, margin: '-80px' }}
                style={{ position: 'relative', maxWidth: 800, margin: '0 auto', padding: '0 24px 120px' }}
            >
                <BeamLine />
                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(4, 1fr)',
                    marginTop: 1,
                }}>
                    {STATS.map((s, i) => (
                        <motion.div
                            key={i}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.08, duration: 0.6, ease: [0.25, 0.1, 0.25, 1] }}
                            style={{
                                padding: '48px 0',
                                borderRight: i < 3 ? '1px solid var(--b-default)' : 'none',
                                paddingLeft: i > 0 ? 24 : 0,
                                paddingRight: i < 3 ? 24 : 0,
                            }}
                        >
                            <div style={{
                                fontSize: 'clamp(2rem, 3.5vw, 2.75rem)',
                                fontWeight: 600,
                                letterSpacing: '-0.04em',
                                lineHeight: 1,
                                color: 'var(--t-primary)',
                                fontVariantNumeric: 'tabular-nums',
                            }}>
                                <CountUp to={s.n} suffix={s.s} decimals={s.d} />
                            </div>
                            <div style={{
                                fontSize: '0.6875rem',
                                color: 'var(--t-tertiary)',
                                fontWeight: 450,
                                letterSpacing: '0.02em',
                                textTransform: 'uppercase',
                                marginTop: 8,
                            }}>{s.label}</div>
                        </motion.div>
                    ))}
                </div>
            </motion.section>

            {/* ─── Features ──────────────────────────────────── */}
            <motion.section
                variants={sectionReveal} initial="hidden" whileInView="show"
                viewport={{ once: true, margin: '-80px' }}
                style={{ position: 'relative', maxWidth: 800, margin: '0 auto', padding: '0 24px 120px' }}
            >
                <BeamLine />
                <div style={{ paddingTop: 80 }}>
                    <p style={{
                        fontSize: '0.6875rem', fontWeight: 500, letterSpacing: '0.08em',
                        textTransform: 'uppercase', color: 'var(--t-tertiary)', marginBottom: 8,
                    }}>Why LatentMesh</p>
                    <h2 style={{
                        fontSize: 'clamp(1.5rem, 3vw, 2.25rem)',
                        fontWeight: 600, letterSpacing: '-0.035em',
                        lineHeight: 1.1,
                        color: 'var(--t-primary)',
                        marginBottom: 48,
                    }}>
                        Built for speed<br />and precision
                    </h2>
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(2, 1fr)',
                        gap: 1,
                        background: 'var(--b-default)',
                        borderRadius: 'var(--r)',
                        overflow: 'hidden',
                    }}>
                        {FEATURES.map((f, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0 }}
                                whileInView={{ opacity: 1 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.06, duration: 0.5 }}
                                style={{
                                    padding: '32px 28px',
                                    background: 'var(--c-bg)',
                                }}
                                onMouseEnter={e => { e.currentTarget.style.background = 'var(--c-bg-elevated)'; }}
                                onMouseLeave={e => { e.currentTarget.style.background = 'var(--c-bg)'; }}
                            >
                                <h3 style={{
                                    fontSize: '0.9375rem',
                                    fontWeight: 550,
                                    color: 'var(--t-primary)',
                                    marginBottom: 8,
                                    letterSpacing: '-0.01em',
                                }}>{f.title}</h3>
                                <p style={{
                                    fontSize: '0.8125rem',
                                    color: 'var(--t-secondary)',
                                    lineHeight: 1.6,
                                    margin: 0,
                                }}>{f.desc}</p>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </motion.section>

            {/* ─── Code ──────────────────────────────────────── */}
            <section style={{ position: 'relative', maxWidth: 680, margin: '0 auto', padding: '0 24px 120px' }}>
                <motion.div variants={sectionReveal} initial="hidden" whileInView="show" viewport={{ once: true, margin: '-80px' }}>
                    <BeamLine />
                    <div style={{ paddingTop: 80 }}>
                        <p style={{
                            fontSize: '0.6875rem', fontWeight: 500, letterSpacing: '0.08em',
                            textTransform: 'uppercase', color: 'var(--t-tertiary)', marginBottom: 8,
                        }}>Quickstart</p>
                        <h2 style={{
                            fontSize: 'clamp(1.5rem, 3vw, 2.25rem)',
                            fontWeight: 600, letterSpacing: '-0.035em',
                            lineHeight: 1.1,
                            color: 'var(--t-primary)',
                            marginBottom: 36,
                        }}>
                            Up and running fast
                        </h2>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                            <motion.div
                                initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ duration: 0.5 }}
                                className="code-block"
                            >
                                <div className="code-block-header"><span>bash</span><CopyBtn text={installText} /></div>
                                <pre><code>pip install latentmesh</code></pre>
                            </motion.div>
                            <motion.div
                                initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }} transition={{ duration: 0.5, delay: 0.1 }}
                                className="code-block"
                            >
                                <div className="code-block-header"><span>python</span><CopyBtn text={quickRaw} /></div>
                                <pre><code>
<span className="kw">from</span> latentmesh <span className="kw">import</span> ({"\n"}
{"    "}LatentLLM, LatentState,{"\n"}
{"    "}PlanPrimitive, ReasonPrimitive, ReviewPrimitive,{"\n"}
){"\n"}
<span className="kw">from</span> latentmesh.persistent_cache <span className="kw">import</span> ({"\n"}
{"    "}MemoryKVStore, GlobalPrefixCache,{"\n"}
){"\n"}
<span className="kw">from</span> langgraph.graph <span className="kw">import</span> StateGraph, START, END{"\n"}
{"\n"}
store = <span className="fn">MemoryKVStore</span>(){"\n"}
cache = <span className="fn">GlobalPrefixCache</span>(store){"\n"}
llm   = <span className="fn">LatentLLM</span>(<span className="str">"Qwen/Qwen3-0.6B"</span>,{"\n"}
{"                   "}device=<span className="str">"cuda"</span>, global_cache=cache){"\n"}
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
{"    "}<span className="str">"messages"</span>: [{"{"}<span className="str">"role"</span>: <span className="str">"user"</span>,{"\n"}
{"                  "}<span className="str">"content"</span>: <span className="str">"Prove √2 is irrational"</span>{"}"}],{"\n"}
{"    "}<span className="str">"tokens_so_far"</span>: <span className="num">0</span>,{"\n"}
{"}"}){"\n"}
<span className="fn">print</span>(result[<span className="str">"latent"</span>].text)
                                </code></pre>
                            </motion.div>
                        </div>
                    </div>
                </motion.div>
            </section>

            {/* ─── Team ──────────────────────────────────────── */}
            <motion.section
                variants={sectionReveal} initial="hidden" whileInView="show"
                viewport={{ once: true, margin: '-60px' }}
                style={{ position: 'relative', maxWidth: 800, margin: '0 auto', padding: '0 24px 80px' }}
            >
                <BeamLine />
                <div style={{ paddingTop: 80 }}>
                    <p style={{
                        fontSize: '0.6875rem', fontWeight: 500, letterSpacing: '0.08em',
                        textTransform: 'uppercase', color: 'var(--t-tertiary)', marginBottom: 36,
                    }}>Team</p>
                    <div style={{ display: 'flex', gap: 32, flexWrap: 'wrap' }}>
                        {TEAM.map((m, i) => (
                            <motion.div
                                key={m.name}
                                initial={{ opacity: 0, y: 12 }}
                                whileInView={{ opacity: 1, y: 0 }}
                                viewport={{ once: true }}
                                transition={{ delay: i * 0.06, duration: 0.5 }}
                                style={{
                                    display: 'flex', alignItems: 'center', gap: 10,
                                    padding: '6px 10px 6px 6px',
                                    borderRadius: 8,
                                    transition: 'background 0.15s ease',
                                    cursor: 'default',
                                }}
                                onMouseEnter={e => { e.currentTarget.style.background = 'var(--c-surface)'; }}
                                onMouseLeave={e => { e.currentTarget.style.background = 'transparent'; }}
                            >
                                <div style={{
                                    width: 36, height: 36, borderRadius: '50%',
                                    border: '1px solid var(--b-default)',
                                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                                    fontSize: '0.6875rem', fontWeight: 600,
                                    color: 'var(--t-secondary)',
                                    background: 'var(--c-surface)',
                                }}>
                                    {m.i}
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.8125rem', fontWeight: 550, color: 'var(--t-primary)', letterSpacing: '-0.01em' }}>{m.name}</div>
                                    <div style={{ fontSize: '0.6875rem', color: 'var(--t-tertiary)', marginTop: 1 }}>{m.role}</div>
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            </motion.section>

            {/* ─── Footer ──────────────────────────────────────── */}
            <footer style={{
                position: 'relative',
                maxWidth: 800, margin: '0 auto',
                padding: '0 24px',
            }}>
                <div style={{ height: 1, background: 'var(--b-default)' }} />
                <div style={{
                    padding: '24px 0',
                    display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                    flexWrap: 'wrap', gap: 16,
                }}>
                    <div>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                            <span style={{
                                width: 18, height: 18, borderRadius: 5,
                                background: 'var(--c-tint)',
                                display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.5rem', fontWeight: 700, color: '#fff',
                            }}>L</span>
                            <span style={{ fontSize: '0.8125rem', fontWeight: 550, color: 'var(--t-secondary)' }}>LatentMesh</span>
                        </div>
                        <span style={{ fontSize: '0.6875rem', color: 'var(--t-tertiary)' }}>
                            MIT License · Multi-agent KV-cache communication
                        </span>
                    </div>
                    <div style={{ display: 'flex', gap: 20, fontSize: '0.75rem' }}>
                        <a href="https://github.com/shayhacker/LatentMesh" target="_blank" rel="noopener noreferrer" style={{ color: 'var(--t-tertiary)', transition: 'color 0.15s' }}>GitHub</a>
                        <Link to="/docs" style={{ color: 'var(--t-tertiary)', transition: 'color 0.15s' }}>Docs</Link>
                    </div>
                </div>
            </footer>
        </div>
    );
}
