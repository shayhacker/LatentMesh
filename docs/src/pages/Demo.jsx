import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import gsap from 'gsap';
import '../styles/demo.css';

const TASK =
  'Build a safe canary deployment pipeline with automated tests, performance tuning, and rollback guardrails.';

const AGENTS = [
  {
    id: 'single',
    label: 'Single Agent',
    subtitle: 'One model does every step in sequence.',
    speedMs: 15,
    code: [
      '# sequential: plan -> code -> test -> tune',
      'def run_canary(payload):',
      "    region    = payload['region']",
      '    artifacts = build_release()',
      '    upload(artifacts, region)',
      '    health    = check_services(region)',
      '    if not health.ok:',
      "        return rollback('health check failed')",
      "    return {'status': 'stable'}",
    ],
    terminal: [
      '$ pytest tests/ -q',
      '....F....F',
      'FAILED tests/test_canary.py::test_rollback',
      'FAILED tests/test_canary.py::test_parallel_upload',
      '108 passed, 2 failed in 18.3s',
      '$ python deploy.py --canary --region us-east-1',
      'uploading artifacts... done',
      'health check: ok',
      'canary: stable  | p95: 312ms',
    ],
    metrics: [
      { label: 'Speed', value: '824ms' },
      { label: 'Tokens', value: '6.6k' },
      { label: 'Tests', value: '108/124' },
      { label: 'Cost', value: '$0.17' },
    ],
  },
  {
    id: 'multi',
    label: 'Multi-Agent',
    subtitle: 'Planner spawns API, test, and perf specialists in parallel.',
    speedMs: 11,
    code: [
      '[planner] decompose -> api | tests | perf',
      '[api] async def run_canary(payload):',
      '[api]     arts = await build_async()',
      '[api]     await asyncio.gather(*[upload(a) for a in arts])',
      '[tests] def test_rollback_guard(c):',
      "[tests]     assert c.post('/roll').status == 200",
      '[tests] def test_parallel_upload(c): ...',
      '[perf]  LRU cache + connection pooling',
      '[planner] merge patches -> validate -> commit',
    ],
    terminal: [
      '$ pytest tests/ -q',
      '..................................................',
      '118 passed in 9.7s',
      '$ python deploy.py --canary --parallel',
      'parallel upload: done',
      'coordination overhead: +31ms',
      'canary: stable  | p95: 241ms',
    ],
    metrics: [
      { label: 'Speed', value: '608ms' },
      { label: 'Tokens', value: '23.1k' },
      { label: 'Tests', value: '118/124' },
      { label: 'Cost', value: '$0.57' },
    ],
  },
  {
    id: 'latent',
    label: 'Multi-Agent + Latent Space',
    subtitle: 'Agents share a fused state vector - minimal tokens, zero merge conflicts.',
    speedMs: 8,
    code: [
      '<latent> goals - constraints - history </latent>',
      'agent.api  -> run_canary(payload, ctx=latent)',
      'agent.test -> synthesize(edges=latent.graph)',
      'agent.perf -> apply(cache=latent.traces)',
      '',
      'patch = latent_fusion(api, test, perf)',
      'gate  = rollback_guard(threshold=0.15)',
      'deploy(patch, gate)',
    ],
    terminal: [
      '$ pytest tests/ -q',
      '..................................................',
      '124 passed in 6.1s',
      '$ python deploy.py --canary --latent',
      'latent fusion: 0 conflicts',
      'canary: stable  | p95: 198ms',
    ],
    metrics: [
      { label: 'Speed', value: '442ms' },
      { label: 'Tokens', value: '3.5k' },
      { label: 'Tests', value: '124/124' },
      { label: 'Cost', value: '$0.09' },
    ],
  },
];

function jitter(base) {
  return base * (1 + (Math.random() * 0.6 - 0.3));
}

function Blink() {
  const [on, setOn] = useState(true);
  useEffect(() => {
    const id = setInterval(() => setOn((v) => !v), 530);
    return () => clearInterval(id);
  }, []);
  return <span className="blink" style={{ opacity: on ? 1 : 0 }} />;
}

function Metric({ label, value }) {
  const ref = useRef(null);
  useEffect(() => {
    if (!ref.current) return;
    const m = value.match(/^([^0-9-]*)(-?[\d.]+)(.*)$/);
    if (!m) {
      ref.current.textContent = value;
      return;
    }
    const [, pre, num, suf] = m;
    const target = Number(num);
    const obj = { n: 0 };
    gsap.to(obj, {
      n: target,
      duration: 0.8,
      ease: 'power2.out',
      onUpdate() {
        if (ref.current)
          ref.current.textContent =
            pre + (target >= 10 ? obj.n.toFixed(1) : obj.n.toFixed(2)) + suf;
      },
    });
  }, [value]);

  return (
    <div className="mc">
      <div className="mc-label">{label}</div>
      <div className="mc-value" ref={ref}>
        {value}
      </div>
    </div>
  );
}

function TerminalCard({ agent, startDelay, onFinish }) {
  const [codeLines, setCodeLines] = useState([]);
  const [termLines, setTermLines] = useState([]);
  const [done, setDone] = useState(false);
  const codeEl = useRef(null);
  const termEl = useRef(null);
  const cbRef = useRef(onFinish);
  cbRef.current = onFinish;

  useEffect(() => {
    let dead = false;
    const allCode = agent.code;
    const allTerm = agent.terminal;
    let ci = 0;
    let ti = 0;

    function nextCode() {
      if (dead) return;
      if (ci < allCode.length) {
        setCodeLines((p) => [...p, allCode[ci]]);
        ci++;
        if (codeEl.current) codeEl.current.scrollTop = 99999;
        setTimeout(nextCode, jitter(agent.speedMs * 18));
      } else {
        setTimeout(nextTerm, 200);
      }
    }

    function nextTerm() {
      if (dead) return;
      if (ti < allTerm.length) {
        setTermLines((p) => [...p, allTerm[ti]]);
        ti++;
        if (termEl.current) termEl.current.scrollTop = 99999;
        setTimeout(nextTerm, jitter(agent.speedMs * 14));
      } else {
        setDone(true);
        cbRef.current();
      }
    }

    const kickoff = setTimeout(nextCode, startDelay);
    return () => {
      dead = true;
      clearTimeout(kickoff);
    };
  }, [agent, startDelay]);

  return (
    <div className={'tc' + (agent.id === 'latent' ? ' tc-win' : '')}>
      <div className="tc-bar">
        <span className="dot dot-r" />
        <span className="dot dot-y" />
        <span className="dot dot-g" />
        <span className="tc-title">{agent.label}</span>
        {agent.id === 'latent' && <span className="tc-badge">Best</span>}
      </div>
      <div className="tc-lbl">Patch stream</div>
      <div className="tc-pane tc-code" ref={codeEl}>
        <pre>
          {codeLines.join('\n')}
          {!done && <Blink />}
        </pre>
      </div>

      <div className="tc-lbl">Terminal</div>
      <div className="tc-pane tc-shell" ref={termEl}>
        <pre>{termLines.join('\n')}</pre>
      </div>

      {done && (
        <motion.div
          className="tc-metrics"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.35 }}
        >
          {agent.metrics.map((m) => (
            <Metric key={m.label} label={m.label} value={m.value} />
          ))}
        </motion.div>
      )}
    </div>
  );
}

export default function Demo() {
  const [showCards, setShowCards] = useState(false);
  const [typed, setTyped] = useState('');
  const [doneCount, setDoneCount] = useState(0);

  useEffect(() => {
    let dead = false;
    let i = 0;

    function tick() {
      if (dead) return;
      i++;
      if (i <= TASK.length) {
        setTyped(TASK.slice(0, i));
        setTimeout(tick, jitter(14));
      } else {
        setTimeout(() => {
          if (!dead) setShowCards(true);
        }, 350);
      }
    }

    setTimeout(tick, 400);
    return () => {
      dead = true;
    };
  }, []);

  const onCardDone = useCallback(() => setDoneCount((n) => n + 1), []);

  if (!showCards) {
    return (
      <div className="demo-page">
        <div className="root-wrap">
          <div className="prompt-center">
            <div className="pw">
              <div className="pw-bar">
                <span className="dot dot-r" />
                <span className="dot dot-y" />
                <span className="dot dot-g" />
                <span className="pw-title">Task Prompt</span>
              </div>
              <div className="pw-body">
                <span className="pw-caret">&gt;</span>
                <span className="pw-text">{typed}</span>
                <Blink />
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="demo-page">
      <div className="root-wrap">
        <div className="grid3">
          {AGENTS.map((a, i) => (
            <TerminalCard
              key={a.id}
              agent={a}
              startDelay={i * 100}
              onFinish={onCardDone}
            />
          ))}
        </div>
        <AnimatePresence>
          {doneCount >= 3 && (
            <motion.div
              className="summary"
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: 8 }}
            >
              Latent-space coordination: <strong>84.8 % fewer tokens</strong> -{' '}
              <strong>7.7x faster</strong> - <strong>full test coverage</strong>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
