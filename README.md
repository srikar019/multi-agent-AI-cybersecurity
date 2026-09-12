# CyberArena

Multi-agent AI cybersecurity defense simulation (Stage 1).

An LLM-driven **Red Team** agent generates live, adaptive attacks against a
synthetic network. An LLM-driven **Blue Team** pipeline (Triage → Log Analysis
→ Incident Report) investigates the resulting logs with no prior knowledge of
the attack and produces a human-readable incident report. A ground-truth
evaluator scores detection accuracy, evidence precision/recall, and false
positive rate.

Everything runs in simulation: synthetic logs, sandboxed hosts, no real
exploit code, no real targets.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │  Simulated Network (synthetic logs) │
                    │  mail01 web01 db01 dc01 ws-*        │
                    └──────▲──────────────────────▲───────┘
                           │ writes logs          │ queries logs
              ┌────────────┴─────────┐   ┌────────┴──────────────────┐
              │  RED TEAM (LLM)      │   │  BLUE TEAM pipeline (LLM) │
              │  recon → phish/brute │   │  Triage → Log Analysis     │
              │  → priv-esc → lateral│   │  → Incident Report         │
              │  → exfiltration      │   │                           │
              └──────────────────────┘   └────────┬──────────────────┘
                                                  │ findings
                                        ┌─────────▼─────────┐
                                        │  EVALUATOR         │
                                        │  ground truth vs.  │
                                        │  detection/findings│
                                        └───────────────────┘
```

Reference: Castro et al., IEEE CAI 2025 — RL defenders beat scripted red
agents for speed/consistency but are black boxes. This project's
differentiator: an *agentic* Red Team (adaptive adversary) plus an
explainable, evidence-cited investigation pipeline.

## Quickstart

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. configure LLM (copy .env.example to .env and add a key)
#    Provider auto-detects: OpenAI > Anthropic > Gemini > mock (offline)
#    e.g. for Google Gemini:
#      GOOGLE_API_KEY=...
#      LLM_PROVIDER=gemini
#      GEMINI_MODEL=gemini-3.1-flash-lite   # higher free-tier quota than 3.6/3.7-flash

# 3. run one scenario end-to-end (Red Team + Blue Team + report)
python -m cyberarena run phishing_lateral

# 4. evaluation harness over all scenarios
python -m cyberarena eval --runs 1

# 5. run tests (no API key required)
pytest
```

## Commands

| Command | Description |
| --- | --- |
| `python -m cyberarena list` | List scenarios |
| `python -m cyberarena run <scenario> --provider <openai\|anthropic\|gemini\|mock>` | Full trace of one scenario incl. incident report |
| `python -m cyberarena eval [keys...] --runs N` | Metrics over scenarios; JSON saved to `outputs/` |

## Scenarios

| Key | Type | Kill chain |
| --- | --- | --- |
| `phishing_lateral` | attack | phishing → credential theft → lateral movement → exfiltration |
| `bruteforce_escalation` | attack | brute force → foothold → privilege escalation → exfiltration |
| `benign_baseline` | benign | normal traffic only (false-positive check) |

Each attack log line carries hidden ground-truth metadata (kill-chain phase)
that is stripped from the Blue Team's view and used only by the evaluator.

## Metrics

- **Detection**: triage `is_incident` vs. whether an attack actually ran →
  TP/FP/TN/FN, accuracy, FPR, FNR.
- **Evidence precision/recall/F1**: overlap between log IDs the analyst flags
  and the true attack log IDs.
- **Log-level FP rate** on benign runs: fraction of a normal corpus wrongly
  flagged.

## Project layout

```
src/cyberarena/
  config.py            env config
  llm_factory.py       OpenAI / Anthropic / mock providers
  sim/                 network topology, log store, noise, digest, runtime, scenarios
  red_team/            LangGraph ReAct agent + tools that mutate the sim
  blue_team/           triage, log-analysis (ReAct), incident report; LangGraph pipeline
  eval/                ground-truth metrics + evaluation runner
  cli.py               command line interface
tests/                 simulator, tools, and metrics tests (no LLM needed)
```

## Notes / limitations (honest caveats)

- Red Team adaptation (Phase B+): the red agent has an explicit
  adaptation playbook (rotate infrastructure on source-IP blocks, retry
  exfiltration via a different destination, re-harvest after credential
  resets, pivot through intermediate hosts), and tool failures return
  actionable recovery guidance. The attacker infrastructure pool is 6
  addresses so the defender must spend an action per address it burns.
  Offline tests prove every recovery path mechanically works; live duels
  (gemini flash-lite) show red adapting, but the defender still wins most
  duels -- red now honestly reports "objective not achieved" instead of
  falsely claiming success.
- LLM agents are non-deterministic and slow; run `eval --runs N` to smooth
  variance.
- Findings depend on the model's ability to cite exact log IDs; the analyst
  agent is prompted to require them, and retries once if it returns nothing.
- Google Gemini free tier caps newer models (e.g. `gemini-3.6-flash`,
  `gemini-3.7-flash`) at ~20 requests/day — one scenario run uses roughly that
  many calls. Use `gemini-3.1-flash-lite` (much larger free quota) or a paid
  tier for real workloads.
- The `mock` provider exercises wiring only — use it to validate the harness,
  not results.
- Sample Stage-1 result (gemini-3.1-flash-lite, 1 run each): detection
  accuracy 100% (2/2 attacks detected, 0/1 benign flagged), evidence precision
  1.00, recall 0.64-0.82. Known weakness: the analyst over-flags benign logs
  (log FP rate ~13% on the benign corpus).
- Planned later stages: threat intelligence, root-cause, risk-assessment
  agents, and a cross-agent correlation layer for multi-domain attacks.