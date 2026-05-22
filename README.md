# DMTA Agent

Autonomous drug discovery agent that runs **Design → Make → Test → Analyze** loops,
connecting LLM reasoning to physical lab instruments.

Built as a proof-of-concept for the Lilly Frontier AI / Lab Automation Integration role.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│                 FastAPI Service              │
│  POST /campaigns   GET /campaigns/{id}/run  │
└──────────────────────┬──────────────────────┘
                       │ SSE stream
              ┌────────▼────────┐
              │   DMTAAgent     │  ← orchestrates the loop
              └──┬──────────┬───┘
                 │          │
        ┌────────▼──┐  ┌────▼──────────┐
        │  Claude   │  │  Instruments  │
        │  (LLM)    │  │  Layer        │
        └───────────┘  └──┬────────┬───┘
           Design +       │        │
           Analyze     Hamilton  AssayStation
                        STAR     (TR-FRET +
                       (Make)     ADMET)
                                  (Test)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| **SSE streaming** | Client sees each phase event in real-time; no polling needed |
| **Instrument abstraction layer** | Swap mock → real SDK without touching agent logic |
| **Pydantic throughout** | Every LLM response and instrument payload is validated |
| **Parallel dispensing** | `asyncio.gather()` fires all Hamilton jobs concurrently |
| **Stateless agent** | Campaign state lives in Redis (or in-memory for dev); agent is pure function |

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
uvicorn main:app --reload --port 8000
```

```bash
# Start a campaign
curl -X POST http://localhost:8000/campaigns \
  -H "Content-Type: application/json" \
  -d '{"goal": "JAK2-selective inhibitor: IC50 < 1 nM, >100x selectivity vs JAK1"}'

# Stream agent events (replace {id} with campaign_id from above)
curl -N http://localhost:8000/campaigns/{id}/run
```

---

## Docker

```bash
docker build -t dmta-agent .
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 dmta-agent
```

---

## Connecting real instruments

In `agent/instruments.py`, replace the `_dispense_real` and `_real_panel` methods:

```python
# Hamilton VENUS SDK
async def _dispense_real(self, job: DispenseJob) -> DispenseResult:
    import venus
    result = await venus.execute_method("Dispense", job.model_dump())
    return DispenseResult(**result)

# LIMS (e.g. Benchling, Dotmatics)
async def _real_panel(self, compounds) -> list[AssayResult]:
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{self.lims_url}/assay/jak2", json=[...])
    return [AssayResult(**x) for x in r.json()]
```

The agent loop in `agent/dmta_agent.py` does not change.

---

## Production checklist

- [ ] Replace in-memory `campaigns` dict with Redis
- [ ] Add JWT auth middleware
- [ ] Wire real instrument SDKs in `instruments.py`
- [ ] Set `mock_instruments=False` in `main.py`
- [ ] Deploy as Kubernetes Deployment + Service
- [ ] Add Prometheus metrics on `/metrics`
