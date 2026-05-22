"""
DMTA Agent — FastAPI service

Endpoints:
  POST /campaigns                  → start a new campaign, returns campaign_id
  GET  /campaigns/{id}/run         → SSE stream of agent events
  POST /campaigns/{id}/approve     → scientist approval gate (approve / edit / stop)
  GET  /campaigns/{id}             → current campaign state (JSON)
  GET  /health                     → liveness probe (for K8s)

Run locally:
  uvicorn main:app --reload --port 8000
"""

import asyncio, logging, uuid
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent.models import ApprovalRequest, CampaignRequest, CampaignState, CampaignStatus
from agent.dmta_agent import DMTAAgent, register_campaign, submit_approval

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")

app = FastAPI(
    title="DMTA Agent",
    description="Agentic lab automation: Design → Make → Test → Analyze",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store — swap for Redis in production
campaigns: dict[str, CampaignState] = {}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Kubernetes liveness / readiness probe."""
    return {"status": "ok"}


@app.post("/campaigns", status_code=201)
async def create_campaign(req: CampaignRequest) -> dict:
    """Register a new DMTA campaign and initialise its approval gate."""
    campaign_id = str(uuid.uuid4())
    campaigns[campaign_id] = CampaignState(
        campaign_id=campaign_id,
        goal=req.goal,
    )
    register_campaign(campaign_id)   # sets up asyncio.Event for approval gating
    return {"campaign_id": campaign_id, "goal": req.goal}


@app.get("/campaigns/{campaign_id}/run")
async def run_campaign(campaign_id: str):
    """
    SSE stream — runs the DMTA agent loop, pausing at each human-in-the-loop gate.
    The stream stays open while the agent awaits approval; POST /approve unblocks it.
    """
    state = campaigns.get(campaign_id)
    if not state:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if state.status == CampaignStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Campaign already running")

    agent = DMTAAgent(mock_instruments=True)

    async def event_stream() -> AsyncIterator[str]:
        async for event in agent.run(state):
            yield event
        yield f"data: {state.model_dump_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        },
    )


@app.post("/campaigns/{campaign_id}/approve", status_code=200)
async def approve_campaign(campaign_id: str, req: ApprovalRequest) -> dict:
    """
    Scientist approval gate.

    Called by the UI after the agent emits an `approval_required` event.
    Unblocks the waiting agent loop so it can proceed to the next iteration.

    Body:
      { "decision": "approve" | "edit" | "stop", "override_note": "optional text" }
    """
    state = campaigns.get(campaign_id)
    if not state:
        raise HTTPException(status_code=404, detail="Campaign not found")
    if not state.awaiting_approval:
        raise HTTPException(status_code=409, detail="Campaign is not awaiting approval")

    submit_approval(campaign_id, req.decision, req.override_note)
    return {"campaign_id": campaign_id, "decision": req.decision}


@app.get("/campaigns/{campaign_id}")
async def get_campaign(campaign_id: str) -> CampaignState:
    """Poll current state of a campaign (alternative to SSE)."""
    state = campaigns.get(campaign_id)
    if not state:
        raise HTTPException(status_code=404, detail="Campaign not found")
    return state
