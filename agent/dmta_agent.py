"""
DMTAAgent — the core agentic loop.

Orchestrates Design → Make → Test → Analyze iterations using:
  - Anthropic Claude for scientific reasoning (Design + Analyze phases)
  - HamiltonSTAR for compound dispensing (Make phase)
  - AssayStation for experimental readouts (Test phase)

Emits structured SSE events so the API layer can stream progress to any client.
"""

from __future__ import annotations
import asyncio, json, logging, re
from typing import AsyncIterator

import anthropic

from agent.models import (
    ApprovalDecision, AssayResult, CampaignState, CampaignStatus,
    CompoundSpec, DesignOutput, DispenseJob, IterationRecord, AnalysisOutput,
)
from agent.instruments import HamiltonSTAR, AssayStation

log = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────────────────
MODEL   = "claude-opus-4-5"
SYS_PROMPT = (
    "You are an autonomous AI lab agent executing DMTA cycles for drug discovery. "
    "You reason like a senior medicinal chemist with deep SAR knowledge. "
    "Always respond with ONLY valid JSON — no markdown, no preamble, no explanation."
)

# ── Per-campaign approval gates (asyncio.Event + decision store) ──────────────
# Keyed by campaign_id. The agent awaits the event; the API endpoint sets it.
_approval_events:    dict[str, asyncio.Event] = {}
_approval_decisions: dict[str, dict]          = {}


def register_campaign(campaign_id: str) -> None:
    """Called by main.py when a campaign is created."""
    _approval_events[campaign_id]    = asyncio.Event()
    _approval_decisions[campaign_id] = {}


def submit_approval(campaign_id: str, decision: str, override_note: str | None) -> None:
    """Called by the /approve endpoint to unblock the waiting agent."""
    _approval_decisions[campaign_id] = {
        "decision":      decision,
        "override_note": override_note,
    }
    _approval_events[campaign_id].set()


def _event(kind: str, **payload) -> str:
    """Format a Server-Sent Event string."""
    return f"data: {json.dumps({'type': kind, **payload})}\n\n"


def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    clean = re.sub(r"```json\s*|```\s*", "", text).strip()
    return json.loads(clean)


class DMTAAgent:
    """
    Autonomous DMTA campaign agent with human-in-the-loop approval gates.

    After each Analyze phase the agent pauses, surfaces findings + proposed
    next steps, and waits for a scientist to Approve / Edit / Stop before
    continuing. The gate is implemented as an asyncio.Event so the SSE
    stream stays open and the scientist's POST /approve unblocks it.

    Usage:
        agent = DMTAAgent(mock_instruments=True)
        async for event in agent.run(state):
            yield event
    """

    def __init__(self, mock_instruments: bool = True):
        self.client   = anthropic.AsyncAnthropic()
        self.hamilton = HamiltonSTAR(mock=mock_instruments)
        self.assay    = AssayStation(mock=mock_instruments)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, state: CampaignState) -> AsyncIterator[str]:
        state.status = CampaignStatus.RUNNING
        yield _event("status", value="running", campaign_id=state.campaign_id)

        prev_best:     IterationRecord | None = None
        override_note: str | None             = None   # scientist's edit from prior gate

        for i in range(3):
            yield _event("iteration_start", iteration=i + 1)

            # ── DESIGN ────────────────────────────────────────────────────────
            yield _event("phase", value="design", iteration=i + 1)
            design = await self._design(state.goal, i, prev_best, override_note)
            override_note = None   # consumed
            yield _event("design_complete",
                         iteration=i + 1,
                         hypothesis=design.hypothesis,
                         compounds=[c.model_dump() for c in design.compounds])

            # ── MAKE ──────────────────────────────────────────────────────────
            yield _event("phase", value="make", iteration=i + 1)
            dispense_results = await self._make(design.compounds, i)
            passed = sum(1 for r in dispense_results if r.qc_passed)
            yield _event("make_complete", iteration=i + 1,
                         compounds_dispensed=len(dispense_results),
                         qc_passed=passed)

            # ── TEST ──────────────────────────────────────────────────────────
            yield _event("phase", value="test", iteration=i + 1)
            assay_results = await self._test(design.compounds, i)
            yield _event("test_complete",
                         iteration=i + 1,
                         results=[r.model_dump() for r in assay_results])

            # ── ANALYZE ───────────────────────────────────────────────────────
            yield _event("phase", value="analyze", iteration=i + 1)
            analysis = await self._analyze(state.goal, assay_results, i)
            yield _event("analysis_complete", iteration=i + 1, **analysis.model_dump())

            # ── Record ────────────────────────────────────────────────────────
            record = IterationRecord(
                iteration=i + 1,
                design=design,
                results=assay_results,
                analysis=analysis,
            )
            state.iterations.append(record)
            prev_best = record

            # ── Goal achieved — no gate needed ────────────────────────────────
            if analysis.goal_achieved:
                state.lead_compound = analysis.best_compound
                state.status        = CampaignStatus.COMPLETE
                yield _event("campaign_complete",
                             lead=analysis.best_compound,
                             iterations=i + 1)
                return

            # ── Last iteration — no gate needed ───────────────────────────────
            if i == 2:
                break

            # ── HUMAN-IN-THE-LOOP GATE ────────────────────────────────────────
            # Pause the agent and ask the scientist to approve / edit / stop.
            cid   = state.campaign_id
            event = _approval_events[cid]
            event.clear()
            state.awaiting_approval = True

            yield _event(
                "approval_required",
                iteration        = i + 1,
                lead             = analysis.best_compound,
                convergence_score= analysis.convergence_score,
                sar_trends       = analysis.sar_trends,
                next_steps       = analysis.next_steps,
                reasoning        = analysis.reasoning,
            )

            log.info(f"Campaign {cid} paused — awaiting scientist approval after iter {i+1}")
            await event.wait()           # ← blocks until POST /approve fires

            decision     = _approval_decisions[cid].get("decision", "approve")
            override_note= _approval_decisions[cid].get("override_note")
            state.awaiting_approval = False

            yield _event("approval_received",
                         iteration=i + 1,
                         decision=decision,
                         override_note=override_note)

            if decision == ApprovalDecision.STOP:
                state.lead_compound = analysis.best_compound
                state.status        = CampaignStatus.COMPLETE
                yield _event("campaign_complete",
                             lead=state.lead_compound,
                             iterations=i + 1,
                             stopped_by_scientist=True)
                return

            # APPROVE or EDIT — continue loop (override_note fed into next Design)

        # All iterations done
        best = max(state.iterations, key=lambda r: r.analysis.convergence_score)
        state.lead_compound = best.analysis.best_compound
        state.status        = CampaignStatus.COMPLETE
        yield _event("campaign_complete",
                     lead=state.lead_compound,
                     iterations=len(state.iterations))

    # ── Phase implementations ─────────────────────────────────────────────────

    async def _design(
        self,
        goal: str,
        iteration: int,
        prev: IterationRecord | None,
        override_note: str | None = None,
    ) -> DesignOutput:
        """Call LLM to propose next compound set, grounded in prior SAR.
        
        override_note: optional scientist instruction that overrides the agent's
        own next_steps — injected when the scientist chose 'Edit' at the approval gate.
        """

        if iteration == 0:
            prompt = (
                f'Campaign goal: "{goal}"\n\n'
                f"Iteration 1 — design 4 diverse compounds to begin SAR exploration.\n"
                f'Return JSON: {{"rationale":"str","hypothesis":"str",'
                f'"compounds":[{{"name":"str","scaffold":"str","modification":"str"}}]}}'
            )
        else:
            best_data     = prev.results[:2] if prev else []
            scientist_dir = (
                f"\n\nSCIENTIST OVERRIDE: {override_note}\n"
                f"Prioritise this direction over your own next_steps."
            ) if override_note else ""
            prompt = (
                f'Campaign goal: "{goal}"\n\n'
                f"Iteration {iteration + 1}. "
                f"Previous lead: {prev.analysis.best_compound}. "
                f"Best data: {json.dumps([r.model_dump() for r in best_data])}.\n"
                f"SAR so far: {prev.analysis.sar_trends}"
                f"{scientist_dir}\n\n"
                f"Design 4 next-gen compounds that build on these learnings.\n"
                f'Return JSON: {{"rationale":"str","hypothesis":"str",'
                f'"compounds":[{{"name":"str","scaffold":"str","modification":"str"}}]}}'
            )

        msg = await self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _parse_json(msg.content[0].text)

        return DesignOutput(
            rationale  = raw.get("rationale", ""),
            hypothesis = raw.get("hypothesis", ""),
            compounds  = [CompoundSpec(**c) for c in raw.get("compounds", [])[:4]],
        )

    async def _make(
        self, compounds: list[CompoundSpec], iteration: int
    ) -> list:
        """Dispatch dispensing jobs to Hamilton in parallel."""
        jobs = [
            DispenseJob(
                compound_id   = c.name,
                source_well   = f"A{i+1}",
                dest_plate    = f"P{iteration+1:02d}",
                volume_nl     = 100.0,
            )
            for i, c in enumerate(compounds[:4])
        ]
        import asyncio
        results = await asyncio.gather(*[self.hamilton.dispense(j) for j in jobs])
        return list(results)

    async def _test(
        self, compounds: list[CompoundSpec], iteration: int
    ) -> list[AssayResult]:
        """Run assay panel and return results from LIMS."""
        return await self.assay.run_panel(compounds, iteration)

    async def _analyze(
        self,
        goal: str,
        results: list[AssayResult],
        iteration: int,
    ) -> AnalysisOutput:
        """Call LLM to reason over assay data and produce SAR analysis."""

        prompt = (
            f'Campaign goal: "{goal}"\n\n'
            f"Iteration {iteration + 1} assay results:\n"
            f"{json.dumps([r.model_dump() for r in results], indent=2)}\n\n"
            f"Identify the best compound, key SAR trends, whether the goal is achieved, "
            f"and what to change next cycle.\n"
            f'Return JSON: {{"best_compound":"str","goal_achieved":bool,'
            f'"convergence_score":0-100,"sar_trends":"str","reasoning":"str","next_steps":"str"}}'
        )

        msg = await self.client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYS_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = _parse_json(msg.content[0].text)
        return AnalysisOutput(**raw)
