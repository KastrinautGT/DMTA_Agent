"""
Pydantic models for the DMTA agentic loop.
Strict typing throughout — every instrument payload and LLM response is validated.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ── Enums ─────────────────────────────────────────────────────────────────────

class DMTAPhase(str, Enum):
    DESIGN  = "design"
    MAKE    = "make"
    TEST    = "test"
    ANALYZE = "analyze"


class CampaignStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"


# ── Instrument layer ──────────────────────────────────────────────────────────

class CompoundSpec(BaseModel):
    """A compound proposed by the agent for synthesis."""
    name:         str
    scaffold:     str
    modification: str


class DispenseJob(BaseModel):
    """Payload sent to the Hamilton STAR liquid handler."""
    compound_id:    str
    source_well:    str
    dest_plate:     str
    volume_nl:      float = Field(description="Volume in nanolitres")
    concentration_mm: float = 10.0


class DispenseResult(BaseModel):
    compound_id: str
    success:     bool
    actual_volume_nl: float
    qc_passed:   bool


class AssayResult(BaseModel):
    """Single compound readout from the assay panel."""
    name:     str
    ic50_nm:  float = Field(description="JAK2 IC50 in nM")
    sel_jak1: float = Field(description="Selectivity vs JAK1 (fold)")
    sol_ug_ml: float = Field(description="Kinetic solubility μg/mL")
    hlm_t12_min: float = Field(description="HLM half-life in minutes")
    log_p:    float


# ── Agent layer ───────────────────────────────────────────────────────────────

class DesignOutput(BaseModel):
    """Structured output from the Design agent call."""
    rationale:  str
    hypothesis: str
    compounds:  list[CompoundSpec]


class AnalysisOutput(BaseModel):
    """Structured output from the Analyze agent call."""
    best_compound:      str
    goal_achieved:      bool
    convergence_score:  int = Field(ge=0, le=100)
    sar_trends:         str
    reasoning:          str
    next_steps:         str


# ── Campaign state ────────────────────────────────────────────────────────────

class IterationRecord(BaseModel):
    iteration:  int
    design:     DesignOutput
    results:    list[AssayResult]
    analysis:   AnalysisOutput


class CampaignRequest(BaseModel):
    goal:         str = Field(description="Drug discovery campaign goal")
    max_iters:    int = Field(default=3, ge=1, le=5)
    target:       str = Field(default="JAK2")


class ApprovalDecision(str, Enum):
    APPROVE = "approve"   # run next iteration as proposed
    EDIT    = "edit"      # run next iteration with scientist's override note
    STOP    = "stop"      # end campaign, advance current lead


class ApprovalRequest(BaseModel):
    """Posted by the scientist to unblock the agent."""
    decision:      ApprovalDecision
    override_note: Optional[str] = None   # used when decision == EDIT


class CampaignState(BaseModel):
    campaign_id:   str
    goal:          str
    status:        CampaignStatus = CampaignStatus.PENDING
    iterations:    list[IterationRecord] = []
    lead_compound: Optional[str] = None
    awaiting_approval: bool = False
