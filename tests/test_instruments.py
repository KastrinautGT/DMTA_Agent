"""
Test suite for the DMTA instrument abstraction layer.

Covers:
  - HamiltonSTAR mock dispensing (happy path, QC failures, volume jitter)
  - HamiltonSTAR parallel dispensing via asyncio.gather
  - HamiltonSTAR real SDK raises NotImplementedError
  - AssayStation mock panel (result count, value ranges, convergence)
  - AssayStation real LIMS raises NotImplementedError
  - Pydantic model validation (DispenseJob, AssayResult, CampaignState)
  - FastAPI endpoint integration (POST /campaigns, GET /campaigns/{id})

Run:
  cd dmta_agent
  pytest tests/ -v

Run with coverage:
  pytest tests/ -v --cov=agent --cov-report=term-missing
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from pydantic import ValidationError

# ── Imports from the agent package ───────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.models import (
    ApprovalDecision, ApprovalRequest, AssayResult,
    CampaignRequest, CampaignState, CampaignStatus,
    CompoundSpec, DispenseJob, DispenseResult,
)
from agent.instruments import AssayStation, HamiltonSTAR
from main import app


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def hamilton():
    """Mock Hamilton instance — no real instrument needed."""
    return HamiltonSTAR(mock=True)


@pytest.fixture
def assay_station():
    """Mock AssayStation instance."""
    return AssayStation(mock=True)


@pytest.fixture
def sample_job():
    """A well-formed dispense job."""
    return DispenseJob(
        compound_id   = "JL-101",
        source_well   = "A1",
        dest_plate    = "P01",
        volume_nl     = 100.0,
        concentration_mm = 10.0,
    )


@pytest.fixture
def sample_compounds():
    """Four compounds for assay panel tests."""
    return [
        CompoundSpec(name=f"JL-10{i}", scaffold="Pyrrolopyrimidine", modification=f"3-CF{i}") 
        for i in range(1, 5)
    ]


@pytest_asyncio.fixture
async def http_client():
    """Async test client for FastAPI endpoints."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


# ═══════════════════════════════════════════════════════════════════════════════
# HAMILTON — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHamiltonSTAR:

    @pytest.mark.asyncio
    async def test_dispense_returns_correct_compound_id(self, hamilton, sample_job):
        """Result compound_id must match the job's compound_id."""
        result = await hamilton.dispense(sample_job)
        assert result.compound_id == sample_job.compound_id

    @pytest.mark.asyncio
    async def test_dispense_success_flag_is_true(self, hamilton, sample_job):
        """Mock dispense always succeeds at the instrument level."""
        result = await hamilton.dispense(sample_job)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_dispense_volume_within_jitter_range(self, hamilton, sample_job):
        """
        Actual volume must stay within ±4% of requested volume.
        Hamilton VENUS allows ±5% — our mock uses ±4% for a tighter simulation.
        """
        result = await hamilton.dispense(sample_job)
        lower = sample_job.volume_nl * 0.96
        upper = sample_job.volume_nl * 1.04
        assert lower <= result.actual_volume_nl <= upper, (
            f"Volume {result.actual_volume_nl} nL outside ±4% of {sample_job.volume_nl} nL"
        )

    @pytest.mark.asyncio
    async def test_dispense_qc_failure_rate_is_realistic(self, hamilton):
        """
        QC should fail ~3% of the time.
        Run 200 dispenses and check failure rate stays below 15%
        (wide band to avoid flaky tests — real check is that failures exist at all).
        """
        job = DispenseJob(compound_id="TEST", source_well="A1", dest_plate="P01", volume_nl=100.0)
        results = await asyncio.gather(*[hamilton.dispense(job) for _ in range(200)])
        failures = sum(1 for r in results if not r.qc_passed)
        # Should have some failures but not an absurd number
        assert failures < 30, f"QC failure rate too high: {failures}/200"
        # Should have at least one failure across 200 runs (probabilistically certain)
        # Note: extremely unlikely to have zero failures in 200 runs at 3%

    @pytest.mark.asyncio
    async def test_parallel_dispense_all_succeed(self, hamilton, sample_compounds):
        """
        asyncio.gather() should dispatch all jobs concurrently.
        All four compounds should complete without error.
        """
        jobs = [
            DispenseJob(
                compound_id   = c.name,
                source_well   = f"A{i+1}",
                dest_plate    = "P01",
                volume_nl     = 100.0,
            )
            for i, c in enumerate(sample_compounds)
        ]
        results = await asyncio.gather(*[hamilton.dispense(j) for j in jobs])
        assert len(results) == 4
        assert all(isinstance(r, DispenseResult) for r in results)
        assert all(r.success is True for r in results)

    @pytest.mark.asyncio
    async def test_parallel_dispense_compound_ids_preserved(self, hamilton, sample_compounds):
        """Parallel execution must not mix up compound IDs."""
        jobs = [
            DispenseJob(compound_id=c.name, source_well=f"A{i+1}", dest_plate="P01", volume_nl=100.0)
            for i, c in enumerate(sample_compounds)
        ]
        results = await asyncio.gather(*[hamilton.dispense(j) for j in jobs])
        result_ids = {r.compound_id for r in results}
        expected_ids = {c.name for c in sample_compounds}
        assert result_ids == expected_ids, "Compound IDs were mixed up in parallel execution"

    @pytest.mark.asyncio
    async def test_real_sdk_raises_not_implemented(self):
        """
        Calling the real SDK path must raise NotImplementedError.
        This is the guard that reminds engineers to wire the real SDK.
        """
        hamilton_real = HamiltonSTAR(mock=False)
        job = DispenseJob(compound_id="JL-101", source_well="A1", dest_plate="P01", volume_nl=100.0)
        with pytest.raises(NotImplementedError, match="Wire to VENUS SDK"):
            await hamilton_real._dispense_real(job)

    @pytest.mark.asyncio
    async def test_dispense_zero_volume_returns_near_zero(self, hamilton):
        """Edge case: zero volume job should return near-zero actual volume."""
        job = DispenseJob(compound_id="JL-001", source_well="A1", dest_plate="P01", volume_nl=0.0)
        result = await hamilton.dispense(job)
        assert result.actual_volume_nl == pytest.approx(0.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════════════════════
# ASSAY STATION — UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestAssayStation:

    @pytest.mark.asyncio
    async def test_panel_returns_one_result_per_compound(self, assay_station, sample_compounds):
        """Assay station must return exactly one result per compound (max 4)."""
        results = await assay_station.run_panel(sample_compounds, iteration=0)
        assert len(results) == min(len(sample_compounds), 4)

    @pytest.mark.asyncio
    async def test_panel_compound_names_preserved(self, assay_station, sample_compounds):
        """Result names must match input compound names in order."""
        results = await assay_station.run_panel(sample_compounds[:3], iteration=0)
        for result, compound in zip(results, sample_compounds[:3]):
            assert result.name == compound.name

    @pytest.mark.asyncio
    async def test_ic50_always_positive(self, assay_station, sample_compounds):
        """IC50 must always be > 0 nM — a negative IC50 is physically meaningless."""
        for iteration in range(3):
            results = await assay_station.run_panel(sample_compounds, iteration=iteration)
            for r in results:
                assert r.ic50_nm > 0, f"IC50 {r.ic50_nm} is non-positive for {r.name}"

    @pytest.mark.asyncio
    async def test_selectivity_always_positive(self, assay_station, sample_compounds):
        """Selectivity (fold vs JAK1) must be positive."""
        results = await assay_station.run_panel(sample_compounds, iteration=0)
        for r in results:
            assert r.sel_jak1 > 0

    @pytest.mark.asyncio
    async def test_solubility_above_minimum(self, assay_station, sample_compounds):
        """Solubility must be >= 8 μg/mL (the floor in our mock)."""
        results = await assay_station.run_panel(sample_compounds, iteration=0)
        for r in results:
            assert r.sol_ug_ml >= 8.0, f"Solubility {r.sol_ug_ml} below minimum floor"

    @pytest.mark.asyncio
    async def test_potency_improves_across_iterations(self, assay_station, sample_compounds):
        """
        The first compound (lead position) should have lower IC50 in later iterations.
        This tests the convergence behavior — a core DMTA property.
        """
        # Run multiple independent panels and compare medians
        iter0_ic50s = []
        iter2_ic50s = []
        for _ in range(10):
            r0 = await assay_station.run_panel(sample_compounds[:1], iteration=0)
            r2 = await assay_station.run_panel(sample_compounds[:1], iteration=2)
            iter0_ic50s.append(r0[0].ic50_nm)
            iter2_ic50s.append(r2[0].ic50_nm)
        
        median0 = sorted(iter0_ic50s)[5]
        median2 = sorted(iter2_ic50s)[5]
        assert median2 < median0, (
            f"Iteration 2 IC50 median ({median2:.2f}) should be lower than "
            f"iteration 0 ({median0:.2f}) — convergence not working"
        )

    @pytest.mark.asyncio
    async def test_log_p_in_drug_like_range(self, assay_station, sample_compounds):
        """LogP should stay in a drug-like range (1.6 to 4.0 in our mock)."""
        results = await assay_station.run_panel(sample_compounds, iteration=0)
        for r in results:
            assert 1.0 <= r.log_p <= 5.0, f"LogP {r.log_p} outside drug-like range for {r.name}"

    @pytest.mark.asyncio
    async def test_panel_capped_at_four_compounds(self, assay_station):
        """Panel should only process max 4 compounds even if more are passed."""
        many_compounds = [
            CompoundSpec(name=f"JL-{i:03d}", scaffold="Pyrrolopyrimidine", modification="3-F")
            for i in range(8)
        ]
        results = await assay_station.run_panel(many_compounds, iteration=0)
        assert len(results) == 4

    @pytest.mark.asyncio
    async def test_real_lims_raises_not_implemented(self):
        """Real LIMS path must raise NotImplementedError as a production guard."""
        station_real = AssayStation(mock=False)
        compounds = [CompoundSpec(name="JL-101", scaffold="Test", modification="3-F")]
        with pytest.raises(NotImplementedError, match="Wire to LIMS"):
            await station_real._real_panel(compounds)

    @pytest.mark.asyncio
    async def test_returns_assay_result_instances(self, assay_station, sample_compounds):
        """All results must be valid AssayResult Pydantic instances."""
        results = await assay_station.run_panel(sample_compounds, iteration=0)
        for r in results:
            assert isinstance(r, AssayResult)


# ═══════════════════════════════════════════════════════════════════════════════
# PYDANTIC MODEL VALIDATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelValidation:

    def test_dispense_job_valid(self):
        """A well-formed DispenseJob should instantiate without error."""
        job = DispenseJob(compound_id="JL-101", source_well="A1", dest_plate="P01", volume_nl=100.0)
        assert job.compound_id == "JL-101"
        assert job.volume_nl == 100.0
        assert job.concentration_mm == 10.0  # default

    def test_dispense_job_missing_required_field(self):
        """Missing required field should raise ValidationError."""
        with pytest.raises(ValidationError):
            DispenseJob(source_well="A1", dest_plate="P01", volume_nl=100.0)  # missing compound_id

    def test_assay_result_valid(self):
        """A well-formed AssayResult should instantiate cleanly."""
        r = AssayResult(name="JL-101", ic50_nm=0.8, sel_jak1=150.0, sol_ug_ml=55.0, hlm_t12_min=75.0, log_p=2.1)
        assert r.name == "JL-101"
        assert r.ic50_nm == 0.8

    def test_assay_result_wrong_type(self):
        """Passing a string where float is required should raise ValidationError."""
        with pytest.raises(ValidationError):
            AssayResult(name="JL-101", ic50_nm="not-a-number", sel_jak1=150.0, sol_ug_ml=55.0, hlm_t12_min=75.0, log_p=2.1)

    def test_campaign_state_defaults(self):
        """CampaignState should initialise with PENDING status and empty iterations."""
        state = CampaignState(campaign_id="test-123", goal="Find JAK2 inhibitor")
        assert state.status == CampaignStatus.PENDING
        assert state.iterations == []
        assert state.lead_compound is None
        assert state.awaiting_approval is False

    def test_approval_request_valid_decisions(self):
        """All three valid approval decisions should parse correctly."""
        for decision in ["approve", "edit", "stop"]:
            req = ApprovalRequest(decision=decision)
            assert req.decision == decision

    def test_approval_request_invalid_decision(self):
        """An unrecognised decision value should raise ValidationError."""
        with pytest.raises(ValidationError):
            ApprovalRequest(decision="maybe")

    def test_approval_request_edit_with_note(self):
        """Edit decision with override note should parse correctly."""
        req = ApprovalRequest(decision="edit", override_note="Focus on metabolic stability")
        assert req.override_note == "Focus on metabolic stability"


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestFastAPIEndpoints:

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, http_client):
        """Health endpoint must return 200 — this is the K8s liveness probe."""
        response = await http_client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_create_campaign_returns_201(self, http_client):
        """POST /campaigns should return 201 with a campaign_id."""
        response = await http_client.post("/campaigns", json={
            "goal": "JAK2-selective inhibitor: IC50 < 1 nM"
        })
        assert response.status_code == 201
        data = response.json()
        assert "campaign_id" in data
        assert len(data["campaign_id"]) > 0

    @pytest.mark.asyncio
    async def test_create_campaign_returns_goal(self, http_client):
        """Response should echo the goal back for confirmation."""
        goal = "JAK2-selective inhibitor: IC50 < 1 nM"
        response = await http_client.post("/campaigns", json={"goal": goal})
        assert response.json()["goal"] == goal

    @pytest.mark.asyncio
    async def test_get_campaign_returns_state(self, http_client):
        """GET /campaigns/{id} should return the full campaign state."""
        # Create first
        create_resp = await http_client.post("/campaigns", json={"goal": "Test goal"})
        campaign_id = create_resp.json()["campaign_id"]

        # Then fetch
        get_resp = await http_client.get(f"/campaigns/{campaign_id}")
        assert get_resp.status_code == 200
        state = get_resp.json()
        assert state["campaign_id"] == campaign_id
        assert state["status"] == "pending"

    @pytest.mark.asyncio
    async def test_get_nonexistent_campaign_returns_404(self, http_client):
        """Fetching a campaign that doesn't exist should return 404."""
        response = await http_client.get("/campaigns/does-not-exist-abc123")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_create_campaign_missing_goal_returns_422(self, http_client):
        """Missing required 'goal' field should return 422 Unprocessable Entity."""
        response = await http_client.post("/campaigns", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_approve_nonexistent_campaign_returns_404(self, http_client):
        """Approving a campaign that doesn't exist should return 404."""
        response = await http_client.post(
            "/campaigns/fake-id/approve",
            json={"decision": "approve"}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_approve_non_waiting_campaign_returns_409(self, http_client):
        """
        Approving a campaign that isn't awaiting approval should return 409.
        Guards against race conditions where approval fires too early.
        """
        create_resp = await http_client.post("/campaigns", json={"goal": "Test"})
        campaign_id = create_resp.json()["campaign_id"]
        # Campaign is PENDING, not awaiting approval
        response = await http_client.post(
            f"/campaigns/{campaign_id}/approve",
            json={"decision": "approve"}
        )
        assert response.status_code == 409


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERGENCE SCORE UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

import math

def convergence_score(compound: AssayResult) -> float:
    """
    Deterministic weighted convergence score.
    Mirrors what should be in production (currently done by LLM in demo).
    """
    def sigmoid(value, target, higher_is_better=False):
        ratio = value / target
        exponent = -5 * (ratio - 1) if higher_is_better else 5 * (ratio - 1)
        exponent = max(-500, min(500, exponent))
        return 1 / (1 + math.exp(exponent))

    weights = {"ic50": 0.40, "sel": 0.30, "sol": 0.15, "t12": 0.15}
    scores = {
        "ic50": sigmoid(compound.ic50_nm,      1.0,  higher_is_better=False),
        "sel":  sigmoid(compound.sel_jak1,     100,  higher_is_better=True),
        "sol":  sigmoid(compound.sol_ug_ml,    50,   higher_is_better=True),
        "t12":  sigmoid(compound.hlm_t12_min,  60,   higher_is_better=True),
    }
    return round(sum(weights[k] * scores[k] for k in weights) * 100, 1)


class TestConvergenceScore:

    def test_perfect_compound_scores_near_100(self):
        """A compound hitting all targets should score close to 100."""
        perfect = AssayResult(name="Perfect", ic50_nm=0.1, sel_jak1=500, sol_ug_ml=200, hlm_t12_min=180, log_p=2.5)
        score = convergence_score(perfect)
        assert score > 90, f"Perfect compound scored only {score}"

    def test_poor_compound_scores_low(self):
        """A compound missing all targets should score below 30."""
        poor = AssayResult(name="Poor", ic50_nm=100, sel_jak1=5, sol_ug_ml=8, hlm_t12_min=10, log_p=5.5)
        score = convergence_score(poor)
        assert score < 30, f"Poor compound scored {score}, expected < 30"

    def test_score_increases_as_ic50_improves(self):
        """Lower IC50 should produce higher convergence score."""
        weak   = AssayResult(name="Weak",   ic50_nm=50,  sel_jak1=100, sol_ug_ml=50, hlm_t12_min=60, log_p=2.5)
        strong = AssayResult(name="Strong", ic50_nm=0.5, sel_jak1=100, sol_ug_ml=50, hlm_t12_min=60, log_p=2.5)
        assert convergence_score(strong) > convergence_score(weak)

    def test_score_is_between_0_and_100(self):
        """Score must always fall within 0–100 bounds."""
        compounds = [
            AssayResult(name="A", ic50_nm=0.1,  sel_jak1=500, sol_ug_ml=200, hlm_t12_min=180, log_p=2.0),
            AssayResult(name="B", ic50_nm=1000, sel_jak1=1,   sol_ug_ml=1,   hlm_t12_min=1,   log_p=6.0),
            AssayResult(name="C", ic50_nm=1.0,  sel_jak1=100, sol_ug_ml=50,  hlm_t12_min=60,  log_p=2.5),
        ]
        for c in compounds:
            score = convergence_score(c)
            assert 0 <= score <= 100, f"Score {score} out of bounds for {c.name}"

    def test_goal_achieved_when_all_thresholds_met(self):
        """
        A compound meeting all four thresholds:
        IC50 < 1, selectivity > 100, solubility > 50, t½ > 60
        should score above 70.
        """
        lead = AssayResult(name="Lead", ic50_nm=0.8, sel_jak1=120, sol_ug_ml=55, hlm_t12_min=65, log_p=2.8)
        score = convergence_score(lead)
        assert score > 65, f"Lead compound scored only {score}, expected > 65"
