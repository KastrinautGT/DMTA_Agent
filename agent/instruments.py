"""
Instrument abstraction layer.

Each class here wraps a physical instrument's SDK behind a clean async interface.
Swap the mock implementations for real SDK calls (e.g. Hamilton VENUS, Tecan
FluentControl, OpenTrons Python API) without touching the agent logic above.
"""

from __future__ import annotations
import asyncio, random, logging
from agent.models import CompoundSpec, DispenseJob, DispenseResult, AssayResult

log = logging.getLogger(__name__)


# ── Hamilton STAR liquid handler ──────────────────────────────────────────────

class HamiltonSTAR:
    """
    Async wrapper around the Hamilton VENUS SDK.

    Production:  replace _dispense_real() body with venus.execute_method(job)
    Mock:        simulates timing + realistic error rates
    """

    def __init__(self, host: str = "localhost", port: int = 7000, mock: bool = True):
        self.host = host
        self.port = port
        self.mock = mock
        log.info(f"HamiltonSTAR init — {'MOCK' if mock else f'{host}:{port}'}")

    async def dispense(self, job: DispenseJob) -> DispenseResult:
        if self.mock:
            return await self._dispense_mock(job)
        return await self._dispense_real(job)

    async def _dispense_mock(self, job: DispenseJob) -> DispenseResult:
        await asyncio.sleep(0.3)          # simulate instrument latency
        jitter      = random.uniform(0.96, 1.04)
        qc_passed   = random.random() > 0.03   # 3% QC fail rate
        actual_vol  = round(job.volume_nl * jitter, 2)
        log.debug(f"  Hamilton → {job.compound_id} {actual_vol} nL → {job.dest_plate}:{job.source_well}")
        return DispenseResult(
            compound_id=job.compound_id,
            success=True,
            actual_volume_nl=actual_vol,
            qc_passed=qc_passed,
        )

    async def _dispense_real(self, job: DispenseJob) -> DispenseResult:
        # Replace with: import venus; result = venus.execute(...)
        raise NotImplementedError("Wire to VENUS SDK here")


# ── Analytical instruments (TR-FRET plate reader + LCMS) ─────────────────────

class AssayStation:
    """
    Runs the JAK2 TR-FRET biochemical assay + selectivity / ADMET panel.
    
    Production: POST to LIMS endpoint or call instrument driver directly.
    Mock: returns statistically realistic data that converges toward the goal.
    """

    def __init__(self, lims_url: str = "http://lims.internal", mock: bool = True):
        self.lims_url = lims_url
        self.mock     = mock
        log.info(f"AssayStation init — {'MOCK' if mock else lims_url}")

    async def run_panel(
        self,
        compounds: list[CompoundSpec],
        iteration: int,
    ) -> list[AssayResult]:
        """Run full assay panel, return one AssayResult per compound."""
        if self.mock:
            return await self._mock_panel(compounds, iteration)
        return await self._real_panel(compounds)

    async def _mock_panel(
        self, compounds: list[CompoundSpec], iteration: int
    ) -> list[AssayResult]:
        await asyncio.sleep(0.5)   # simulate plate reader time

        results = []
        boost   = 1 / (iteration + 1)        # potency improves each cycle

        for i, cpd in enumerate(compounds[:4]):
            lead = (i == 0)
            results.append(AssayResult(
                name       = cpd.name,
                ic50_nm    = round(max(0.1, 80 * boost * random.uniform(0.3, 0.7) * (0.45 if lead else 1.1)), 2),
                sel_jak1   = round(min(800, 4 * (iteration + 1) * random.uniform(1.2, 2.0) * (1.6 if lead else 0.75))),
                sol_ug_ml  = round(max(8, 12 + iteration * 22 + random.uniform(0, 14))),
                hlm_t12_min= round(max(12, 18 + iteration * 24 + random.uniform(0, 12))),
                log_p      = round(random.uniform(1.6, 4.0), 2),
            ))
        return results

    async def _real_panel(self, compounds: list[CompoundSpec]) -> list[AssayResult]:
        # Replace with: response = await httpx.post(f"{self.lims_url}/assay", ...)
        raise NotImplementedError("Wire to LIMS here")
