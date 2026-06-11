"""
FastAPI REST API server for the Climate Simulation Platform.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

from .main import ClimateSimulationPlatform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("climate_api")

app = FastAPI(
    title="Global Climate Simulation Platform API",
    description="REST API for large-scale climate simulation and analysis",
    version="1.0.0",
)

platform: Optional[ClimateSimulationPlatform] = None


class SimulationRequest(BaseModel):
    start_time: Optional[str] = Field(None, description="Start time ISO format")
    duration_days: int = Field(1, ge=1, le=36500, description="Simulation duration in days")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for sandboxed execution")
    with_ensemble: bool = Field(False, description="Run ensemble forecast")
    ensemble_size: Optional[int] = Field(None, ge=2, le=1000, description="Ensemble member count")


class AssimilationRequest(BaseModel):
    observations: List[Dict[str, Any]] = Field(..., description="List of observation dictionaries")


class StatusResponse(BaseModel):
    initialized: bool
    version: str
    system_name: str
    tenants: int
    registered_sources: int
    active_workers: int
    timestamp: str


@app.on_event("startup")
async def startup_event():
    global platform
    platform = ClimateSimulationPlatform()
    platform.initialize()
    logger.info("API server initialized")


@app.get("/", tags=["System"])
async def root():
    return {
        "service": "Global Climate Simulation Platform",
        "version": "1.0.0",
        "status": "running",
        "endpoints": [
            "/status",
            "/simulation",
            "/assimilation",
            "/visualization",
            "/fingerprint/search",
            "/tenants",
            "/qc/validate",
        ],
    }


@app.get("/status", response_model=StatusResponse, tags=["System"])
async def get_status():
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    status = platform.get_status()
    status["timestamp"] = datetime.now().isoformat()
    return status


@app.post("/simulation", tags=["Simulation"])
async def run_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    st = None
    if request.start_time:
        try:
            st = datetime.fromisoformat(request.start_time)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_time format, use ISO format")

    duration = timedelta(days=request.duration_days)

    try:
        result, qc_report = platform.run_simulation(
            start_time=st,
            duration=duration,
            tenant_id=request.tenant_id,
            with_ensemble=request.with_ensemble,
            ensemble_size=request.ensemble_size,
        )

        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "duration_days": request.duration_days,
            "variables": list(result.data_vars.keys()),
            "qc_pass_rate": qc_report.get("overall_pass_rate", 0),
            "qc_passed": qc_report.get("passed", False),
            "qc_issues": qc_report.get("issues", [])[:10],
        }
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/assimilation", tags=["Data Assimilation"])
async def run_assimilation(request: AssimilationRequest):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        from .assimilation import Observation

        observations = []
        for obs_dict in request.observations:
            obs = Observation(
                value=float(obs_dict["value"]),
                error=float(obs_dict.get("error", 1.0)),
                latitude=float(obs_dict["latitude"]),
                longitude=float(obs_dict["longitude"]),
                altitude=float(obs_dict["altitude"]) if "altitude" in obs_dict else None,
                timestamp=datetime.fromisoformat(obs_dict["timestamp"]) if "timestamp" in obs_dict else None,
                variable=obs_dict.get("variable", ""),
                observation_type=obs_dict.get("observation_type", ""),
            )
            observations.append(obs)

        lat = np.linspace(-90, 90, 73)
        lon = np.linspace(0, 358.75, 288)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        temp_bg = 288 - 30 * np.abs(np.sin(np.radians(lat_grid)))
        background = xr.Dataset(
            {
                "temperature": (["lat", "lon"], temp_bg),
                "pressure": (["lat", "lon"], 101325 * np.ones_like(temp_bg)),
            },
            coords={"lat": lat, "lon": lon},
        )

        result = platform.run_data_assimilation_cycle(background, observations)

        return {
            "status": "completed",
            "method": result.method,
            "computation_time_seconds": result.computation_time,
            "observations_assimilated": result.observations_assimilated,
            "observations_rejected": result.observations_rejected,
            "analysis_variables": list(result.analysis.data_vars.keys()),
        }
    except Exception as e:
        logger.error(f"Assimilation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualization", tags=["Visualization"])
async def get_visualizations(
    variables: Optional[str] = Query(None, description="Comma-separated list of variables"),
    format: str = Query("json", description="Output format: json, html, png"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        if not platform.coupled_model or not platform.coupled_model.states:
            return {
                "visualizations": [],
                "total": 0,
                "status": "no_data",
                "message": "No model state available. Run a simulation first or use default initialization.",
                "available_components": list(platform.coupled_model.components.keys()) if platform.coupled_model else [],
            }

        states = platform.coupled_model.get_combined_state()
        var_list = variables.split(",") if variables else None

        viz_results = platform.generate_visualizations(
            states, variables=var_list
        )

        outputs = []
        for name, output in viz_results.items():
            outputs.append({
                "name": name,
                "format": output.format.value,
                "title": output.title,
                "variables": output.variables,
                "metadata": output.metadata,
            })

        return {
            "visualizations": outputs,
            "total": len(outputs),
            "status": "ok",
        }
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/fingerprint/search", tags=["Climate Fingerprint"])
async def search_similar_events(
    top_k: int = Query(10, ge=1, le=100, description="Number of similar events to return"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        states = platform.coupled_model.get_combined_state()
        results = platform.find_similar_climate_events(states, top_k=top_k)

        formatted = []
        for r in results:
            formatted.append({
                "matched_event_id": r.matched_event.event_id,
                "matched_event_description": r.matched_event.description,
                "similarity_score": r.similarity_score,
                "temporal_similarity": r.temporal_similarity,
                "spatial_similarity": r.spatial_similarity,
                "start_time": r.matched_event.start_time.isoformat(),
                "end_time": r.matched_event.end_time.isoformat(),
            })

        return {
            "query_time": datetime.now().isoformat(),
            "top_k": top_k,
            "results": formatted,
        }
    except Exception as e:
        logger.error(f"Fingerprint search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tenants", tags=["Multi-Tenant"])
async def list_tenants():
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    tenants = []
    for tenant in platform.tenant_manager.list_tenants():
        tenants.append({
            "tenant_id": tenant.tenant_id,
            "name": tenant.name,
            "role": tenant.role.value,
            "active": tenant.is_active,
            "users": len(tenant.users),
            "sandboxes": len(tenant.sandboxes),
            "workspaces": len(tenant.workspaces),
            "quota_utilization": tenant.get_quota_utilization(),
        })

    return {"tenants": tenants, "total": len(tenants)}


@app.post("/qc/validate", tags=["Quality Control"])
async def validate_dataset(
    min_pass_rate: float = 0.95,
    data_source: str = "default_model_state",
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        if not platform.coupled_model or not platform.coupled_model.states:
            return {
                "passed": False,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": 0.0,
                "total_points": 0,
                "failed_points": 0,
                "issues": ["No model state available for validation"],
                "checks_summary": {},
                "data_source": data_source,
                "status": "no_data",
                "message": "No model state available. Run a simulation first or use default initialization.",
            }

        states = platform.coupled_model.get_combined_state()

        if data_source == "default_model_state" and platform.data_cleaner is not None:
            qc_result = platform.data_cleaner.run_qc(states)
            variable_details = qc_result.variable_details
            all_pass_rates = [
                details["pass_rate"]
                for details in variable_details.values()
            ]
            overall_pass_rate = float(np.mean(all_pass_rates)) if all_pass_rates else 1.0
            passed = all(
                pr >= min_pass_rate for pr in all_pass_rates
            ) if all_pass_rates else False

            total_points = sum(
                details["total_points"] for details in variable_details.values()
            )
            failed_points = sum(
                details["failed_points"] for details in variable_details.values()
            )

            checks_summary = {}
            for var_name, details in variable_details.items():
                checks_summary[var_name] = details

            return {
                "passed": passed,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": overall_pass_rate,
                "total_points": total_points,
                "failed_points": failed_points,
                "issues": [
                    f"{f.variable}: {f.check_name} - {f.message}"
                    for f in qc_result.failures
                ][:20],
                "checks_summary": checks_summary,
                "data_source": data_source,
                "status": "ok",
            }
        else:
            passed, report = platform.qc_engine.validate(states, min_pass_rate=min_pass_rate)

            return {
                "passed": passed,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": report.get("overall_pass_rate", 0),
                "total_points": report.get("total_points", 0),
                "failed_points": report.get("failed_points", 0),
                "issues": report.get("issues", [])[:20],
                "checks_summary": report.get("results", {}),
                "data_source": data_source,
                "status": "ok",
            }
    except Exception as e:
        logger.error(f"QC validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/diagnostics", tags=["System"])
async def system_diagnostics():
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        return {
            "platform_status": platform.get_status(),
            "tenant_manager_status": platform.tenant_manager.get_system_status(),
            "adaptive_mesh_stats": (
                platform.adaptive_mesh.get_mesh_statistics() if platform.adaptive_mesh else None
            ),
            "ensemble_info": {
                "ensemble_size": (
                    platform.ensemble_forecast.ensemble_size if platform.ensemble_forecast else None
                ),
                "effective_size": (
                    platform.ensemble_forecast.effective_ensemble_size if platform.ensemble_forecast else None
                ),
            } if platform.ensemble_forecast else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def start_api_server(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    start_api_server()
