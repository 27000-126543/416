"""
FastAPI REST API server for the Climate Simulation Platform.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Body
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


class PipelineValidateRequest(BaseModel):
    variable_name: str
    data: List[List[float]]
    latitudes: List[float]
    longitudes: List[float]
    unit: Optional[str] = None


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
            "/qc/history",
            "/ingestion/pipeline/validate",
            "/ingestion/pipeline/summaries",
            "/ingestion/dashboard",
            "/ingestion/sources/{source_id}/start",
            "/ingestion/sources/{source_id}/stop",
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
            "usage_summary": tenant.get_usage_summary(),
        })

    return {"tenants": tenants, "total": len(tenants)}


@app.get("/tenants/{tenant_id}/usage", tags=["Multi-Tenant"])
async def get_tenant_usage(tenant_id: str):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    usage = platform.tenant_manager.get_tenant_usage(tenant_id)
    if usage is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")
    return usage


@app.post("/tenants/{tenant_id}/allocate", tags=["Multi-Tenant"])
async def allocate_tenant_resources(
    tenant_id: str,
    storage_tb: float = Body(0.0, embed=True),
    compute_hours_month: float = Body(0.0, embed=True),
    concurrent_jobs: int = Body(0, embed=True),
    memory_gb: float = Body(0.0, embed=True),
    gpu_count: int = Body(0, embed=True),
    bandwidth_gbps: float = Body(0.0, embed=True),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    from .multi_tenant.manager import ResourceQuota
    requested = ResourceQuota(
        storage_tb=storage_tb,
        compute_hours_month=compute_hours_month,
        concurrent_jobs=concurrent_jobs,
        memory_gb=memory_gb,
        gpu_count=gpu_count,
        bandwidth_gbps=bandwidth_gbps,
    )

    ok, reasons = platform.tenant_manager.allocate_resources_detailed(tenant_id, requested)
    if not ok:
        return {
            "success": False,
            "tenant_id": tenant_id,
            "requested": {
                "storage_tb": storage_tb,
                "compute_hours_month": compute_hours_month,
                "concurrent_jobs": concurrent_jobs,
                "memory_gb": memory_gb,
                "gpu_count": gpu_count,
                "bandwidth_gbps": bandwidth_gbps,
            },
            "insufficient": reasons,
            "current_usage": platform.tenant_manager.get_tenant_usage(tenant_id),
        }

    return {
        "success": True,
        "tenant_id": tenant_id,
        "requested": {
            "storage_tb": storage_tb,
            "compute_hours_month": compute_hours_month,
            "concurrent_jobs": concurrent_jobs,
            "memory_gb": memory_gb,
            "gpu_count": gpu_count,
            "bandwidth_gbps": bandwidth_gbps,
        },
        "current_usage": platform.tenant_manager.get_tenant_usage(tenant_id),
    }


@app.post("/tenants/{tenant_id}/release", tags=["Multi-Tenant"])
async def release_tenant_resources(
    tenant_id: str,
    storage_tb: float = Body(0.0, embed=True),
    compute_hours_month: float = Body(0.0, embed=True),
    concurrent_jobs: int = Body(0, embed=True),
    memory_gb: float = Body(0.0, embed=True),
    gpu_count: int = Body(0, embed=True),
    bandwidth_gbps: float = Body(0.0, embed=True),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    from .multi_tenant.manager import ResourceQuota
    released = ResourceQuota(
        storage_tb=storage_tb,
        compute_hours_month=compute_hours_month,
        concurrent_jobs=concurrent_jobs,
        memory_gb=memory_gb,
        gpu_count=gpu_count,
        bandwidth_gbps=bandwidth_gbps,
    )

    tenant = platform.tenant_manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

    platform.tenant_manager.release_resources(tenant_id, released)

    return {
        "success": True,
        "tenant_id": tenant_id,
        "released": {
            "storage_tb": storage_tb,
            "compute_hours_month": compute_hours_month,
            "concurrent_jobs": concurrent_jobs,
            "memory_gb": memory_gb,
            "gpu_count": gpu_count,
            "bandwidth_gbps": bandwidth_gbps,
        },
        "current_usage": platform.tenant_manager.get_tenant_usage(tenant_id),
    }


@app.get("/tenants/{tenant_id}/history", tags=["Multi-Tenant"])
async def get_tenant_history(
    tenant_id: str,
    event_type: Optional[str] = Query(None, description="Filter by event type: allocate or release"),
    limit: int = Query(50, ge=1, le=500),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    tenant = platform.tenant_manager.get_tenant(tenant_id)
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

    history = platform.tenant_manager.get_tenant_history(tenant_id, event_type=event_type)
    recent = history[-limit:] if len(history) > limit else history

    events = []
    for evt in reversed(recent):
        events.append({
            "event_id": evt.event_id,
            "event_type": evt.event_type,
            "timestamp": evt.timestamp.isoformat(),
            "resources": evt.resources,
            "reason": evt.reason,
        })

    return {
        "tenant_id": tenant_id,
        "total_events": len(history),
        "events": events,
    }


@app.post("/qc/validate", tags=["Quality Control"])
async def validate_dataset(
    min_pass_rate: float = 0.95,
    data_source: str = "default_model_state",
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        import uuid

        if not platform.coupled_model or not platform.coupled_model.states:
            return {
                "record_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "data_source": data_source,
                "passed": False,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": 0.0,
                "total_points": 0,
                "failed_points": 0,
                "variables": [],
                "variable_details": {},
                "issues": ["No model state available for validation"],
                "checks_summary": {},
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

            issues = [
                f"{f.variable}: {f.check_name} - {f.message}"
                for f in qc_result.failures
            ][:20]

            variables = list(variable_details.keys())

            record_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()

            return {
                "record_id": record_id,
                "timestamp": timestamp,
                "data_source": data_source,
                "passed": passed,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": overall_pass_rate,
                "total_points": total_points,
                "failed_points": failed_points,
                "variables": variables,
                "variable_details": variable_details,
                "issues": issues,
                "checks_summary": variable_details,
                "status": "ok",
            }
        else:
            passed, report = platform.qc_engine.validate(
                states, min_pass_rate=min_pass_rate, data_source=data_source
            )

            return {
                "record_id": report.get("record_id"),
                "timestamp": report.get("timestamp"),
                "data_source": data_source,
                "passed": passed,
                "min_pass_rate": min_pass_rate,
                "overall_pass_rate": report.get("overall_pass_rate", 0),
                "total_points": report.get("total_points", 0),
                "failed_points": report.get("failed_points", 0),
                "variables": report.get("variables", []),
                "variable_details": report.get("variable_details", {}),
                "issues": report.get("issues", [])[:20],
                "checks_summary": report.get("results", {}),
                "status": "ok",
            }
    except Exception as e:
        logger.error(f"QC validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/qc/history", tags=["Quality Control"])
async def get_qc_history(
    data_source: Optional[str] = Query(None, description="Filter by data source"),
    variable: Optional[str] = Query(None, description="Filter by variable name"),
    passed: Optional[bool] = Query(None, description="Filter by pass status"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")

    try:
        start_dt = None
        end_dt = None

        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format, use ISO format")

        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format, use ISO format")

        records = platform.qc_engine.query_history(
            data_source=data_source,
            variable=variable,
            passed=passed,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit,
        )

        summaries = []
        for record in records:
            summaries.append({
                "record_id": record.record_id,
                "data_source": record.data_source,
                "passed": record.passed,
                "pass_rate": record.overall_pass_rate,
                "timestamp": record.timestamp.isoformat(),
                "variables": record.variables,
                "num_issues": len(record.issues),
            })

        return {
            "total": len(summaries),
            "limit": limit,
            "records": summaries,
        }
    except Exception as e:
        logger.error(f"QC history query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingestion/pipeline/validate", tags=["Ingestion Pipeline"])
async def validate_pipeline(
    request: PipelineValidateRequest,
    source_id: str = Query("unknown", description="Source identifier"),
    source_type: str = Query("satellite_remote_sensing", description="Source type"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    if not platform.data_cleaner:
        raise HTTPException(status_code=503, detail="Data cleaner not initialized")
    
    try:
        import numpy as np
        import xarray as xr
        
        data_array = np.array(request.data)
        lat_array = np.array(request.latitudes)
        lon_array = np.array(request.longitudes)
        
        ds = xr.Dataset(
            {
                request.variable_name: (["lat", "lon"], data_array),
            },
            coords={
                "lat": lat_array,
                "lon": lon_array,
            },
        )
        
        if request.unit:
            ds[request.variable_name].attrs["units"] = request.unit
        
        result, summary = platform.data_cleaner.run_qc_with_summary(
            ds, source_id=source_id, source_type=source_type
        )
        
        if platform.ingestion_manager:
            platform.ingestion_manager._qc_summaries.append(summary)
        
        summary_dict = {
            "summary_id": summary.summary_id,
            "source_id": summary.source_id,
            "source_type": summary.source_type,
            "timestamp": summary.timestamp.isoformat(),
            "original_anomaly_points": summary.original_anomaly_points,
            "original_nan_count": summary.original_nan_count,
            "original_total_points": summary.original_total_points,
            "cleaned_nan_count": summary.cleaned_nan_count,
            "cleaning_interpolated_points": summary.cleaning_interpolated_points,
            "modified_points": summary.modified_points,
            "final_pass_rate": summary.final_pass_rate,
            "passed": summary.passed,
            "variable_summaries": summary.variable_summaries,
            "qc_failures_detail": summary.qc_failures_detail,
        }
        
        return {
            "status": "completed",
            "summary": summary_dict,
        }
    except Exception as e:
        logger.error(f"Pipeline validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingestion/pipeline/summaries", tags=["Ingestion Pipeline"])
async def get_pipeline_summaries(
    limit: int = Query(10, ge=1, le=1000, description="Maximum number of summaries to return"),
    source_id: Optional[str] = Query(None, description="Filter by source ID"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    try:
        summaries = []
        if platform.ingestion_manager:
            all_summaries = platform.ingestion_manager.get_qc_summaries()
            if source_id:
                all_summaries = [s for s in all_summaries if s.source_id == source_id]
            
            for summary in all_summaries[-limit:]:
                summaries.append({
                    "summary_id": summary.summary_id,
                    "source_id": summary.source_id,
                    "source_type": summary.source_type,
                    "timestamp": summary.timestamp.isoformat(),
                    "original_anomaly_points": summary.original_anomaly_points,
                    "original_nan_count": summary.original_nan_count,
                    "original_total_points": summary.original_total_points,
                    "cleaned_nan_count": summary.cleaned_nan_count,
                    "cleaning_interpolated_points": summary.cleaning_interpolated_points,
                    "modified_points": summary.modified_points,
                    "final_pass_rate": summary.final_pass_rate,
                    "passed": summary.passed,
                })
        
        return {
            "total": len(summaries),
            "limit": limit,
            "summaries": summaries,
        }
    except Exception as e:
        logger.error(f"Failed to get pipeline summaries: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ingestion/dashboard", tags=["Ingestion Pipeline"])
async def get_ingestion_dashboard():
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    if not platform.ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")
    
    try:
        dashboards = platform.ingestion_manager.get_all_dashboards()
        
        result = {}
        for source_id, dashboard in dashboards.items():
            result[source_id] = {
                "source_id": dashboard["source_id"],
                "source_type": dashboard["source_type"],
                "state": dashboard["state"],
                "session": {
                    "session_start": dashboard["current_session"]["session_start"].isoformat() if dashboard["current_session"]["session_start"] else None,
                    "session_end": dashboard["current_session"]["session_end"].isoformat() if dashboard["current_session"]["session_end"] else None,
                    "session_chunks": dashboard["current_session"]["session_chunks"],
                    "session_bytes": dashboard["current_session"]["session_bytes"],
                    "session_rate_mbps": dashboard["current_session"]["session_rate_mbps"],
                    "last_data_time": dashboard["current_session"]["last_data_time"].isoformat() if dashboard["current_session"]["last_data_time"] else None,
                },
                "cumulative": dashboard["cumulative"],
                "rejection": {
                    "rejected_chunks": dashboard["rejection"]["rejected_chunks"],
                    "rejected_bytes": dashboard["rejection"]["rejected_bytes"],
                    "last_rejected_time": dashboard["rejection"]["last_rejected_time"].isoformat() if dashboard["rejection"]["last_rejected_time"] else None,
                },
                "status": dashboard["status"],
                "last_update": dashboard["last_update"].isoformat() if dashboard["last_update"] else None,
            }
        
        return {
            "total_sources": len(result),
            "sources": result,
        }
    except Exception as e:
        logger.error(f"Failed to get ingestion dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingestion/sources/{source_id}/start", tags=["Ingestion Pipeline"])
async def start_ingestion_source(source_id: str):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    if not platform.ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")
    
    try:
        result = platform.ingestion_manager.start_ingestion(source_id=source_id)
        return {
            "status": "started",
            "source_id": result["source_id"],
            "session_id": result["session_id"],
            "state": result["state"],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start ingestion for {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingestion/sources/{source_id}/stop", tags=["Ingestion Pipeline"])
async def stop_ingestion_source(
    source_id: str,
    reason: str = Query("", description="Reason for stopping"),
):
    if not platform:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    if not platform.ingestion_manager:
        raise HTTPException(status_code=503, detail="Ingestion manager not initialized")
    
    try:
        result = platform.ingestion_manager.stop_ingestion(source_id=source_id, reason=reason)
        return {
            "status": "stopped",
            "source_id": result["source_id"],
            "state": result["state"],
            "session_chunks": result["session_chunks"],
            "session_bytes": result["session_bytes"],
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to stop ingestion for {source_id}: {e}")
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
