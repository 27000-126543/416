"""
Multi-tenant management with data sandboxes, permissions, and dynamic scaling.
"""

import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import psutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    STORAGE = "storage"
    COMPUTE = "compute"
    MEMORY = "memory"
    NETWORK = "network"
    GPU = "gpu"
    DATASET = "dataset"
    WORKFLOW = "workflow"
    RESULT = "result"


class TenantRole(Enum):
    ADMIN = "admin"
    OPERATIONAL = "operational"
    RESEARCH = "research"
    ACADEMIC = "academic"
    VIEWER = "viewer"


class Permission(Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    SHARE = "share"
    ADMIN = "admin"


@dataclass
class ResourceQuota:
    storage_tb: float = 0.0
    compute_hours_month: float = 0.0
    concurrent_jobs: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    bandwidth_gbps: float = 0.0

    @classmethod
    def default_research_quota(cls) -> "ResourceQuota":
        return cls(
            storage_tb=10.0,
            compute_hours_month=1000.0,
            concurrent_jobs=10,
            memory_gb=128.0,
            gpu_count=0,
            bandwidth_gbps=1.0,
        )

    def exceeds(self, usage: "ResourceQuota") -> Tuple[bool, List[str]]:
        exceeded = []
        if usage.storage_tb > self.storage_tb:
            exceeded.append(f"storage: {usage.storage_tb:.2f}TB > {self.storage_tb:.2f}TB")
        if usage.compute_hours_month > self.compute_hours_month:
            exceeded.append(f"compute: {usage.compute_hours_month:.1f}h > {self.compute_hours_month:.1f}h")
        if usage.concurrent_jobs > self.concurrent_jobs:
            exceeded.append(f"concurrent jobs: {usage.concurrent_jobs} > {self.concurrent_jobs}")
        if usage.memory_gb > self.memory_gb:
            exceeded.append(f"memory: {usage.memory_gb:.1f}GB > {self.memory_gb:.1f}GB")
        if usage.gpu_count > self.gpu_count:
            exceeded.append(f"GPU: {usage.gpu_count} > {self.gpu_count}")
        if usage.bandwidth_gbps > self.bandwidth_gbps:
            exceeded.append(f"bandwidth: {usage.bandwidth_gbps:.2f}Gbps > {self.bandwidth_gbps:.2f}Gbps")
        return len(exceeded) > 0, exceeded

    def utilization_pct(self, usage: "ResourceQuota") -> Dict[str, float]:
        return {
            "storage": min(100.0, usage.storage_tb / max(self.storage_tb, 1e-9) * 100),
            "compute": min(100.0, usage.compute_hours_month / max(self.compute_hours_month, 1e-9) * 100),
            "concurrent_jobs": min(100.0, usage.concurrent_jobs / max(self.concurrent_jobs, 1e-9) * 100),
            "memory": min(100.0, usage.memory_gb / max(self.memory_gb, 1e-9) * 100),
            "gpu": min(100.0, usage.gpu_count / max(self.gpu_count, 1) * 100),
            "bandwidth": min(100.0, usage.bandwidth_gbps / max(self.bandwidth_gbps, 1e-9) * 100),
        }


@dataclass
class ResourceUsage:
    tenant_id: str
    storage_tb: float = 0.0
    compute_hours_month: float = 0.0
    concurrent_jobs: int = 0
    memory_gb: float = 0.0
    gpu_count: int = 0
    bandwidth_gbps: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

    def to_quota(self) -> ResourceQuota:
        return ResourceQuota(
            storage_tb=self.storage_tb,
            compute_hours_month=self.compute_hours_month,
            concurrent_jobs=self.concurrent_jobs,
            memory_gb=self.memory_gb,
            gpu_count=self.gpu_count,
            bandwidth_gbps=self.bandwidth_gbps,
        )


@dataclass
class User:
    user_id: str
    username: str
    email: str
    tenant_id: str
    role: TenantRole = TenantRole.VIEWER
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_permission(self, perm: Permission) -> bool:
        if self.role == TenantRole.ADMIN:
            return True
        return perm in self.permissions

    def add_permission(self, perm: Permission):
        self.permissions.add(perm)

    def remove_permission(self, perm: Permission):
        self.permissions.discard(perm)


@dataclass
class AccessControlEntry:
    resource_id: str
    resource_type: ResourceType
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    permissions: Set[Permission] = field(default_factory=set)
    granted_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None


class AccessControlList:
    def __init__(self):
        self._entries: Dict[str, AccessControlEntry] = {}

    def grant(
        self,
        resource_id: str,
        resource_type: ResourceType,
        permissions: Set[Permission],
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ):
        entry_id = self._entry_id(resource_id, user_id, tenant_id)
        entry = AccessControlEntry(
            resource_id=resource_id,
            resource_type=resource_type,
            user_id=user_id,
            tenant_id=tenant_id,
            permissions=permissions,
            expires_at=expires_at,
        )
        self._entries[entry_id] = entry

    def revoke(
        self,
        resource_id: str,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        entry_id = self._entry_id(resource_id, user_id, tenant_id)
        if entry_id in self._entries:
            del self._entries[entry_id]

    def check(
        self,
        resource_id: str,
        permission: Permission,
        user: Optional[User] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        now = datetime.now()

        user_entry_id = self._entry_id(resource_id, user.user_id if user else None, None)
        if user_entry_id in self._entries:
            entry = self._entries[user_entry_id]
            if entry.expires_at is None or entry.expires_at > now:
                if permission in entry.permissions:
                    return True

        if user and user.role == TenantRole.ADMIN:
            return True

        tenant_entry_id = self._entry_id(resource_id, None, tenant_id or (user.tenant_id if user else None))
        if tenant_entry_id in self._entries:
            entry = self._entries[tenant_entry_id]
            if entry.expires_at is None or entry.expires_at > now:
                if permission in entry.permissions:
                    return True

        if user and permission == Permission.READ:
            user_tenant_id = user.tenant_id
            public_entry_id = self._entry_id(resource_id, None, "*")
            if public_entry_id in self._entries:
                entry = self._entries[public_entry_id]
                if permission in entry.permissions:
                    return True

        return False

    def list_permissions(self, resource_id: str) -> List[AccessControlEntry]:
        return [e for e in self._entries.values() if e.resource_id == resource_id]

    @staticmethod
    def _entry_id(resource_id: str, user_id: Optional[str], tenant_id: Optional[str]) -> str:
        return f"{resource_id}:{user_id or 'none'}:{tenant_id or 'none'}"


@dataclass
class DataSandbox:
    sandbox_id: str
    tenant_id: str
    name: str
    base_path: str
    storage_quota_tb: float = 10.0
    is_isolated: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    collaborators: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        return Path(self.base_path) / self.sandbox_id / "data"

    @property
    def scratch_path(self) -> Path:
        return Path(self.base_path) / self.sandbox_id / "scratch"

    @property
    def results_path(self) -> Path:
        return Path(self.base_path) / self.sandbox_id / "results"

    def initialize(self):
        for p in [self.data_path, self.scratch_path, self.results_path]:
            p.mkdir(parents=True, exist_ok=True)

    def get_storage_used_tb(self) -> float:
        total = 0.0
        base = Path(self.base_path) / self.sandbox_id
        if base.exists():
            for item in base.rglob("*"):
                if item.is_file():
                    try:
                        total += item.stat().st_size
                    except OSError:
                        pass
        return total / (1024 ** 4)

    def add_collaborator(self, tenant_id: str):
        self.collaborators.add(tenant_id)

    def remove_collaborator(self, tenant_id: str):
        self.collaborators.discard(tenant_id)

    def can_access(self, tenant_id: str) -> bool:
        return tenant_id == self.tenant_id or tenant_id in self.collaborators


@dataclass
class CollaborationWorkspace:
    workspace_id: str
    name: str
    owner_tenant_id: str
    description: str = ""
    member_tenants: Set[str] = field(default_factory=set)
    member_users: Set[str] = field(default_factory=set)
    shared_datasets: Set[str] = field(default_factory=set)
    shared_results: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    permissions: Dict[str, Set[Permission]] = field(default_factory=dict)
    is_active: bool = True

    def add_tenant_member(self, tenant_id: str, perms: Optional[Set[Permission]] = None):
        self.member_tenants.add(tenant_id)
        if perms:
            self.permissions[tenant_id] = perms

    def add_user_member(self, user_id: str, perms: Optional[Set[Permission]] = None):
        self.member_users.add(user_id)
        if perms:
            self.permissions[user_id] = perms

    def remove_member(self, member_id: str):
        self.member_tenants.discard(member_id)
        self.member_users.discard(member_id)
        self.permissions.pop(member_id, None)

    def share_dataset(self, dataset_id: str):
        self.shared_datasets.add(dataset_id)

    def share_result(self, result_id: str):
        self.shared_results.add(result_id)

    def unshare(self, resource_id: str):
        self.shared_datasets.discard(resource_id)
        self.shared_results.discard(resource_id)

    def check_permission(self, member_id: str, perm: Permission) -> bool:
        if member_id == self.owner_tenant_id:
            return True
        perms = self.permissions.get(member_id, set())
        return perm in perms

    def is_member(self, tenant_id: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        if tenant_id == self.owner_tenant_id:
            return True
        if tenant_id and tenant_id in self.member_tenants:
            return True
        if user_id and user_id in self.member_users:
            return True
        return False


@dataclass
class Tenant:
    tenant_id: str
    name: str
    role: TenantRole = TenantRole.RESEARCH
    quota: ResourceQuota = field(default_factory=ResourceQuota)
    usage: ResourceUsage = field(default=None)
    sandboxes: Dict[str, DataSandbox] = field(default_factory=dict)
    users: Dict[str, User] = field(default_factory=dict)
    workspaces: Dict[str, CollaborationWorkspace] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.usage is None:
            self.usage = ResourceUsage(tenant_id=self.tenant_id)

    def get_quota_utilization(self) -> Dict[str, float]:
        return self.quota.utilization_pct(self.usage.to_quota())

    def check_quota(self) -> Tuple[bool, List[str]]:
        return self.quota.exceeds(self.usage.to_quota())

    def create_sandbox(
        self,
        name: str,
        base_path: str,
        storage_quota_tb: Optional[float] = None,
        is_isolated: bool = True,
    ) -> DataSandbox:
        sandbox = DataSandbox(
            sandbox_id=str(uuid.uuid4()),
            tenant_id=self.tenant_id,
            name=name,
            base_path=base_path,
            storage_quota_tb=storage_quota_tb or self.quota.storage_tb,
            is_isolated=is_isolated,
        )
        sandbox.initialize()
        self.sandboxes[sandbox.sandbox_id] = sandbox
        return sandbox

    def add_user(
        self,
        username: str,
        email: str,
        role: TenantRole = TenantRole.VIEWER,
        permissions: Optional[Set[Permission]] = None,
    ) -> User:
        user = User(
            user_id=str(uuid.uuid4()),
            username=username,
            email=email,
            tenant_id=self.tenant_id,
            role=role,
            permissions=permissions or set(),
        )
        self.users[user.user_id] = user
        return user

    def remove_user(self, user_id: str):
        if user_id in self.users:
            del self.users[user_id]

    def create_workspace(
        self,
        name: str,
        description: str = "",
    ) -> CollaborationWorkspace:
        workspace = CollaborationWorkspace(
            workspace_id=str(uuid.uuid4()),
            name=name,
            owner_tenant_id=self.tenant_id,
            description=description,
        )
        self.workspaces[workspace.workspace_id] = workspace
        return workspace


@dataclass
class ScalingPolicy:
    enabled: bool = True
    policy: str = "auto"
    min_workers: int = 16
    max_workers: int = 1024
    scale_up_threshold: float = 0.75
    scale_down_threshold: float = 0.25
    cooldown_seconds: int = 300
    scale_up_step: int = 8
    scale_down_step: int = 4


class DynamicScaler:
    def __init__(self, policy: Optional[ScalingPolicy] = None):
        self.policy = policy or ScalingPolicy()
        self.current_workers = self.policy.min_workers
        self._last_scale_time: Optional[datetime] = None
        self._lock = threading.Lock()
        self._scale_history: List[Tuple[datetime, int, float]] = []

    def get_target_workers(self, current_load: float) -> int:
        if not self.policy.enabled:
            return self.current_workers

        now = datetime.now()
        with self._lock:
            if self._last_scale_time:
                cooldown = timedelta(seconds=self.policy.cooldown_seconds)
                if now - self._last_scale_time < cooldown:
                    return self.current_workers

            target = self.current_workers
            if current_load >= self.policy.scale_up_threshold:
                target = min(
                    self.current_workers + self.policy.scale_up_step,
                    self.policy.max_workers,
                )
            elif current_load <= self.policy.scale_down_threshold:
                target = max(
                    self.current_workers - self.policy.scale_down_step,
                    self.policy.min_workers,
                )

            if target != self.current_workers:
                self._scale_history.append((now, target, current_load))
                self._last_scale_time = now
                self.current_workers = target
                logger.info(f"Scaling workers: {self.current_workers} -> {target} (load: {current_load:.2f})")

            return self.current_workers

    def measure_system_load(self) -> float:
        try:
            cpu_load = psutil.cpu_percent(interval=1) / 100.0
            memory_load = psutil.virtual_memory().percent / 100.0
            return max(cpu_load, memory_load)
        except Exception:
            return 0.5

    def set_worker_count(self, n: int):
        n = max(self.policy.min_workers, min(n, self.policy.max_workers))
        with self._lock:
            self.current_workers = n
            self._last_scale_time = datetime.now()

    @property
    def scale_history(self) -> List[Tuple[datetime, int, float]]:
        return self._scale_history.copy()


class TenantManager:
    def __init__(
        self,
        base_sandbox_path: str = "./sandboxes",
        default_quota: Optional[ResourceQuota] = None,
        enable_sandbox_isolation: bool = True,
        scaling_policy: Optional[ScalingPolicy] = None,
    ):
        self.base_sandbox_path = base_sandbox_path
        self.default_quota = default_quota or ResourceQuota.default_research_quota()
        self.enable_sandbox_isolation = enable_sandbox_isolation
        self.tenants: Dict[str, Tenant] = {}
        self.acl = AccessControlList()
        self.scaler = DynamicScaler(scaling_policy)
        self._lock = threading.Lock()

        Path(self.base_sandbox_path).mkdir(parents=True, exist_ok=True)

    def create_tenant(
        self,
        tenant_id: Optional[str],
        name: str,
        role: TenantRole = TenantRole.RESEARCH,
        quota: Optional[ResourceQuota] = None,
    ) -> Tenant:
        tid = tenant_id or str(uuid.uuid4())
        with self._lock:
            if tid in self.tenants:
                raise ValueError(f"Tenant {tid} already exists")

            tenant = Tenant(
                tenant_id=tid,
                name=name,
                role=role,
                quota=quota or ResourceQuota(**self.default_quota.__dict__),
            )
            self.tenants[tid] = tenant

            tenant.create_sandbox(
                name=f"{name}_default",
                base_path=self.base_sandbox_path,
                is_isolated=self.enable_sandbox_isolation,
            )

        logger.info(f"Created tenant: {tid} ({name}) with role {role.value}")
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self.tenants.get(tenant_id)

    def remove_tenant(self, tenant_id: str):
        with self._lock:
            if tenant_id in self.tenants:
                tenant = self.tenants[tenant_id]
                for sandbox in tenant.sandboxes.values():
                    sandbox_path = Path(self.base_sandbox_path) / sandbox.sandbox_id
                    if sandbox_path.exists():
                        import shutil
                        shutil.rmtree(sandbox_path, ignore_errors=True)
                del self.tenants[tenant_id]
                logger.info(f"Removed tenant: {tenant_id}")

    def list_tenants(self) -> List[Tenant]:
        return sorted(self.tenants.values(), key=lambda t: t.created_at)

    def check_resource_available(
        self,
        tenant_id: str,
        requested: ResourceQuota,
    ) -> Tuple[bool, List[str]]:
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            return False, [f"Tenant {tenant_id} not found"]
        if not tenant.is_active:
            return False, [f"Tenant {tenant_id} is inactive"]

        projected_usage = ResourceUsage(
            tenant_id=tenant_id,
            storage_tb=tenant.usage.storage_tb + requested.storage_tb,
            compute_hours_month=tenant.usage.compute_hours_month + requested.compute_hours_month,
            concurrent_jobs=tenant.usage.concurrent_jobs + requested.concurrent_jobs,
            memory_gb=tenant.usage.memory_gb + requested.memory_gb,
            gpu_count=tenant.usage.gpu_count + requested.gpu_count,
            bandwidth_gbps=tenant.usage.bandwidth_gbps + requested.bandwidth_gbps,
        )
        exceeded, reasons = tenant.quota.exceeds(projected_usage.to_quota())
        return (not exceeded), reasons

    def allocate_resources(self, tenant_id: str, requested: ResourceQuota) -> bool:
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            return False

        ok, _ = self.check_resource_available(tenant_id, requested)
        if not ok:
            return False

        with self._lock:
            tenant.usage.storage_tb += requested.storage_tb
            tenant.usage.compute_hours_month += requested.compute_hours_month
            tenant.usage.concurrent_jobs += requested.concurrent_jobs
            tenant.usage.memory_gb += requested.memory_gb
            tenant.usage.gpu_count += requested.gpu_count
            tenant.usage.bandwidth_gbps += requested.bandwidth_gbps
            tenant.usage.last_updated = datetime.now()

        return True

    def release_resources(self, tenant_id: str, released: ResourceQuota):
        tenant = self.get_tenant(tenant_id)
        if tenant is None:
            return

        with self._lock:
            tenant.usage.storage_tb = max(0.0, tenant.usage.storage_tb - released.storage_tb)
            tenant.usage.compute_hours_month = max(0.0, tenant.usage.compute_hours_month - released.compute_hours_month)
            tenant.usage.concurrent_jobs = max(0, tenant.usage.concurrent_jobs - released.concurrent_jobs)
            tenant.usage.memory_gb = max(0.0, tenant.usage.memory_gb - released.memory_gb)
            tenant.usage.gpu_count = max(0, tenant.usage.gpu_count - released.gpu_count)
            tenant.usage.bandwidth_gbps = max(0.0, tenant.usage.bandwidth_gbps - released.bandwidth_gbps)
            tenant.usage.last_updated = datetime.now()

    def grant_resource_permission(
        self,
        resource_id: str,
        resource_type: ResourceType,
        permissions: Set[Permission],
        to_tenant_id: Optional[str] = None,
        to_user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        owner_tenant_id: Optional[str] = None,
    ) -> bool:
        if owner_tenant_id:
            owner = self.get_tenant(owner_tenant_id)
            if not owner or owner.role not in [TenantRole.ADMIN, TenantRole.OPERATIONAL]:
                return False

        self.acl.grant(
            resource_id=resource_id,
            resource_type=resource_type,
            permissions=permissions,
            user_id=to_user_id,
            tenant_id=to_tenant_id,
            expires_at=expires_at,
        )
        return True

    def check_access(
        self,
        resource_id: str,
        permission: Permission,
        user: Optional[User] = None,
        tenant_id: Optional[str] = None,
    ) -> bool:
        return self.acl.check(resource_id, permission, user, tenant_id)

    def create_collaboration_workspace(
        self,
        owner_tenant_id: str,
        name: str,
        description: str = "",
    ) -> Optional[CollaborationWorkspace]:
        owner = self.get_tenant(owner_tenant_id)
        if not owner:
            return None
        return owner.create_workspace(name, description)

    def share_with_workspace(
        self,
        workspace_id: str,
        resource_id: str,
        resource_type: ResourceType,
        permissions: Set[Permission],
    ) -> bool:
        for tenant in self.tenants.values():
            if workspace_id in tenant.workspaces:
                ws = tenant.workspaces[workspace_id]
                if resource_type == ResourceType.DATASET:
                    ws.share_dataset(resource_id)
                elif resource_type == ResourceType.RESULT:
                    ws.share_result(resource_id)

                for member_tenant_id in ws.member_tenants:
                    self.acl.grant(
                        resource_id=resource_id,
                        resource_type=resource_type,
                        permissions=permissions,
                        tenant_id=member_tenant_id,
                    )
                return True
        return False

    def scale_workers(self, tenant_id: Optional[str] = None) -> int:
        load = self.scaler.measure_system_load()
        if tenant_id:
            tenant = self.get_tenant(tenant_id)
            if tenant:
                utils = tenant.get_quota_utilization()
                load = max(load, utils.get("concurrent_jobs", 0) / 100)
        return self.scaler.get_target_workers(load)

    def get_worker_count(self) -> int:
        return self.scaler.current_workers

    def get_system_status(self) -> Dict[str, Any]:
        tenant_stats = []
        for tenant in self.tenants.values():
            quota_ok, issues = tenant.check_quota()
            tenant_stats.append({
                "tenant_id": tenant.tenant_id,
                "name": tenant.name,
                "role": tenant.role.value,
                "active": tenant.is_active,
                "utilization": tenant.get_quota_utilization(),
                "over_quota": not quota_ok,
                "issues": issues,
                "users": len(tenant.users),
                "sandboxes": len(tenant.sandboxes),
                "workspaces": len(tenant.workspaces),
            })

        return {
            "num_tenants": len(self.tenants),
            "active_workers": self.scaler.current_workers,
            "scaling_policy": self.scaler.policy.__dict__,
            "system_load": self.scaler.measure_system_load(),
            "tenants": tenant_stats,
        }

    def reset_monthly_usage(self):
        with self._lock:
            for tenant in self.tenants.values():
                tenant.usage.compute_hours_month = 0.0
                tenant.usage.concurrent_jobs = 0
                tenant.usage.last_updated = datetime.now()
