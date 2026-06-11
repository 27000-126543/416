"""
Task orchestration engine with workflow management, checkpointing, and parameter optimization.
"""

import hashlib
import json
import logging
import pickle
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class Task:
    def __init__(
        self,
        task_id: Optional[str] = None,
        name: str = "",
        function: Optional[Callable] = None,
        args: Optional[Tuple] = None,
        kwargs: Optional[Dict] = None,
        dependencies: Optional[List[str]] = None,
        priority: int = 0,
        retry_attempts: int = 3,
        retry_backoff_seconds: int = 60,
        checkpoint: bool = True,
        tenant_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.task_id = task_id or str(uuid.uuid4())
        self.name = name or self.task_id
        self.function = function
        self.args = args or ()
        self.kwargs = kwargs or {}
        self.dependencies: List[str] = dependencies or []
        self.priority = priority
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_seconds
        self.checkpoint = checkpoint
        self.tenant_id = tenant_id
        self.metadata = metadata or {}

        self.status = TaskStatus.PENDING
        self.result: Optional[TaskResult] = None
        self.current_attempt = 0

    def add_dependency(self, task_id: str):
        if task_id not in self.dependencies:
            self.dependencies.append(task_id)

    def execute(self) -> TaskResult:
        start_time = datetime.now()
        self.status = TaskStatus.RUNNING
        self.current_attempt += 1

        try:
            if self.function is None:
                raise ValueError(f"Task {self.task_id} has no function assigned")

            output = self.function(*self.args, **self.kwargs)
            checksum = self._compute_checksum(output)

            result = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.COMPLETED,
                output=output,
                start_time=start_time,
                end_time=datetime.now(),
                checksum=checksum,
                metadata=self.metadata,
            )
            self.status = TaskStatus.COMPLETED
            self.result = result
            return result

        except Exception as e:
            logger.error(f"Task {self.task_id} failed (attempt {self.current_attempt}): {e}")
            result = TaskResult(
                task_id=self.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=datetime.now(),
                metadata=self.metadata,
            )
            if self.current_attempt >= self.retry_attempts:
                self.status = TaskStatus.FAILED
            self.result = result
            return result

    @staticmethod
    def _compute_checksum(output: Any) -> Optional[str]:
        try:
            if isinstance(output, np.ndarray):
                return hashlib.sha256(output.tobytes()).hexdigest()
            elif isinstance(output, (dict, list, tuple, str, int, float)):
                return hashlib.sha256(json.dumps(output, sort_keys=True, default=str).encode()).hexdigest()
            return None
        except Exception:
            return None

    @property
    def is_ready(self) -> bool:
        return self.status == TaskStatus.READY

    @property
    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    def can_retry(self) -> bool:
        return self.current_attempt < self.retry_attempts


class DependencyManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._reverse_deps: Dict[str, Set[str]] = {}

    def add_task(self, task: Task):
        self._tasks[task.task_id] = task
        if task.task_id not in self._reverse_deps:
            self._reverse_deps[task.task_id] = set()
        for dep_id in task.dependencies:
            if dep_id not in self._reverse_deps:
                self._reverse_deps[dep_id] = set()
            self._reverse_deps[dep_id].add(task.task_id)

    def remove_task(self, task_id: str):
        if task_id in self._tasks:
            del self._tasks[task_id]
        if task_id in self._reverse_deps:
            del self._reverse_deps[task_id]
        for task in self._tasks.values():
            if task_id in task.dependencies:
                task.dependencies.remove(task_id)

    def get_ready_tasks(self) -> List[Task]:
        ready = []
        for task in self._tasks.values():
            if task.status == TaskStatus.PENDING:
                deps_completed = all(
                    dep_id in self._tasks
                    and self._tasks[dep_id].status == TaskStatus.COMPLETED
                    for dep_id in task.dependencies
                )
                if deps_completed:
                    task.status = TaskStatus.READY
                    ready.append(task)
        ready.sort(key=lambda t: (-t.priority, t.task_id))
        return ready

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self._tasks.values())

    def topological_sort(self) -> List[Task]:
        in_degree = {tid: len(t.dependencies) for tid, t in self._tasks.items()}
        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            queue.sort(key=lambda tid: (-self._tasks[tid].priority, tid))
            current = queue.pop(0)
            result.append(self._tasks[current])

            for dependent in self._reverse_deps.get(current, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._tasks):
            raise ValueError("Circular dependency detected in workflow")

        return result

    def get_dependents(self, task_id: str) -> Set[str]:
        return self._reverse_deps.get(task_id, set()).copy()

    def get_dependencies(self, task_id: str) -> List[str]:
        if task_id in self._tasks:
            return self._tasks[task_id].dependencies.copy()
        return []

    def has_circular_dependency(self) -> bool:
        try:
            self.topological_sort()
            return False
        except ValueError:
            return True


@dataclass
class Checkpoint:
    checkpoint_id: str
    task_id: str
    workflow_id: Optional[str]
    state: bytes
    timestamp: datetime
    step: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class CheckpointManager:
    def __init__(self, checkpoint_dir: str = "./checkpoints", interval_seconds: int = 3600):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.interval_seconds = interval_seconds
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint: Dict[str, datetime] = {}

    def save(
        self,
        task: Task,
        state: Any,
        workflow_id: Optional[str] = None,
        step: int = 0,
        force: bool = False,
    ) -> Optional[Checkpoint]:
        if not task.checkpoint:
            return None

        now = datetime.now()
        last = self._last_checkpoint.get(task.task_id)
        if not force and last and (now - last).total_seconds() < self.interval_seconds:
            return None

        try:
            state_bytes = pickle.dumps(state)
            checkpoint_id = hashlib.sha256(state_bytes + task.task_id.encode() + str(step).encode()).hexdigest()[:16]

            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                task_id=task.task_id,
                workflow_id=workflow_id,
                state=state_bytes,
                timestamp=now,
                step=step,
                metadata={"task_name": task.name, **task.metadata},
            )

            file_path = self._get_path(checkpoint_id)
            with open(file_path, "wb") as f:
                pickle.dump(checkpoint, f)

            self._last_checkpoint[task.task_id] = now
            logger.debug(f"Saved checkpoint {checkpoint_id} for task {task.task_id}")
            return checkpoint

        except Exception as e:
            logger.error(f"Failed to save checkpoint for task {task.task_id}: {e}")
            return None

    def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        file_path = self._get_path(checkpoint_id)
        if not file_path.exists():
            return None

        try:
            with open(file_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None

    def load_latest(self, task_id: str) -> Optional[Checkpoint]:
        task_dir = self.checkpoint_dir
        checkpoints = []

        for f in task_dir.glob("*.cpkl"):
            try:
                with open(f, "rb") as fp:
                    cp = pickle.load(fp)
                    if cp.task_id == task_id:
                        checkpoints.append(cp)
            except Exception:
                continue

        if checkpoints:
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
            return checkpoints[0]
        return None

    def list_checkpoints(self, task_id: Optional[str] = None, workflow_id: Optional[str] = None) -> List[Checkpoint]:
        result = []
        for f in self.checkpoint_dir.glob("*.cpkl"):
            try:
                with open(f, "rb") as fp:
                    cp = pickle.load(fp)
                    if task_id and cp.task_id != task_id:
                        continue
                    if workflow_id and cp.workflow_id != workflow_id:
                        continue
                    result.append(cp)
            except Exception:
                continue
        result.sort(key=lambda c: c.timestamp)
        return result

    def delete(self, checkpoint_id: str) -> bool:
        file_path = self._get_path(checkpoint_id)
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def clear_old(self, retention_days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        for f in self.checkpoint_dir.glob("*.cpkl"):
            try:
                with open(f, "rb") as fp:
                    cp = pickle.load(fp)
                    if cp.timestamp < cutoff:
                        f.unlink()
                        deleted += 1
            except Exception:
                continue
        return deleted

    def _get_path(self, checkpoint_id: str) -> Path:
        return self.checkpoint_dir / f"{checkpoint_id}.cpkl"


class ResultValidator:
    def __init__(self, algorithm: str = "sha256"):
        self.algorithm = algorithm

    def compute_checksum(self, data: Any) -> Optional[str]:
        try:
            if isinstance(data, np.ndarray):
                return hashlib.new(self.algorithm, data.tobytes()).hexdigest()
            elif isinstance(data, bytes):
                return hashlib.new(self.algorithm, data).hexdigest()
            elif isinstance(data, (str, int, float, bool)):
                return hashlib.new(self.algorithm, str(data).encode()).hexdigest()
            elif isinstance(data, (dict, list, tuple)):
                return hashlib.new(self.algorithm, json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            else:
                serialized = pickle.dumps(data)
                return hashlib.new(self.algorithm, serialized).hexdigest()
        except Exception:
            return None

    def validate(self, result: TaskResult, expected_checksum: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        issues = []
        valid = True

        if result.status != TaskStatus.COMPLETED:
            valid = False
            issues.append(f"Task status is {result.status.value}, not completed")
            return valid, {"issues": issues}

        actual_checksum = self.compute_checksum(result.output)
        if expected_checksum and actual_checksum != expected_checksum:
            valid = False
            issues.append(f"Checksum mismatch: expected {expected_checksum}, got {actual_checksum}")

        if result.output is None:
            issues.append("Task output is None")

        if hasattr(result.output, 'size') and result.output.size == 0:
            issues.append("Task output is empty")

        if result.start_time and result.end_time:
            if result.end_time < result.start_time:
                valid = False
                issues.append("End time is before start time")

        return valid, {"issues": issues, "actual_checksum": actual_checksum}

    def validate_workflow(self, results: Dict[str, TaskResult]) -> Tuple[bool, Dict[str, Any]]:
        all_valid = True
        details = {}

        for task_id, result in results.items():
            valid, info = self.validate(result)
            details[task_id] = info
            if not valid:
                all_valid = False

        return all_valid, details


@dataclass
class ParameterRange:
    name: str
    min_value: float
    max_value: float
    distribution: str = "uniform"
    discrete: bool = False
    values: Optional[List[Any]] = None

    def sample(self, n: int, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        r = rng or np.random.RandomState()
        if self.values is not None:
            indices = r.randint(0, len(self.values), n)
            return np.array([self.values[i] for i in indices])
        if self.distribution == "uniform":
            samples = r.uniform(self.min_value, self.max_value, n)
        elif self.distribution == "normal":
            mean = (self.min_value + self.max_value) / 2
            std = (self.max_value - self.min_value) / 6
            samples = r.normal(mean, std, n)
            samples = np.clip(samples, self.min_value, self.max_value)
        elif self.distribution == "loguniform":
            samples = np.exp(r.uniform(np.log(self.min_value), np.log(self.max_value), n))
        else:
            samples = r.uniform(self.min_value, self.max_value, n)
        if self.discrete:
            samples = np.round(samples).astype(int)
        return samples


class LatinHypercubeSampler:
    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def sample(self, parameters: List[ParameterRange], n_samples: int) -> List[Dict[str, Any]]:
        if not parameters:
            return [{}]

        n_params = len(parameters)
        result_samples = np.zeros((n_samples, n_params))

        for j, param in enumerate(parameters):
            permuted = self.rng.permutation(n_samples)
            u = (permuted + self.rng.uniform(size=n_samples)) / n_samples

            if param.values is not None:
                indices = np.floor(u * len(param.values)).astype(int)
                result_samples[:, j] = indices
            elif param.distribution == "loguniform":
                result_samples[:, j] = np.exp(
                    np.log(param.min_value) + u * (np.log(param.max_value) - np.log(param.min_value))
                )
            elif param.distribution == "normal":
                from scipy.stats import norm
                mean = (param.min_value + param.max_value) / 2
                std = (param.max_value - param.min_value) / 6
                result_samples[:, j] = norm.ppf(u, mean, std)
                result_samples[:, j] = np.clip(result_samples[:, j], param.min_value, param.max_value)
            else:
                result_samples[:, j] = param.min_value + u * (param.max_value - param.min_value)

            if param.discrete:
                result_samples[:, j] = np.round(result_samples[:, j])

        samples = []
        for i in range(n_samples):
            sample_dict = {}
            for j, param in enumerate(parameters):
                if param.values is not None:
                    sample_dict[param.name] = param.values[int(result_samples[i, j])]
                else:
                    sample_dict[param.name] = result_samples[i, j]
            samples.append(sample_dict)

        return samples


class BayesianOptimizer:
    def __init__(
        self,
        parameters: List[ParameterRange],
        objective_function: Callable[[Dict[str, Any]], float],
        maximize: bool = True,
        n_initial: int = 10,
        acquisition: str = "expected_improvement",
        seed: Optional[int] = None,
    ):
        self.parameters = parameters
        self.objective = objective_function
        self.maximize = maximize
        self.n_initial = n_initial
        self.acquisition = acquisition
        self.rng = np.random.RandomState(seed)
        self.sampler = LatinHypercubeSampler(seed)

        self.history_x: List[Dict[str, Any]] = []
        self.history_y: List[float] = []
        self._best_x: Optional[Dict[str, Any]] = None
        self._best_y: Optional[float] = None

    def _params_to_array(self, params: Dict[str, Any]) -> np.ndarray:
        arr = []
        for p in self.parameters:
            if p.values is not None:
                arr.append(float(p.values.index(params[p.name])))
            else:
                arr.append(float(params[p.name]))
        return np.array(arr)

    def _array_to_params(self, arr: np.ndarray) -> Dict[str, Any]:
        result = {}
        for i, p in enumerate(self.parameters):
            if p.values is not None:
                idx = int(np.clip(arr[i], 0, len(p.values) - 1))
                result[p.name] = p.values[idx]
            elif p.discrete:
                result[p.name] = int(np.round(arr[i]))
            else:
                result[p.name] = float(arr[i])
        return result

    def run(self, n_iterations: int = 100) -> Tuple[Dict[str, Any], float]:
        if len(self.history_x) == 0:
            initial_samples = self.sampler.sample(self.parameters, self.n_initial)
            for sample in initial_samples:
                self._evaluate(sample)

        for _ in range(n_iterations):
            next_params = self._propose_next()
            self._evaluate(next_params)

        return self._best_x, self._best_y

    def _evaluate(self, params: Dict[str, Any]):
        y = self.objective(params)
        self.history_x.append(params)
        self.history_y.append(y)

        if self.maximize:
            if self._best_y is None or y > self._best_y:
                self._best_y = y
                self._best_x = params
        else:
            if self._best_y is None or y < self._best_y:
                self._best_y = y
                self._best_x = params

    def _propose_next(self) -> Dict[str, Any]:
        if len(self.history_x) < 2:
            return self.sampler.sample(self.parameters, 1)[0]

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

            X_train = np.array([self._params_to_array(p) for p in self.history_x])
            y_train = np.array(self.history_y)

            if not self.maximize:
                y_train = -y_train

            kernel = ConstantKernel(1.0) * Matern(nu=2.5)
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5, random_state=42)
            gp.fit(X_train, y_train)

            bounds = []
            for p in self.parameters:
                if p.values is not None:
                    bounds.append((0, len(p.values) - 1))
                else:
                    bounds.append((p.min_value, p.max_value))

            n_candidates = 1000
            candidates = self.sampler.sample(self.parameters, n_candidates)
            X_candidates = np.array([self._params_to_array(c) for c in candidates])

            mu, sigma = gp.predict(X_candidates, return_std=True)
            sigma = np.maximum(sigma, 1e-9)

            if self.acquisition == "expected_improvement":
                best = np.max(y_train)
                z = (mu - best) / sigma
                from scipy.stats import norm
                ei = (mu - best) * norm.cdf(z) + sigma * norm.pdf(z)
                best_idx = np.argmax(ei)
            elif self.acquisition == "probability_of_improvement":
                best = np.max(y_train)
                z = (mu - best) / sigma
                from scipy.stats import norm
                pi = norm.cdf(z)
                best_idx = np.argmax(pi)
            elif self.acquisition == "upper_confidence_bound":
                kappa = 2.576
                ucb = mu + kappa * sigma
                best_idx = np.argmax(ucb)
            else:
                best_idx = np.argmax(mu)

            return candidates[best_idx]

        except Exception as e:
            logger.warning(f"Bayesian optimization failed, using random sample: {e}")
            return self.sampler.sample(self.parameters, 1)[0]

    @property
    def best_parameters(self) -> Optional[Dict[str, Any]]:
        return self._best_x

    @property
    def best_value(self) -> Optional[float]:
        return self._best_y

    @property
    def history(self) -> List[Tuple[Dict[str, Any], float]]:
        return list(zip(self.history_x, self.history_y))


@dataclass
class ParameterSweepResult:
    parameter_combinations: List[Dict[str, Any]]
    results: List[Any]
    best_index: Optional[int]
    objective_values: Optional[List[float]] = None


class ParameterSweep:
    def __init__(
        self,
        parameters: List[ParameterRange],
        method: str = "latin_hypercube",
        max_samples: int = 1000,
        seed: Optional[int] = None,
    ):
        self.parameters = parameters
        self.method = method
        self.max_samples = max_samples
        self.seed = seed
        self.sampler = LatinHypercubeSampler(seed)

    def generate_combinations(self, n_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        n = n_samples or min(self.max_samples, 100)

        if self.method == "grid":
            return self._grid_search(n)
        elif self.method == "random":
            return self._random_search(n)
        elif self.method == "latin_hypercube":
            return self.sampler.sample(self.parameters, n)
        elif self.method == "sobol":
            return self._sobol_search(n)
        else:
            return self.sampler.sample(self.parameters, n)

    def _grid_search(self, n_samples: int) -> List[Dict[str, Any]]:
        per_param = max(2, int(np.ceil(n_samples ** (1 / max(1, len(self.parameters))))))
        param_values = []
        for p in self.parameters:
            if p.values is not None:
                values = p.values[:per_param]
            elif p.discrete:
                values = list(range(int(p.min_value), int(p.max_value) + 1,
                                    max(1, int((p.max_value - p.min_value) / (per_param - 1)))))
            else:
                values = np.linspace(p.min_value, p.max_value, per_param).tolist()
            param_values.append(values)

        combinations = []
        if param_values:
            mesh = np.meshgrid(*param_values, indexing="ij")
            flat = [m.ravel() for m in mesh]
            for i in range(min(n_samples, len(flat[0]))):
                combo = {}
                for j, p in enumerate(self.parameters):
                    combo[p.name] = flat[j][i]
                combinations.append(combo)
        return combinations

    def _random_search(self, n_samples: int) -> List[Dict[str, Any]]:
        return self.sampler.sample(self.parameters, n_samples)

    def _sobol_search(self, n_samples: int) -> List[Dict[str, Any]]:
        return self.sampler.sample(self.parameters, n_samples)

    def run(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], Any],
        n_samples: Optional[int] = None,
        objective_fn: Optional[Callable[[Any], float]] = None,
        maximize: bool = True,
    ) -> ParameterSweepResult:
        combinations = self.generate_combinations(n_samples)
        results = []
        objectives = []

        for combo in combinations:
            result = evaluate_fn(combo)
            results.append(result)
            if objective_fn:
                objectives.append(objective_fn(result))

        best_idx = None
        if objectives:
            if maximize:
                best_idx = int(np.argmax(objectives))
            else:
                best_idx = int(np.argmin(objectives))

        return ParameterSweepResult(
            parameter_combinations=combinations,
            results=results,
            best_index=best_idx,
            objective_values=objectives if objectives else None,
        )


class Workflow:
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        name: str = "",
        description: str = "",
        tenant_id: Optional[str] = None,
    ):
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name or self.workflow_id
        self.description = description
        self.tenant_id = tenant_id
        self.tasks: Dict[str, Task] = {}
        self.dependency_manager = DependencyManager()

    def add_task(self, task: Task) -> str:
        if task.tenant_id is None and self.tenant_id:
            task.tenant_id = self.tenant_id
        self.tasks[task.task_id] = task
        self.dependency_manager.add_task(task)
        return task.task_id

    def remove_task(self, task_id: str):
        if task_id in self.tasks:
            del self.tasks[task_id]
            self.dependency_manager.remove_task(task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

    def get_ready_tasks(self) -> List[Task]:
        return self.dependency_manager.get_ready_tasks()

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def topological_order(self) -> List[Task]:
        return self.dependency_manager.topological_sort()

    def add_dependency(self, task_id: str, depends_on: str):
        if task_id in self.tasks and depends_on in self.tasks:
            self.tasks[task_id].add_dependency(depends_on)

    def validate(self) -> Tuple[bool, List[str]]:
        issues = []

        if self.dependency_manager.has_circular_dependency():
            issues.append("Circular dependency detected")

        for task_id, task in self.tasks.items():
            for dep_id in task.dependencies:
                if dep_id not in self.tasks:
                    issues.append(f"Task {task_id} depends on missing task {dep_id}")
            if task.function is None:
                issues.append(f"Task {task_id} has no callable function")

        return len(issues) == 0, issues


@dataclass
class WorkflowExecution:
    execution_id: str
    workflow_id: str
    start_time: datetime
    status: TaskStatus = TaskStatus.RUNNING
    end_time: Optional[datetime] = None
    results: Dict[str, TaskResult] = field(default_factory=dict)
    current_task: Optional[str] = None
    errors: List[str] = field(default_factory=list)

    @property
    def completion_percentage(self) -> float:
        if not self.results:
            return 0.0
        completed = sum(1 for r in self.results.values() if r.status == TaskStatus.COMPLETED)
        total = max(len(self.results), 1)
        return completed / total * 100

    @property
    def duration(self) -> Optional[timedelta]:
        if self.end_time:
            return self.end_time - self.start_time
        return None


class OrchestrationEngine:
    def __init__(
        self,
        max_parallel_tasks: int = 1024,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_interval_seconds: int = 3600,
        retry_attempts_default: int = 3,
        retry_backoff_seconds: int = 60,
        checksum_algorithm: str = "sha256",
    ):
        self.max_parallel_tasks = max_parallel_tasks
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, checkpoint_interval_seconds)
        self.retry_attempts_default = retry_attempts_default
        self.retry_backoff_seconds = retry_backoff_seconds
        self.validator = ResultValidator(checksum_algorithm)

        self._workflows: Dict[str, Workflow] = {}
        self._executions: Dict[str, WorkflowExecution] = {}
        self._running: Set[str] = set()
        self._task_results_cache: Dict[str, TaskResult] = {}

    def create_workflow(
        self,
        name: str = "",
        description: str = "",
        tenant_id: Optional[str] = None,
    ) -> Workflow:
        workflow = Workflow(name=name, description=description, tenant_id=tenant_id)
        self._workflows[workflow.workflow_id] = workflow
        return workflow

    def register_workflow(self, workflow: Workflow):
        self._workflows[workflow.workflow_id] = workflow

    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        return self._workflows.get(workflow_id)

    def execute_task(self, task: Task, workflow_id: Optional[str] = None) -> TaskResult:
        latest_cp = None
        if task.checkpoint:
            latest_cp = self.checkpoint_manager.load_latest(task.task_id)
            if latest_cp:
                try:
                    restored_state = pickle.loads(latest_cp.state)
                    task.kwargs["_restored_state"] = restored_state
                    task.kwargs["_restored_step"] = latest_cp.step
                    logger.info(f"Resuming task {task.task_id} from checkpoint {latest_cp.checkpoint_id}")
                except Exception as e:
                    logger.warning(f"Failed to restore checkpoint for task {task.task_id}: {e}")

        result = task.execute()

        if result.status == TaskStatus.FAILED and task.can_retry():
            time.sleep(task.retry_backoff_seconds)
            return self.execute_task(task, workflow_id)

        if result.status == TaskStatus.COMPLETED:
            valid, validation_info = self.validator.validate(result)
            if not valid:
                logger.warning(f"Task {task.task_id} validation failed: {validation_info.get('issues')}")

        if task.checkpoint and result.output is not None:
            self.checkpoint_manager.save(
                task, result.output, workflow_id=workflow_id, step=0, force=True
            )

        self._task_results_cache[task.task_id] = result
        return result

    def execute_workflow(
        self,
        workflow_id: str,
        validate_results: bool = True,
        resume_from: Optional[str] = None,
    ) -> WorkflowExecution:
        workflow = self._workflows.get(workflow_id)
        if workflow is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        valid, issues = workflow.validate()
        if not valid:
            raise ValueError(f"Workflow validation failed: {issues}")

        execution = WorkflowExecution(
            execution_id=str(uuid.uuid4()),
            workflow_id=workflow_id,
            start_time=datetime.now(),
        )
        self._executions[execution.execution_id] = execution
        self._running.add(execution.execution_id)

        if resume_from:
            resume_execution = self._executions.get(resume_from)
            if resume_execution:
                execution.results = resume_execution.results.copy()
                for task_id, result in execution.results.items():
                    if task_id in workflow.tasks:
                        workflow.tasks[task_id].status = result.status

        try:
            ordered_tasks = workflow.topological_order()

            for task in ordered_tasks:
                if execution.execution_id not in self._running:
                    execution.status = TaskStatus.CANCELLED
                    break

                if task.status == TaskStatus.COMPLETED and task.task_id in execution.results:
                    continue

                execution.current_task = task.task_id
                result = self.execute_task(task, workflow_id)
                execution.results[task.task_id] = result

                if result.status == TaskStatus.FAILED:
                    execution.errors.append(f"Task {task.task_id} failed: {result.error}")
                    execution.status = TaskStatus.FAILED
                    break

            if execution.status != TaskStatus.FAILED and execution.status != TaskStatus.CANCELLED:
                if validate_results:
                    all_valid, details = self.validator.validate_workflow(execution.results)
                    if not all_valid:
                        execution.errors.append(f"Result validation failed")
                        execution.status = TaskStatus.FAILED
                    else:
                        execution.status = TaskStatus.COMPLETED
                else:
                    execution.status = TaskStatus.COMPLETED

        except Exception as e:
            execution.errors.append(f"Workflow execution error: {str(e)}")
            execution.status = TaskStatus.FAILED
        finally:
            execution.end_time = datetime.now()
            self._running.discard(execution.execution_id)

        return execution

    def run_parameter_sweep(
        self,
        base_task_fn: Callable[[Dict[str, Any]], Any],
        parameters: List[ParameterRange],
        n_samples: Optional[int] = None,
        method: str = "latin_hypercube",
        objective_fn: Optional[Callable[[Any], float]] = None,
        maximize: bool = True,
        workflow_name: str = "parameter_sweep",
    ) -> ParameterSweepResult:
        sweep = ParameterSweep(parameters, method=method)
        combinations = sweep.generate_combinations(n_samples)

        workflow = self.create_workflow(name=workflow_name)
        task_map: Dict[str, Dict[str, Any]] = {}

        for combo in combinations:
            task = Task(
                name=f"sweep_{json.dumps(combo, sort_keys=True)}",
                function=base_task_fn,
                kwargs={"params": combo},
            )
            tid = workflow.add_task(task)
            task_map[tid] = combo

        execution = self.execute_workflow(workflow.workflow_id)

        results = []
        objectives = []
        ordered_ids = [t.task_id for t in workflow.get_all_tasks()]

        for tid in ordered_ids:
            result = execution.results.get(tid)
            if result and result.output is not None:
                results.append(result.output)
                if objective_fn:
                    objectives.append(objective_fn(result.output))
            else:
                results.append(None)
                if objective_fn:
                    objectives.append(float('inf') if not maximize else float('-inf'))

        best_idx = None
        if objectives:
            valid_objectives = [(i, o) for i, o in enumerate(objectives) if np.isfinite(o)]
            if valid_objectives:
                if maximize:
                    best_idx = max(valid_objectives, key=lambda x: x[1])[0]
                else:
                    best_idx = min(valid_objectives, key=lambda x: x[1])[0]

        return ParameterSweepResult(
            parameter_combinations=[task_map[tid] for tid in ordered_ids],
            results=results,
            best_index=best_idx,
            objective_values=objectives if objectives else None,
        )

    def run_bayesian_optimization(
        self,
        objective_function: Callable[[Dict[str, Any]], float],
        parameters: List[ParameterRange],
        n_iterations: int = 100,
        n_initial: int = 10,
        maximize: bool = True,
    ) -> Tuple[Dict[str, Any], float, BayesianOptimizer]:
        optimizer = BayesianOptimizer(
            parameters=parameters,
            objective_function=objective_function,
            maximize=maximize,
            n_initial=n_initial,
        )
        best_params, best_value = optimizer.run(n_iterations)
        return best_params, best_value, optimizer

    def cancel_execution(self, execution_id: str):
        if execution_id in self._running:
            self._running.discard(execution_id)
            if execution_id in self._executions:
                self._executions[execution_id].status = TaskStatus.CANCELLED
                self._executions[execution_id].end_time = datetime.now()

    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        return self._executions.get(execution_id)

    def list_executions(self, workflow_id: Optional[str] = None) -> List[WorkflowExecution]:
        result = []
        for execution in self._executions.values():
            if workflow_id and execution.workflow_id != workflow_id:
                continue
            result.append(execution)
        result.sort(key=lambda e: e.start_time, reverse=True)
        return result

    def cleanup_old_executions(self, retention_days: int = 30) -> int:
        cutoff = datetime.now() - timedelta(days=retention_days)
        deleted = 0
        for exec_id in list(self._executions.keys()):
            exec_obj = self._executions[exec_id]
            if exec_obj.end_time and exec_obj.end_time < cutoff:
                del self._executions[exec_id]
                deleted += 1

        deleted += self.checkpoint_manager.clear_old(retention_days)
        return deleted
