"""Registry of worker and analyzer plugins by name."""
from typing import Dict, Optional

class Worker:
    """A registered worker; ``path`` is its dotted class path for ``worker_extension_cls``."""
    def __init__(self, worker_class):
        self.path = f"{worker_class.__module__}.{worker_class.__name__}"

class Analyzer:
    """A registered analyzer; ``analyzer`` is its class."""
    def __init__(self, analyzer_class):
        self.analyzer = analyzer_class

class PluginRegistry:
    """Process-wide name registry of MIA workers and analyzers."""
    _workers: Dict[str, Worker] = {}
    _analyzers: Dict[str, Analyzer] = {}

    @classmethod
    def register_worker(cls, name: str, worker_class):
        """Register ``worker_class`` under ``name``."""
        cls._workers[name] = Worker(worker_class)

    @classmethod
    def get_worker(cls, name: str) -> Optional[Worker]:
        """The worker registered as ``name``, or None."""
        return cls._workers.get(name)

    @classmethod
    def list_workers(cls) -> list:
        """Names of the registered workers."""
        return list(cls._workers.keys())

    @classmethod
    def register_analyzer(cls, name: str, worker_class):
        """Register the analyzer class ``worker_class`` under ``name``."""
        cls._analyzers[name] = Analyzer(worker_class)

    @classmethod
    def get_analyzer(cls, name: str) -> Optional[Analyzer]:
        """The analyzer registered as ``name``, or None."""
        return cls._analyzers.get(name)

    @classmethod
    def list_analyzers(cls) -> list:
        """Names of the registered analyzers."""
        return list(cls._analyzers.keys())

    @classmethod
    def list(cls) -> Dict[str, list]:
        """``{"workers": [...], "analyzers": [...]}``: every registered name."""
        return {
            "workers": cls.list_workers(),
            "analyzers": cls.list_analyzers()
        }

