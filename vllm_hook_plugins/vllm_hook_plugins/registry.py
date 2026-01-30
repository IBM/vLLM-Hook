from typing import Dict, Optional

class Worker:
    def __init__(self, worker_class):
        # self.cls = worker_class
        self.path = f"{worker_class.__module__}.{worker_class.__name__}"

class Analyzer:
    def __init__(self, analyzer_class):
        self.analyzer = analyzer_class
        # self.name = analyzer_class.__name__
    
class PluginRegistry:

    _workers: Dict[str, Worker] = {}
    _analyzers: Dict[str, Analyzer] = {}
    
    ## workers
    @classmethod
    def register_worker(cls, name: str, worker_class):
        cls._workers[name] = Worker(worker_class)
    
    @classmethod
    def get_worker(cls, name: str) -> Optional[Worker]:
        return cls._workers.get(name)
    
    @classmethod
    def list_workers(cls) -> list:
        return list(cls._workers.keys())
    
    ## analyzers
    @classmethod
    def register_analyzer(cls, name: str, worker_class):
        cls._analyzers[name] = Analyzer(worker_class)
    
    @classmethod
    def get_analyzer(cls, name: str) -> Optional[Analyzer]:
        return cls._analyzers.get(name)
    
    @classmethod
    def list_analyzers(cls) -> list:
        return list(cls._analyzers.keys())
    
    ## overall
    @classmethod
    def list(cls) -> Dict[str, list]:
        return {
            "workers": cls.list_workers(),
            "analyzers": cls.list_analyzers()
        }
    
