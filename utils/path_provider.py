import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from utils.utils import make_dirs


class PathProvider(ABC):
    @abstractmethod
    def provide(self):
        pass


class TemporaryPathProvider(PathProvider):
    def provide(self):
        return tempfile.TemporaryDirectory().name


@dataclass
class SystemPathProvider(PathProvider):
    path: str

    def provide(self) -> str:
        return make_dirs(self.path)
