from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MLIRDialectRegistration:
    dialect_namespace: str
    cmake_option: str
    header: str
    source: str
    registration_function: str
    requires: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "dialect_namespace": self.dialect_namespace,
            "cmake_option": self.cmake_option,
            "header": self.header,
            "source": self.source,
            "registration_function": self.registration_function,
            "requires": self.requires,
        }


def cpp_dialect_registration(repo_root: str | Path | None = None) -> MLIRDialectRegistration:
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    return MLIRDialectRegistration(
        dialect_namespace="cacheir",
        cmake_option="CACHEIR_BUILD_MLIR",
        header=str(root / "cpp" / "mlir" / "include" / "cacheir" / "mlir" / "CacheIRDialect.h"),
        source=str(root / "cpp" / "mlir" / "src" / "CacheIRDialect.cpp"),
        registration_function="cacheir::mlir::registerCacheIRDialect",
        requires=("MLIRConfig.cmake", "MLIRIR"),
    )
