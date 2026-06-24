#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"

namespace cacheir::mlir {

class CacheIRDialect final : public ::mlir::Dialect {
public:
  explicit CacheIRDialect(::mlir::MLIRContext *context);

  static ::llvm::StringRef getDialectNamespace() { return "cacheir"; }
};

void registerCacheIRDialect(::mlir::DialectRegistry &registry);

} // namespace cacheir::mlir
