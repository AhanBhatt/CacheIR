#include "cacheir/mlir/CacheIRDialect.h"

#include "mlir/IR/MLIRContext.h"

namespace cacheir::mlir {

CacheIRDialect::CacheIRDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
                      ::mlir::TypeID::get<CacheIRDialect>()) {
  allowUnknownOperations();
  allowUnknownTypes();
}

void registerCacheIRDialect(::mlir::DialectRegistry &registry) {
  registry.insert<CacheIRDialect>();
}

} // namespace cacheir::mlir
