/**
 * @file
 * @author DeepLink
 * @copyright  (c) 2023, DeepLink.
 */

#include <diopi/functions.h>

#include "../common/acloprunner.hpp"

namespace impl {
namespace ascend {
extern "C" {
DIOPI_API diopiError_t diopiWhere(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t condition, diopiConstTensorHandle_t input,
                                  diopiConstTensorHandle_t other) {
    std::vector<int64_t> dimVec({0});
    diopiSize_t dim = vectorToDiopiSize(dimVec);
    AclOpRunner<3, 1>("Select", ctx).addInput(condition).addInput(input).addInput(other).addOutput(out).run();
    return diopiSuccess;
}
}

}  // namespace ascend
}  // namespace impl
