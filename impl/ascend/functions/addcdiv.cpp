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

DIOPI_API diopiError_t diopiAddcdiv(diopiContextHandle_t ctx, diopiTensorHandle_t out, diopiConstTensorHandle_t input, diopiConstTensorHandle_t tensor1,
                                    diopiConstTensorHandle_t tensor2, const diopiScalar_t* value) {
    diopiTensorHandle_t trOther = nullptr;
    diopiDtype_t dtype;
    diopiGetTensorDtype(out, &dtype);
    makeTensorFromScalar(ctx, value, &trOther, dtype, diopiDevice_t::diopi_device);
    AclOpRunner<4, 1>("Addcdiv", ctx).addInput(input).addInput(tensor1).addInput(tensor2).addInput(trOther).addOutput(out).run();
    return diopiSuccess;
}

DIOPI_API diopiError_t diopiAddcdivInp(diopiContextHandle_t ctx, diopiTensorHandle_t input, diopiConstTensorHandle_t tensor1, diopiConstTensorHandle_t tensor2,
                                       const diopiScalar_t* value) {
    return diopiAddcdiv(ctx, input, input, tensor1, tensor2, value);
}

}  // extern "C"

}  // namespace ascend
}  // namespace impl
