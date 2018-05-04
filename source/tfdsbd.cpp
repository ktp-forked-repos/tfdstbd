#include <unicode/uversion.h>

#if U_ICU_VERSION_MAJOR_NUM < 60
#error Required ICU version >= 60.0
#endif


#include <unicode/unistr.h>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define FEATURES_SIZE 6
#define MAX_LEN 15

using namespace tensorflow;
using icu::UnicodeString;

REGISTER_OP("FeaturesLengthCase")
  .Input("source: string")
  .Output("result: float")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle result_shape;
    TF_RETURN_IF_ERROR(c->Concatenate(c->input(0), c->Vector(FEATURES_SIZE), &result_shape));

    c->set_output(0, result_shape);

    return Status::OK();
    })
  .SetIsStateful();


class FeaturesLengthCaseOp : public OpKernel {
 public:
  explicit FeaturesLengthCaseOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    // Prepare source
    const Tensor* source_tensor;
    OP_REQUIRES_OK(ctx, ctx->input("source", &source_tensor));
    const auto source = source_tensor->flat<string>();

    const uint64 source_len = source_tensor->shape().num_elements();


    // Allocate result
    TensorShape result_shape = source_tensor->shape();
    result_shape.AddDim(FEATURES_SIZE);
    Tensor* result_tensor;
    OP_REQUIRES_OK(
      ctx,
      ctx->allocate_output(
        0,
        result_shape,
        &result_tensor
      )
    );
    auto result = result_tensor->flat<float>();


    // Extract features from source values
    for (uint64 s = 0; s < source_len; s++) {
      string binary_string = source(s);

      if (0 == binary_string.length()) {
        result(s * FEATURES_SIZE + 0) = 0.; // Length
        result(s * FEATURES_SIZE + 1) = 1.; // No case
        result(s * FEATURES_SIZE + 2) = 0.; // Lower case
        result(s * FEATURES_SIZE + 3) = 0.; // Upper case
        result(s * FEATURES_SIZE + 4) = 0.; // Title case
        result(s * FEATURES_SIZE + 5) = 0.; // Mixed case
        continue;
      }

      UnicodeString unicode_string = UnicodeString::fromUTF8(binary_string);

      uint64 length = unicode_string.length();
      length = (length < MAX_LEN) ? length : MAX_LEN;
      result(s * FEATURES_SIZE + 0) = (float)length / MAX_LEN; // Length


      UnicodeString unicode_lower = UnicodeString(unicode_string);
      unicode_lower.toLower();

      UnicodeString unicode_upper = UnicodeString(unicode_string);
      unicode_upper.toUpper();

      if (unicode_lower == unicode_upper) {
        result(s * FEATURES_SIZE + 1) = 1.; // No case
        result(s * FEATURES_SIZE + 2) = 0.; // Lower case
        result(s * FEATURES_SIZE + 3) = 0.; // Upper case
        result(s * FEATURES_SIZE + 4) = 0.; // Title case
        result(s * FEATURES_SIZE + 5) = 0.; // Mixed case
        continue;
      } else {
        result(s * FEATURES_SIZE + 1) = 0.; // No case
      }

      if (unicode_lower == unicode_string) {
        result(s * FEATURES_SIZE + 2) = 1.; // Lower case
        result(s * FEATURES_SIZE + 3) = 0.; // Upper case
        result(s * FEATURES_SIZE + 4) = 0.; // Title case
        result(s * FEATURES_SIZE + 5) = 0.; // Mixed case
        continue;
      } else {
        result(s * FEATURES_SIZE + 2) = 0.; // Lower case
      }

      if (unicode_upper == unicode_string) {
        result(s * FEATURES_SIZE + 3) = 1.; // Upper case
        result(s * FEATURES_SIZE + 4) = 0.; // Title case
        result(s * FEATURES_SIZE + 5) = 0.; // Mixed case
        continue;
      } else {
        result(s * FEATURES_SIZE + 3) = 0.; // Lower case
      }

      UnicodeString unicode_head;
      unicode_string.extract(0, 1, unicode_head);
      UnicodeString unicode_head_upper = UnicodeString(unicode_head);
      unicode_head_upper.toUpper();

      UnicodeString unicode_tail;
      unicode_string.extract(1, unicode_string.length() - 1, unicode_tail);
      UnicodeString unicode_tail_lower = UnicodeString(unicode_tail);
      unicode_tail_lower.toLower();

      if (unicode_head_upper == unicode_head && unicode_tail_lower == unicode_tail) {
        result(s * FEATURES_SIZE + 4) = 1.; // Title case
        result(s * FEATURES_SIZE + 5) = 0.; // Mixed case
        continue;
      } else {
        result(s * FEATURES_SIZE + 4) = 0.; // Lower case
      }

      result(s * FEATURES_SIZE + 5) = 1.; // Mixed case
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("FeaturesLengthCase").Device(DEVICE_CPU), FeaturesLengthCaseOp);

