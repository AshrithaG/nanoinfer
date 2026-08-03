// The model: a static list of nodes over a fixed set of tensors.
//
// Shapes are resolved once at load, then a liveness pass assigns every
// intermediate an offset in one arena. After that, inference allocates nothing.
#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "nanoinfer/ops.h"
#include "nanoinfer/tensor.h"

namespace ni {

enum class Op {
  Conv2d,
  DepthwiseConv2d,
  Linear,
  Relu,
  MaxPool2d,
  GlobalAvgPool,
  Add,
  Softmax,
  Flatten,
  Quantize,
  Dequantize,
};

const char* op_name(Op op);

struct Node {
  Op op;
  std::vector<int> inputs;   // tensor ids
  std::vector<int> outputs;  // tensor ids
  ConvParams conv;
  int pool_k = 2, pool_stride = 2, pool_pad = 0;
  bool relu = false;  // fused activation for Linear/Add
  ConvImpl impl = ConvImpl::Im2colGemm;
  // per-output-channel weight scales for quantized conv/linear
  std::vector<float> w_scales;
};

struct TensorDesc {
  std::string name;
  Shape shape;
  DType dtype = DType::F32;
  bool is_weight = false;
  float scale = 0.0f;
  int32_t zero_point = 0;
  size_t offset = 0;   // into weights blob (weights) or arena (activations)
  size_t nbytes = 0;
};

struct PlanStats {
  size_t arena_bytes = 0;
  size_t sum_of_activation_bytes = 0;  // what a naive allocator would need
  size_t workspace_bytes = 0;
  int reused_buffers = 0;
};

class Model {
 public:
  static std::unique_ptr<Model> load(const std::string& path);

  // Runs the graph. Input is copied in, output tensor is returned by reference.
  const Tensor& run(const float* input, size_t input_floats);

  const Tensor& output() const { return views_[static_cast<size_t>(output_id_)]; }
  const PlanStats& plan() const { return plan_; }
  const std::vector<Node>& nodes() const { return nodes_; }
  const std::vector<TensorDesc>& descs() const { return descs_; }
  const Shape& input_shape() const { return descs_[static_cast<size_t>(input_id_)].shape; }
  const std::string& name() const { return name_; }
  bool quantized() const { return quantized_; }

  // Force a conv implementation everywhere, for the benchmark ladder.
  void force_conv_impl(ConvImpl impl);

 private:
  void plan_memory();
  void bind_views();
  void check_no_aliasing() const;
  void check_dtypes() const;

  std::string name_;
  std::vector<TensorDesc> descs_;
  std::vector<Node> nodes_;
  std::vector<Tensor> views_;
  std::vector<uint8_t> weights_;
  // [K, N] transposes of float Linear weights, one entry per node
  std::vector<std::vector<float>> linear_wt_;
  Arena arena_;
  std::vector<float> workspace_;
  int input_id_ = -1, output_id_ = -1;
  bool quantized_ = false;
  PlanStats plan_;
};

}  // namespace ni
