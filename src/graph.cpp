#include "nanoinfer/graph.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace ni {

const char* op_name(Op op) {
  switch (op) {
    case Op::Conv2d: return "conv2d";
    case Op::DepthwiseConv2d: return "dwconv2d";
    case Op::Linear: return "linear";
    case Op::Relu: return "relu";
    case Op::MaxPool2d: return "maxpool2d";
    case Op::GlobalAvgPool: return "gap";
    case Op::Add: return "add";
    case Op::Softmax: return "softmax";
    case Op::Flatten: return "flatten";
    case Op::Quantize: return "quantize";
    case Op::Dequantize: return "dequantize";
  }
  return "?";
}

namespace {

Op parse_op(const std::string& s) {
  if (s == "conv2d") return Op::Conv2d;
  if (s == "dwconv2d") return Op::DepthwiseConv2d;
  if (s == "linear") return Op::Linear;
  if (s == "relu") return Op::Relu;
  if (s == "maxpool2d") return Op::MaxPool2d;
  if (s == "gap") return Op::GlobalAvgPool;
  if (s == "add") return Op::Add;
  if (s == "softmax") return Op::Softmax;
  if (s == "flatten") return Op::Flatten;
  if (s == "quantize") return Op::Quantize;
  if (s == "dequantize") return Op::Dequantize;
  throw std::runtime_error("unknown op: " + s);
}

DType parse_dtype(const std::string& s) {
  if (s == "f32") return DType::F32;
  if (s == "i8") return DType::I8;
  if (s == "i32") return DType::I32;
  throw std::runtime_error("unknown dtype: " + s);
}

}  // namespace

// Format (text header, then a binary blob):
//   NGM1
//   name <model name>
//   tensors <count>
//   T <name> <dtype> <is_weight> <scale> <zp> <offset> <ndim> <d0..dn>
//   nodes <count>
//   N <op> <nin> <in...> <nout> <out...> <k=v ...>
//   io <input_id> <output_id>
//   DATA
//   <weight bytes>
std::unique_ptr<Model> Model::load(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);

  std::string magic;
  std::getline(f, magic);
  if (magic.rfind("NGM1", 0) != 0) throw std::runtime_error("not an NGM1 file: " + path);

  auto m = std::make_unique<Model>();
  std::string line;
  size_t weight_bytes = 0;

  while (std::getline(f, line)) {
    if (line == "DATA") break;
    std::istringstream ss(line);
    std::string kind;
    ss >> kind;
    if (kind == "name") {
      ss >> m->name_;
    } else if (kind == "weight_bytes") {
      ss >> weight_bytes;
    } else if (kind == "T") {
      TensorDesc d;
      std::string dt;
      int is_w = 0, ndim = 0;
      ss >> d.name >> dt >> is_w >> d.scale >> d.zero_point >> d.offset >> ndim;
      d.dtype = parse_dtype(dt);
      d.is_weight = is_w != 0;
      d.shape.resize(static_cast<size_t>(ndim));
      for (int i = 0; i < ndim; ++i) ss >> d.shape[static_cast<size_t>(i)];
      d.nbytes = numel(d.shape) * dtype_size(d.dtype);
      if (d.dtype == DType::I8) m->quantized_ = true;
      m->descs_.push_back(std::move(d));
    } else if (kind == "N") {
      Node n;
      std::string op;
      int nin = 0, nout = 0;
      ss >> op;
      n.op = parse_op(op);
      ss >> nin;
      n.inputs.resize(static_cast<size_t>(nin));
      for (int i = 0; i < nin; ++i) ss >> n.inputs[static_cast<size_t>(i)];
      ss >> nout;
      n.outputs.resize(static_cast<size_t>(nout));
      for (int i = 0; i < nout; ++i) ss >> n.outputs[static_cast<size_t>(i)];
      std::string kv;
      while (ss >> kv) {
        const auto eq = kv.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = kv.substr(0, eq);
        const std::string v = kv.substr(eq + 1);
        if (k == "stride") n.conv.stride_h = n.conv.stride_w = std::stoi(v);
        else if (k == "stride_h") n.conv.stride_h = std::stoi(v);
        else if (k == "stride_w") n.conv.stride_w = std::stoi(v);
        else if (k == "pad") n.conv.pad_h = n.conv.pad_w = std::stoi(v);
        else if (k == "pad_h") n.conv.pad_h = std::stoi(v);
        else if (k == "pad_w") n.conv.pad_w = std::stoi(v);
        else if (k == "groups") n.conv.groups = std::stoi(v);
        else if (k == "relu") { n.conv.relu = n.relu = std::stoi(v) != 0; }
        else if (k == "k") n.pool_k = std::stoi(v);
        else if (k == "pool_stride") n.pool_stride = std::stoi(v);
        else if (k == "pool_pad") n.pool_pad = std::stoi(v);
      }
      if (n.op == Op::DepthwiseConv2d) n.impl = ConvImpl::DepthwiseNeon;
      m->nodes_.push_back(std::move(n));
    } else if (kind == "io") {
      ss >> m->input_id_ >> m->output_id_;
    }
  }

  m->weights_.resize(weight_bytes);
  if (weight_bytes) f.read(reinterpret_cast<char*>(m->weights_.data()),
                           static_cast<std::streamsize>(weight_bytes));
  if (!f && weight_bytes) throw std::runtime_error("truncated weights in " + path);

  // per-output-channel weight scales live right after the weight tensor
  for (auto& n : m->nodes_) {
    if ((n.op == Op::Conv2d || n.op == Op::DepthwiseConv2d || n.op == Op::Linear) &&
        n.inputs.size() >= 2) {
      const auto& wd = m->descs_[static_cast<size_t>(n.inputs[1])];
      if (wd.dtype == DType::I8) {
        const int oc = wd.shape[0];
        n.w_scales.resize(static_cast<size_t>(oc));
        // scales stored as f32 immediately following the int8 weights
        const auto* sp = reinterpret_cast<const float*>(m->weights_.data() + wd.offset +
                                                        numel(wd.shape));
        std::memcpy(n.w_scales.data(), sp, sizeof(float) * static_cast<size_t>(oc));
      }
    }
  }

  m->plan_memory();
  m->bind_views();
  return m;
}

void Model::plan_memory() {
  // Last node that reads each tensor; after that its buffer can be recycled.
  const size_t nt = descs_.size();
  std::vector<int> last_use(nt, -1);
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (int t : nodes_[i].inputs) last_use[static_cast<size_t>(t)] = static_cast<int>(i);
  last_use[static_cast<size_t>(output_id_)] = static_cast<int>(nodes_.size());

  struct Block { size_t offset, bytes; int free_after; };
  std::vector<Block> blocks;
  size_t high_water = 0, naive_total = 0;
  int reused = 0;

  auto place = [&](int tid, int node_idx) {
    auto& d = descs_[static_cast<size_t>(tid)];
    if (d.is_weight) return;
    const size_t want = align_up(d.nbytes);
    naive_total += want;
    // Reuse the first block that is big enough and was released *strictly
    // before* this node. `free_after == node_idx` means the current node still
    // reads that tensor, and handing its buffer out as this node's output makes
    // input and output alias -- which is invisible for an elementwise op and
    // silently corrupts a GEMM.
    for (auto& b : blocks) {
      if (b.free_after >= 0 && b.free_after < node_idx && b.bytes >= want) {
        d.offset = b.offset;
        b.free_after = last_use[static_cast<size_t>(tid)];
        ++reused;
        return;
      }
    }
    d.offset = high_water;
    high_water += want;
    blocks.push_back({d.offset, want, last_use[static_cast<size_t>(tid)]});
  };

  if (input_id_ >= 0) place(input_id_, 0);
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (int t : nodes_[i].outputs) place(t, static_cast<int>(i));

  arena_.reserve(high_water ? high_water : kAlign);

  // workspace: the largest single conv patch matrix, shared by every conv.
  // The int8 path needs one byte per element instead of four, but it also runs
  // on depthwise nodes, so both are considered.
  size_t ws_floats = 0;
  for (const auto& n : nodes_) {
    if (n.op != Op::Conv2d && n.op != Op::DepthwiseConv2d) continue;
    const auto& xd = descs_[static_cast<size_t>(n.inputs[0])];
    const auto& wd = descs_[static_cast<size_t>(n.inputs[1])];
    const auto& od = descs_[static_cast<size_t>(n.outputs[0])];
    const int C = xd.shape[1] / n.conv.groups;
    const size_t kdim = static_cast<size_t>(C) * wd.shape[2] * wd.shape[3];
    const size_t patch = static_cast<size_t>(od.shape[2]) * od.shape[3];
    size_t as_floats;
    if (wd.dtype == DType::I8) {
      // the int8 patch matrix pads each row up to a 16-byte multiple
      const size_t kpad = (kdim + 15) / 16 * 16;
      as_floats = (patch * kpad + sizeof(float) - 1) / sizeof(float);
    } else {
      as_floats = kdim * patch;
    }
    if (n.op == Op::Conv2d || wd.dtype == DType::I8)
      ws_floats = std::max(ws_floats, as_floats);
  }
  workspace_.assign(ws_floats ? ws_floats : 1, 0.f);

  // Pre-transpose float Linear weights into [K, N]; doing it per call costs an
  // O(K*N) copy every inference, which on these models rivals the matmul.
  linear_wt_.assign(nodes_.size(), {});
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    if (n.op != Op::Linear || n.inputs.size() < 2) continue;
    const auto& wd = descs_[static_cast<size_t>(n.inputs[1])];
    if (wd.dtype != DType::F32) continue;
    const int N_out = wd.shape[0], K = wd.shape[1];
    const auto* src = reinterpret_cast<const float*>(weights_.data() + wd.offset);
    auto& dst = linear_wt_[i];
    dst.resize(static_cast<size_t>(K) * N_out);
    for (int o = 0; o < N_out; ++o)
      for (int k = 0; k < K; ++k)
        dst[static_cast<size_t>(k) * N_out + o] = src[static_cast<size_t>(o) * K + k];
  }

  plan_.arena_bytes = arena_.capacity();
  plan_.sum_of_activation_bytes = naive_total;
  plan_.workspace_bytes = workspace_.size() * sizeof(float);
  plan_.reused_buffers = reused;
}

void Model::bind_views() {
  views_.resize(descs_.size());
  for (size_t i = 0; i < descs_.size(); ++i) {
    const auto& d = descs_[i];
    Tensor t;
    t.shape = d.shape;
    t.dtype = d.dtype;
    t.scale = d.scale;
    t.zero_point = d.zero_point;
    t.data = d.is_weight ? static_cast<void*>(weights_.data() + d.offset)
                         : arena_.at(d.offset);
    views_[i] = t;
  }
  check_no_aliasing();
  check_dtypes();
}

// Every op reads its tensors through a typed pointer, so a graph that hands an
// int8 tensor to a float-only path produces plausible-looking garbage rather
// than an error. Checking the combinations up front turns that into a load
// failure with a name attached.
void Model::check_dtypes() const {
  auto dt = [&](int id) { return descs_[static_cast<size_t>(id)].dtype; };
  auto nm = [&](int id) { return descs_[static_cast<size_t>(id)].name; };
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    const auto in = dt(n.inputs[0]);
    const auto out = dt(n.outputs[0]);
    const std::string where = std::string(op_name(n.op)) + " node " + std::to_string(i);
    switch (n.op) {
      case Op::Relu:
      case Op::Softmax:
        if (in != DType::F32 || out != DType::F32)
          throw std::runtime_error(where + ": only f32 is implemented, got " +
                                   dtype_name(in) + "->" + dtype_name(out));
        break;
      case Op::MaxPool2d:
      case Op::Flatten:
        if (in != out)
          throw std::runtime_error(where + ": dtype must pass through unchanged (" +
                                   nm(n.inputs[0]) + " " + dtype_name(in) + " -> " +
                                   nm(n.outputs[0]) + " " + dtype_name(out) + ")");
        break;
      case Op::Quantize:
        if (in != DType::F32 || out != DType::I8)
          throw std::runtime_error(where + ": expected f32->i8");
        break;
      case Op::Dequantize:
        if (in != DType::I8 || out != DType::F32)
          throw std::runtime_error(where + ": expected i8->f32");
        break;
      case Op::Conv2d:
      case Op::DepthwiseConv2d:
      case Op::Linear: {
        const auto w = dt(n.inputs[1]);
        if (w == DType::I8 && in != DType::I8)
          throw std::runtime_error(where + ": int8 weights need an int8 input, got " +
                                   dtype_name(in));
        if (w == DType::F32 && (in != DType::F32 || out != DType::F32))
          throw std::runtime_error(where + ": float weights need float activations");
        break;
      }
      case Op::Add:
      case Op::GlobalAvgPool:
        break;
    }
  }
}

// A buffer-reuse bug is close to invisible: results are quietly wrong for
// exactly those ops that read a location after writing it, and every test on a
// single layer still passes. So the plan asserts its own key invariant rather
// than trusting the liveness analysis to be right.
void Model::check_no_aliasing() const {
  auto overlaps = [&](int a, int b) {
    const auto& x = descs_[static_cast<size_t>(a)];
    const auto& y = descs_[static_cast<size_t>(b)];
    if (x.is_weight || y.is_weight) return false;
    return x.offset < y.offset + align_up(y.nbytes) &&
           y.offset < x.offset + align_up(x.nbytes);
  };
  for (size_t i = 0; i < nodes_.size(); ++i) {
    const auto& n = nodes_[i];
    // relu and flatten are safe to run in place; nothing else is
    const bool in_place_ok = n.op == Op::Relu || n.op == Op::Flatten;
    if (in_place_ok) continue;
    for (int out : n.outputs)
      for (int in : n.inputs)
        if (overlaps(out, in))
          throw std::runtime_error(
              std::string("memory plan aliases input and output of node ") +
              std::to_string(i) + " (" + op_name(n.op) +
              "): tensors " + descs_[static_cast<size_t>(in)].name + " and " +
              descs_[static_cast<size_t>(out)].name);
  }
}

void Model::force_conv_impl(ConvImpl impl) {
  for (auto& n : nodes_)
    if (n.op == Op::Conv2d) n.impl = impl;
}

const Tensor& Model::run(const float* input, size_t input_floats) {
  Tensor& in = views_[static_cast<size_t>(input_id_)];
  if (input_floats != in.size())
    throw std::runtime_error("input size mismatch: model wants " +
                             std::to_string(in.size()) + ", got " +
                             std::to_string(input_floats));
  std::memcpy(in.data, input, input_floats * sizeof(float));

  for (size_t ni = 0; ni < nodes_.size(); ++ni) {
    const auto& n = nodes_[ni];
    Tensor& out = views_[static_cast<size_t>(n.outputs[0])];
    const Tensor& a = views_[static_cast<size_t>(n.inputs[0])];
    switch (n.op) {
      case Op::Conv2d:
      case Op::DepthwiseConv2d: {
        const Tensor& w = views_[static_cast<size_t>(n.inputs[1])];
        const Tensor* b = n.inputs.size() > 2 ? &views_[static_cast<size_t>(n.inputs[2])]
                                              : nullptr;
        if (w.dtype == DType::I8)
          conv2d_i8(a, w, b, out, n.conv, n.w_scales.data(), workspace_.data(),
                    workspace_.size());
        else
          conv2d(a, w, b, out, n.conv, n.impl, workspace_.data(), workspace_.size());
        break;
      }
      case Op::Linear: {
        const Tensor& w = views_[static_cast<size_t>(n.inputs[1])];
        const Tensor* b = n.inputs.size() > 2 ? &views_[static_cast<size_t>(n.inputs[2])]
                                              : nullptr;
        if (w.dtype == DType::I8)
          linear_i8(a, w, b, out, n.w_scales.data(), n.relu);
        else
          linear(a, w, b, out, n.relu, linear_wt_[ni].data());
        break;
      }
      case Op::Relu:
        if (out.data != a.data) std::memcpy(out.data, a.data, a.nbytes());
        relu(out);
        break;
      case Op::MaxPool2d:
        maxpool2d(a, out, n.pool_k, n.pool_stride, n.pool_pad);
        break;
      case Op::GlobalAvgPool:
        avgpool_global(a, out);
        break;
      case Op::Add:
        add(a, views_[static_cast<size_t>(n.inputs[1])], out, n.relu);
        break;
      case Op::Softmax:
        softmax(a, out);
        break;
      case Op::Flatten:
        if (out.data != a.data) flatten_copy(a, out);
        break;
      case Op::Quantize:
        quantize(a, out);
        break;
      case Op::Dequantize:
        dequantize(a, out);
        break;
    }
  }
  return output();
}

}  // namespace ni
