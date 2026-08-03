// Tensors and the arena they live in.
//
// Every intermediate tensor in a graph is a view into one contiguous arena.
// Sizes are known ahead of time because shapes are static, so the whole
// forward pass runs with zero allocations after warmup.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ni {

enum class DType : uint8_t { F32 = 0, I8 = 1, I32 = 2 };

inline size_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: return 4;
    case DType::I8: return 1;
    case DType::I32: return 4;
  }
  return 4;
}

inline const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::I8: return "i8";
    case DType::I32: return "i32";
  }
  return "?";
}

using Shape = std::vector<int>;

inline size_t numel(const Shape& s) {
  size_t n = 1;
  for (int d : s) n *= static_cast<size_t>(d);
  return n;
}

// A non-owning view of typed data with a shape.
struct Tensor {
  void* data = nullptr;
  Shape shape;
  DType dtype = DType::F32;
  // quantization parameters; scale == 0 means "not quantized"
  float scale = 0.0f;
  int32_t zero_point = 0;

  size_t size() const { return numel(shape); }
  size_t nbytes() const { return size() * dtype_size(dtype); }

  float* f32() { return static_cast<float*>(data); }
  const float* f32() const { return static_cast<const float*>(data); }
  int8_t* i8() { return static_cast<int8_t*>(data); }
  const int8_t* i8() const { return static_cast<const int8_t*>(data); }
  int32_t* i32() { return static_cast<int32_t*>(data); }
  const int32_t* i32() const { return static_cast<const int32_t*>(data); }

  int dim(int i) const { return shape[static_cast<size_t>(i)]; }
  int ndim() const { return static_cast<int>(shape.size()); }
  std::string shape_str() const;
};

// One aligned block of memory that all intermediates are carved out of.
class Arena {
 public:
  explicit Arena(size_t bytes = 0);
  ~Arena();
  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  void reserve(size_t bytes);
  void* at(size_t offset) const;
  size_t capacity() const { return capacity_; }

 private:
  uint8_t* base_ = nullptr;
  size_t capacity_ = 0;
};

constexpr size_t kAlign = 64;

inline size_t align_up(size_t n, size_t a = kAlign) { return (n + a - 1) / a * a; }

}  // namespace ni
