#include "nanoinfer/tensor.h"

#include <cstdlib>
#include <stdexcept>

namespace ni {

std::string Tensor::shape_str() const {
  std::string s = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) s += ",";
    s += std::to_string(shape[i]);
  }
  return s + "]";
}

Arena::Arena(size_t bytes) {
  if (bytes) reserve(bytes);
}

Arena::~Arena() { std::free(base_); }

void Arena::reserve(size_t bytes) {
  if (bytes <= capacity_) return;
  std::free(base_);
  bytes = align_up(bytes);
  base_ = static_cast<uint8_t*>(std::aligned_alloc(kAlign, bytes));
  if (!base_) throw std::bad_alloc();
  capacity_ = bytes;
}

void* Arena::at(size_t offset) const {
  if (offset > capacity_) throw std::out_of_range("arena offset past end");
  return base_ + offset;
}

}  // namespace ni
