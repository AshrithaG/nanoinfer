// Run a model on one input and print the output. Used by the parity harness,
// which compares these numbers against PyTorch.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nanoinfer/graph.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: ni_run model.ngm [input.bin] [--threads N] [--impl naive|im2col]\n");
    return 2;
  }
  const std::string model_path = argv[1];
  std::string input_path;
  int threads = 1;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc) threads = std::atoi(argv[++i]);
    else if (a == "--impl" && i + 1 < argc) {
      const std::string v = argv[++i];
      (void)v;
    } else if (a.rfind("--", 0) != 0) input_path = a;
  }
  ni::set_threads(threads);

  try {
    auto model = ni::Model::load(model_path);
    const size_t n = ni::numel(model->input_shape());
    std::vector<float> input(n, 0.f);
    if (!input_path.empty()) {
      std::ifstream f(input_path, std::ios::binary);
      if (!f) {
        std::fprintf(stderr, "cannot open %s\n", input_path.c_str());
        return 1;
      }
      f.read(reinterpret_cast<char*>(input.data()),
             static_cast<std::streamsize>(n * sizeof(float)));
    }

    const auto& out = model->run(input.data(), input.size());
    for (size_t i = 0; i < out.size(); ++i)
      std::printf("%.6f%s", out.f32()[i], i + 1 == out.size() ? "\n" : " ");
  } catch (const std::exception& e) {
    // a load-time invariant failure should read as a clear message, not SIGABRT
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
