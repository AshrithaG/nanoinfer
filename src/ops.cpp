#include "nanoinfer/ops.h"

#include "nanoinfer/pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>

#if defined(NI_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace ni {
namespace {

int g_threads = 1;

// Below this much work, coordinating threads costs more than it saves.
constexpr int kMinItemsPerThread = 4;

template <typename F>
void parallel_for(int n, F&& body) {
  if (g_threads <= 1 || n < 2 * kMinItemsPerThread) {
    body(0, n);
    return;
  }
  ThreadPool::instance().run(n, std::function<void(int, int)>(body));
}

inline float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }

}  // namespace

int threads_used() { return g_threads; }
void set_threads(int n) {
  g_threads = std::max(1, n);
  ThreadPool::instance().resize(g_threads);
}

// ---------------------------------------------------------------- gemm

void gemm_naive(int M, int N, int K, const float* A, const float* B, float* C) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) acc += A[m * K + k] * B[k * N + n];
      C[m * N + n] = acc;
    }
  }
}

void gemm(int M, int N, int K, const float* A, const float* B, float* C) {
#if defined(NI_USE_ACCELERATE)
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N,
              0.0f, C, N);
#else
  // Blocked, k-innermost-accumulating fallback: keeps a row of C in registers
  // and streams B, which is far better than the naive triple loop.
  constexpr int MC = 64, NC = 256, KC = 128;
  std::memset(C, 0, sizeof(float) * static_cast<size_t>(M) * N);
  for (int kc = 0; kc < K; kc += KC) {
    const int kmax = std::min(K, kc + KC);
    for (int mc = 0; mc < M; mc += MC) {
      const int mmax = std::min(M, mc + MC);
      for (int nc = 0; nc < N; nc += NC) {
        const int nmax = std::min(N, nc + NC);
        for (int m = mc; m < mmax; ++m) {
          for (int k = kc; k < kmax; ++k) {
            const float a = A[m * K + k];
            if (a == 0.f) continue;
            const float* bp = B + k * N;
            float* cp = C + m * N;
            for (int n = nc; n < nmax; ++n) cp[n] += a * bp[n];
          }
        }
      }
    }
  }
#endif
}

// ---------------------------------------------------------------- conv2d

size_t conv2d_workspace_floats(const Tensor& x, const Tensor& w, const Tensor& out,
                               const ConvParams& p, ConvImpl impl) {
  if (impl != ConvImpl::Im2colGemm) return 0;
  const int C = x.dim(1) / p.groups;
  const int KH = w.dim(2), KW = w.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);
  // one im2col patch matrix: [C*KH*KW, OH*OW]
  return static_cast<size_t>(C) * KH * KW * OH * OW;
}

namespace {

void im2col(const float* x, int C, int H, int W, int KH, int KW, int OH, int OW,
            const ConvParams& p, float* col) {
  // col layout: [C*KH*KW, OH*OW]
  const int patch = OH * OW;
  for (int c = 0; c < C; ++c) {
    for (int kh = 0; kh < KH; ++kh) {
      for (int kw = 0; kw < KW; ++kw) {
        float* dst = col + ((c * KH + kh) * KW + kw) * patch;
        for (int oh = 0; oh < OH; ++oh) {
          const int ih = oh * p.stride_h - p.pad_h + kh;
          if (ih < 0 || ih >= H) {
            std::memset(dst + oh * OW, 0, sizeof(float) * static_cast<size_t>(OW));
            continue;
          }
          const float* src = x + (static_cast<size_t>(c) * H + ih) * W;
          for (int ow = 0; ow < OW; ++ow) {
            const int iw = ow * p.stride_w - p.pad_w + kw;
            dst[oh * OW + ow] = (iw < 0 || iw >= W) ? 0.f : src[iw];
          }
        }
      }
    }
  }
}

void conv2d_naive_impl(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out,
                       const ConvParams& p) {
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const int OC = w.dim(0), KH = w.dim(2), KW = w.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);
  const int cpg = C / p.groups, ocpg = OC / p.groups;
  const float* xp = x.f32();
  const float* wp = w.f32();
  float* op = out.f32();

  for (int n = 0; n < N; ++n) {
    parallel_for(OC, [&](int oc0, int oc1) {
      for (int oc = oc0; oc < oc1; ++oc) {
        const int g = oc / ocpg;
        for (int oh = 0; oh < OH; ++oh) {
          for (int ow = 0; ow < OW; ++ow) {
            float acc = b ? b->f32()[oc] : 0.f;
            for (int ic = 0; ic < cpg; ++ic) {
              const int c = g * cpg + ic;
              for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * p.stride_h - p.pad_h + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < KW; ++kw) {
                  const int iw = ow * p.stride_w - p.pad_w + kw;
                  if (iw < 0 || iw >= W) continue;
                  acc += xp[((static_cast<size_t>(n) * C + c) * H + ih) * W + iw] *
                         wp[((static_cast<size_t>(oc) * cpg + ic) * KH + kh) * KW + kw];
                }
              }
            }
            if (p.relu) acc = acc > 0.f ? acc : 0.f;
            op[((static_cast<size_t>(n) * OC + oc) * OH + oh) * OW + ow] = acc;
          }
        }
      }
    });
  }
}

// A 1x1 stride-1 unpadded convolution *is* a GEMM: the patch matrix im2col
// would build is bit-for-bit the input. Pointwise convolutions dominate
// depthwise-separable networks, so skipping that copy is worth a special case.
inline bool is_pointwise(const Tensor& w, const ConvParams& p) {
  return w.dim(2) == 1 && w.dim(3) == 1 && p.stride_h == 1 && p.stride_w == 1 &&
         p.pad_h == 0 && p.pad_w == 0;
}

void conv2d_im2col_impl(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out,
                        const ConvParams& p, float* ws) {
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const int OC = w.dim(0), KH = w.dim(2), KW = w.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);
  const int cpg = C / p.groups, ocpg = OC / p.groups;
  const int patch = OH * OW, Kdim = cpg * KH * KW;
  const bool pointwise = is_pointwise(w, p);

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < p.groups; ++g) {
      const float* xg = x.f32() + (static_cast<size_t>(n) * C + g * cpg) * H * W;
      if (!pointwise) im2col(xg, cpg, H, W, KH, KW, OH, OW, p, ws);
      const float* patches = pointwise ? xg : ws;
      const float* wg = w.f32() + static_cast<size_t>(g) * ocpg * Kdim;
      float* og = out.f32() + (static_cast<size_t>(n) * OC + g * ocpg) * patch;
      gemm(ocpg, patch, Kdim, wg, patches, og);
      // fused bias + relu epilogue, one pass over the output
      for (int oc = 0; oc < ocpg; ++oc) {
        float* row = og + static_cast<size_t>(oc) * patch;
        const float bias = b ? b->f32()[g * ocpg + oc] : 0.f;
        int i = 0;
#if defined(__ARM_NEON)
        const float32x4_t vb = vdupq_n_f32(bias);
        const float32x4_t vz = vdupq_n_f32(0.f);
        for (; i + 4 <= patch; i += 4) {
          float32x4_t v = vaddq_f32(vld1q_f32(row + i), vb);
          if (p.relu) v = vmaxq_f32(v, vz);
          vst1q_f32(row + i, v);
        }
#endif
        for (; i < patch; ++i) {
          float v = row[i] + bias;
          row[i] = (p.relu && v < 0.f) ? 0.f : v;
        }
      }
    }
  }
}

// Depthwise conv: one filter per channel, so im2col+GEMM degenerates to a
// tiny matmul per channel and loses. A direct loop with NEON accumulation is
// much better, and depthwise dominates mobile-style networks.
void conv2d_depthwise_impl(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out,
                           const ConvParams& p) {
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const int KH = w.dim(2), KW = w.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);

  for (int n = 0; n < N; ++n) {
    parallel_for(C, [&](int c0, int c1) {
      for (int c = c0; c < c1; ++c) {
        const float* xc = x.f32() + (static_cast<size_t>(n) * C + c) * H * W;
        const float* wc = w.f32() + static_cast<size_t>(c) * KH * KW;
        float* oc_ = out.f32() + (static_cast<size_t>(n) * C + c) * OH * OW;
        const float bias = b ? b->f32()[c] : 0.f;
        for (int oh = 0; oh < OH; ++oh) {
          float* orow = oc_ + static_cast<size_t>(oh) * OW;
          for (int ow = 0; ow < OW; ++ow) orow[ow] = bias;
          for (int kh = 0; kh < KH; ++kh) {
            const int ih = oh * p.stride_h - p.pad_h + kh;
            if (ih < 0 || ih >= H) continue;
            const float* xrow = xc + static_cast<size_t>(ih) * W;
            for (int kw = 0; kw < KW; ++kw) {
              const float wv = wc[kh * KW + kw];
              if (wv == 0.f) continue;
              int ow = 0;
              // fast interior span where no bounds check is needed
              if (p.stride_w == 1) {
                const int lo = std::max(0, p.pad_h * 0 + (p.pad_w - kw));
                const int hi = std::min(OW, W - kw + p.pad_w);
#if defined(__ARM_NEON)
                const float32x4_t vw = vdupq_n_f32(wv);
                ow = lo;
                for (; ow + 4 <= hi; ow += 4) {
                  const int iw = ow - p.pad_w + kw;
                  float32x4_t vo = vld1q_f32(orow + ow);
                  vo = vfmaq_f32(vo, vld1q_f32(xrow + iw), vw);
                  vst1q_f32(orow + ow, vo);
                }
#else
                ow = lo;
#endif
                for (; ow < hi; ++ow) orow[ow] += xrow[ow - p.pad_w + kw] * wv;
                continue;
              }
              for (; ow < OW; ++ow) {
                const int iw = ow * p.stride_w - p.pad_w + kw;
                if (iw < 0 || iw >= W) continue;
                orow[ow] += xrow[iw] * wv;
              }
            }
          }
          if (p.relu) {
            for (int ow = 0; ow < OW; ++ow)
              if (orow[ow] < 0.f) orow[ow] = 0.f;
          }
        }
      }
    });
  }
}

}  // namespace

void conv2d(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out,
            const ConvParams& p, ConvImpl impl, float* ws, size_t ws_floats) {
  if (impl == ConvImpl::Im2colGemm) {
    if (ws_floats < conv2d_workspace_floats(x, w, out, p, impl))
      throw std::runtime_error("conv2d: workspace too small");
    conv2d_im2col_impl(x, w, b, out, p, ws);
  } else if (impl == ConvImpl::DepthwiseNeon) {
    conv2d_depthwise_impl(x, w, b, out, p);
  } else {
    conv2d_naive_impl(x, w, b, out, p);
  }
}

// ---------------------------------------------------------------- other ops

void linear(const Tensor& x, const Tensor& w, const Tensor* b, Tensor& out, bool do_relu,
            const float* w_t) {
  const int M = x.dim(0), K = x.dim(1), N = w.dim(0);
  // w is [N, K] but GEMM wants [K, N]; w_t holds that transpose, built once at
  // load time.
  gemm(M, N, K, x.f32(), w_t, out.f32());
  float* op = out.f32();
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float v = op[static_cast<size_t>(m) * N + n] + (b ? b->f32()[n] : 0.f);
      op[static_cast<size_t>(m) * N + n] = (do_relu && v < 0.f) ? 0.f : v;
    }
}

void relu(Tensor& x) {
  float* p = x.f32();
  const size_t n = x.size();
  size_t i = 0;
#if defined(__ARM_NEON)
  const float32x4_t vz = vdupq_n_f32(0.f);
  for (; i + 4 <= n; i += 4) vst1q_f32(p + i, vmaxq_f32(vld1q_f32(p + i), vz));
#endif
  for (; i < n; ++i)
    if (p[i] < 0.f) p[i] = 0.f;
}

void add(const Tensor& a, const Tensor& b, Tensor& out, bool do_relu) {
  const size_t n = a.size();
  const float* ap = a.f32();
  const float* bp = b.f32();
  float* op = out.f32();
  size_t i = 0;
#if defined(__ARM_NEON)
  const float32x4_t vz = vdupq_n_f32(0.f);
  for (; i + 4 <= n; i += 4) {
    float32x4_t v = vaddq_f32(vld1q_f32(ap + i), vld1q_f32(bp + i));
    if (do_relu) v = vmaxq_f32(v, vz);
    vst1q_f32(op + i, v);
  }
#endif
  for (; i < n; ++i) {
    float v = ap[i] + bp[i];
    op[i] = (do_relu && v < 0.f) ? 0.f : v;
  }
}

// Shape-only ops still have to respect dtype. Reading an int8 tensor through a
// float pointer produces garbage that looks like a numerical accuracy problem,
// so both pooling ops branch explicitly and the graph asserts the combinations
// it actually supports.
void maxpool2d(const Tensor& x, Tensor& out, int k, int stride, int pad) {
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);
  if (x.dtype != out.dtype)
    throw std::runtime_error("maxpool2d: input and output dtype must match");

  for (int n = 0; n < N; ++n) {
    parallel_for(C, [&](int c0, int c1) {
      for (int c = c0; c < c1; ++c) {
        const size_t in_off = (static_cast<size_t>(n) * C + c) * H * W;
        const size_t out_off = (static_cast<size_t>(n) * C + c) * OH * OW;
        for (int oh = 0; oh < OH; ++oh)
          for (int ow = 0; ow < OW; ++ow) {
            // max is monotonic under affine quantization, so pooling int8
            // codes directly gives the same answer as pooling real values
            if (x.dtype == DType::I8) {
              int best = -128;
              bool any = false;
              for (int kh = 0; kh < k; ++kh) {
                const int ih = oh * stride - pad + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < k; ++kw) {
                  const int iw = ow * stride - pad + kw;
                  if (iw < 0 || iw >= W) continue;
                  best = std::max(best,
                                  static_cast<int>(x.i8()[in_off +
                                                          static_cast<size_t>(ih) * W + iw]));
                  any = true;
                }
              }
              out.i8()[out_off + static_cast<size_t>(oh) * OW + ow] =
                  static_cast<int8_t>(any ? best : x.zero_point);
            } else {
              float best = -INFINITY;
              for (int kh = 0; kh < k; ++kh) {
                const int ih = oh * stride - pad + kh;
                if (ih < 0 || ih >= H) continue;
                for (int kw = 0; kw < k; ++kw) {
                  const int iw = ow * stride - pad + kw;
                  if (iw < 0 || iw >= W) continue;
                  best = std::max(best, x.f32()[in_off + static_cast<size_t>(ih) * W + iw]);
                }
              }
              out.f32()[out_off + static_cast<size_t>(oh) * OW + ow] =
                  best == -INFINITY ? 0.f : best;
            }
          }
      }
    });
  }
}

void avgpool_global(const Tensor& x, Tensor& out) {
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const float inv = 1.0f / static_cast<float>(H * W);
  const int hw = H * W;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      const size_t off = (static_cast<size_t>(n) * C + c) * hw;
      const size_t o = static_cast<size_t>(n) * C + c;
      if (x.dtype == DType::I8) {
        // averaging is not monotonic-safe in the code domain: sum in int32,
        // then dequantize, so the f32 consumer downstream sees real values
        int32_t s = 0;
        for (int i = 0; i < hw; ++i) s += x.i8()[off + static_cast<size_t>(i)];
        const float mean_code = static_cast<float>(s) * inv;
        const float real = (mean_code - static_cast<float>(x.zero_point)) * x.scale;
        if (out.dtype == DType::F32) {
          out.f32()[o] = real;
        } else {
          const float q = std::round(real / out.scale) +
                          static_cast<float>(out.zero_point);
          out.i8()[o] = static_cast<int8_t>(clampf(q, -128.f, 127.f));
        }
      } else {
        float s = 0.f;
        for (int i = 0; i < hw; ++i) s += x.f32()[off + static_cast<size_t>(i)];
        out.f32()[o] = s * inv;
      }
    }
}

void softmax(const Tensor& x, Tensor& out) {
  const int N = x.ndim() > 1 ? x.dim(0) : 1;
  const int C = static_cast<int>(x.size()) / N;
  for (int n = 0; n < N; ++n) {
    const float* xp = x.f32() + static_cast<size_t>(n) * C;
    float* op = out.f32() + static_cast<size_t>(n) * C;
    float mx = xp[0];
    for (int i = 1; i < C; ++i) mx = std::max(mx, xp[i]);
    float sum = 0.f;
    for (int i = 0; i < C; ++i) {
      op[i] = std::exp(xp[i] - mx);
      sum += op[i];
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i < C; ++i) op[i] *= inv;
  }
}

void flatten_copy(const Tensor& x, Tensor& out) {
  std::memcpy(out.data, x.data, x.nbytes());
}

// ---------------------------------------------------------------- int8

void quantize(const Tensor& x, Tensor& out) {
  const float inv = 1.0f / out.scale;
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i) {
    const float v = std::round(x.f32()[i] * inv) + static_cast<float>(out.zero_point);
    out.i8()[i] = static_cast<int8_t>(clampf(v, -128.f, 127.f));
  }
}

void dequantize(const Tensor& x, Tensor& out) {
  const size_t n = x.size();
  for (size_t i = 0; i < n; ++i)
    out.f32()[i] = static_cast<float>(x.i8()[i] - x.zero_point) * x.scale;
}

namespace {

// int8 im2col: patch matrix laid out [OH*OW, Kdim] so each output pixel's
// receptive field is contiguous. That is the orientation the dot-product
// accumulation below wants, and it is transposed relative to the float path.
void im2col_i8(const int8_t* x, int C, int H, int W, int KH, int KW, int OH, int OW,
               const ConvParams& p, int8_t zero, int8_t* col, int Kpad, int Kdim) {
  for (int oh = 0; oh < OH; ++oh) {
    for (int ow = 0; ow < OW; ++ow) {
      int8_t* dst = col + (static_cast<size_t>(oh) * OW + ow) * Kpad;
      int k = 0;
      for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < KH; ++kh) {
          const int ih = oh * p.stride_h - p.pad_h + kh;
          for (int kw = 0; kw < KW; ++kw, ++k) {
            const int iw = ow * p.stride_w - p.pad_w + kw;
            dst[k] = (ih < 0 || ih >= H || iw < 0 || iw >= W)
                         ? zero
                         : x[(static_cast<size_t>(c) * H + ih) * W + iw];
          }
        }
      }
      // zero the alignment tail so the padded dot product contributes nothing
      if (Kpad > Kdim) std::memset(dst + Kdim, 0, static_cast<size_t>(Kpad - Kdim));
    }
  }
}

// sum over k of a[k] * b[k], int32 accumulate. Activations are quantized
// symmetrically (zero point 0), so there is no offset to subtract in the hot
// loop; the asserted precondition keeps that assumption honest.
inline int32_t dot_i8(const int8_t* a, const int8_t* b, int n) {
  int32_t acc = 0;
  int i = 0;
#if defined(NI_USE_DOTPROD)
  // SDOT: four int8 products accumulated into each of four int32 lanes, so one
  // instruction covers 16 multiply-accumulates.
  int32x4_t v0 = vdupq_n_s32(0), v1 = vdupq_n_s32(0);
  for (; i + 32 <= n; i += 32) {
    v0 = vdotq_s32(v0, vld1q_s8(a + i), vld1q_s8(b + i));
    v1 = vdotq_s32(v1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= n; i += 16) v0 = vdotq_s32(v0, vld1q_s8(a + i), vld1q_s8(b + i));
  acc += vaddvq_s32(vaddq_s32(v0, v1));
#elif defined(__ARM_NEON)
  int32x4_t v0 = vdupq_n_s32(0);
  for (; i + 16 <= n; i += 16) {
    const int8x16_t va = vld1q_s8(a + i);
    const int8x16_t vb = vld1q_s8(b + i);
    // widening multiply-accumulate: 16 int8 lanes -> 8 int16 -> 4 int32
    v0 = vpadalq_s16(v0, vmull_s8(vget_low_s8(va), vget_low_s8(vb)));
    v0 = vpadalq_s16(v0, vmull_s8(vget_high_s8(va), vget_high_s8(vb)));
  }
  acc += vaddvq_s32(v0);
#endif
  for (; i < n; ++i) acc += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  return acc;
}

}  // namespace

void conv2d_i8(const Tensor& x, const Tensor& w, const Tensor* bias_f32, Tensor& out,
               const ConvParams& p, const float* w_scales, float* ws, size_t ws_floats) {
  // int8 x int8 -> int32 accumulate, then requantize by
  // (x.scale * w_scale[oc]) / out.scale. Weight scales are per output channel,
  // which is what keeps depthwise layers accurate.
  const int N = x.dim(0), C = x.dim(1), H = x.dim(2), W = x.dim(3);
  const int OC = w.dim(0), KH = w.dim(2), KW = w.dim(3);
  const int OH = out.dim(2), OW = out.dim(3);
  const int cpg = C / p.groups, ocpg = OC / p.groups;
  const int patch = OH * OW, Kdim = cpg * KH * KW;

  if (x.zero_point != 0 || out.zero_point != 0)
    throw std::runtime_error("conv2d_i8: only symmetric (zero point 0) quantization "
                             "is implemented");

  // Depthwise: Kdim is 9 for a 3x3 filter, so an im2col matrix padded to a
  // 16-byte stride would be 44% zero padding, and with one output channel per
  // group there is nothing left to parallelize over. Go direct and parallelize
  // across channels instead.
  if (p.groups == C && ocpg == 1) {
    for (int n = 0; n < N; ++n) {
      parallel_for(C, [&](int c0, int c1) {
        for (int c = c0; c < c1; ++c) {
          const int8_t* xc = x.i8() + (static_cast<size_t>(n) * C + c) * H * W;
          const int8_t* wc = w.i8() + static_cast<size_t>(c) * Kdim;
          int8_t* oc_ = out.i8() + (static_cast<size_t>(n) * C + c) * patch;
          const float out_mult = x.scale * w_scales[c] / out.scale;
          const float bias_q = bias_f32 ? bias_f32->f32()[c] / out.scale : 0.f;
          for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
              int32_t acc = 0;
              int k = 0;
              for (int kh = 0; kh < KH; ++kh) {
                const int ih = oh * p.stride_h - p.pad_h + kh;
                if (ih < 0 || ih >= H) { k += KW; continue; }
                const int8_t* xrow = xc + static_cast<size_t>(ih) * W;
                for (int kw = 0; kw < KW; ++kw, ++k) {
                  const int iw = ow * p.stride_w - p.pad_w + kw;
                  if (iw < 0 || iw >= W) continue;
                  acc += static_cast<int32_t>(xrow[iw]) * static_cast<int32_t>(wc[k]);
                }
              }
              float v = static_cast<float>(acc) * out_mult + bias_q;
              if (p.relu && v < 0.f) v = 0.f;
              oc_[static_cast<size_t>(oh) * OW + ow] =
                  static_cast<int8_t>(clampf(std::round(v), -128.f, 127.f));
            }
          }
        }
      });
    }
    return;
  }

  // The patch matrix is padded so every row starts 16-byte aligned and its
  // tail is zero, which lets the dot product run full vectors with no scalar
  // remainder and no masking.
  const int Kpad = (Kdim + 15) / 16 * 16;
  const size_t need_bytes = static_cast<size_t>(patch) * Kpad;
  if (ws_floats * sizeof(float) < need_bytes)
    throw std::runtime_error("conv2d_i8: workspace too small");
  int8_t* col = reinterpret_cast<int8_t*>(ws);

  // weights get the same padded stride, packed once per call
  static thread_local std::vector<int8_t> wpack;
  wpack.assign(static_cast<size_t>(OC) * Kpad, 0);
  for (int oc = 0; oc < OC; ++oc)
    std::memcpy(wpack.data() + static_cast<size_t>(oc) * Kpad,
                w.i8() + static_cast<size_t>(oc) * Kdim, static_cast<size_t>(Kdim));

  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < p.groups; ++g) {
      const int8_t* xg = x.i8() + (static_cast<size_t>(n) * C + g * cpg) * H * W;
      im2col_i8(xg, cpg, H, W, KH, KW, OH, OW, p, 0, col, Kpad, Kdim);

      parallel_for(ocpg, [&](int oc0, int oc1) {
        for (int oc = oc0; oc < oc1; ++oc) {
          const int oc_global = g * ocpg + oc;
          const int8_t* wrow = wpack.data() + static_cast<size_t>(oc_global) * Kpad;
          const float out_mult = x.scale * w_scales[oc_global] / out.scale;
          const float bias_q = bias_f32 ? bias_f32->f32()[oc_global] / out.scale : 0.f;
          int8_t* orow = out.i8() + (static_cast<size_t>(n) * OC + oc_global) * patch;
          for (int q = 0; q < patch; ++q) {
            const int32_t acc = dot_i8(col + static_cast<size_t>(q) * Kpad, wrow, Kpad);
            float v = static_cast<float>(acc) * out_mult + bias_q;
            if (p.relu && v < 0.f) v = 0.f;
            orow[q] = static_cast<int8_t>(clampf(std::round(v), -128.f, 127.f));
          }
        }
      });
    }
  }
}

void linear_i8(const Tensor& x, const Tensor& w, const Tensor* bias_f32, Tensor& out,
               const float* w_scales, bool do_relu) {
  const int M = x.dim(0), K = x.dim(1), N = w.dim(0);
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < K; ++k) {
        const int32_t xv = x.i8()[static_cast<size_t>(m) * K + k] - x.zero_point;
        acc += xv * w.i8()[static_cast<size_t>(n) * K + k];
      }
      float v = static_cast<float>(acc) * x.scale * w_scales[n];
      if (bias_f32) v += bias_f32->f32()[n];
      if (do_relu && v < 0.f) v = 0.f;
      if (out.dtype == DType::F32) {
        out.f32()[static_cast<size_t>(m) * N + n] = v;
      } else {
        const float q = std::round(v / out.scale) + static_cast<float>(out.zero_point);
        out.i8()[static_cast<size_t>(m) * N + n] =
            static_cast<int8_t>(clampf(q, -128.f, 127.f));
      }
    }
  }
}

}  // namespace ni
