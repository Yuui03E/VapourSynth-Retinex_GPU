// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cstddef>
#include <memory>
#include <vector>

namespace retinex {
enum class Stage { MSRCP, Kirsch, TCanny, Mask };
struct Parameters {
    int width = 0, height = 0, bits = 8, device = 0, op = 1;
    bool floating = false, fast = false;
    double lower = .001, upper = .005;
    float blur = 1.f;
    std::vector<double> sigma{50, 200, 350};
    Stage stage = Stage::Mask;
};
class Workspace {
  public:
    explicit Workspace(const Parameters &parameters);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    void process(const void *src, ptrdiff_t srcStride, void *dst, ptrdiff_t dstStride);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace retinex
