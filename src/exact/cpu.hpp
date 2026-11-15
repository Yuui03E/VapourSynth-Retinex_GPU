// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "gpu.hpp"
namespace retinex {
// Same arithmetic and boundary conventions as the exact CUDA pipeline.
class CpuWorkspace {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit CpuWorkspace(const Parameters &p);
    ~CpuWorkspace();
    void process(const void *, ptrdiff_t, void *, ptrdiff_t);
};
} // namespace retinex
