// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "common.hpp"
#include <memory>
#include <vector>
#include <string>
namespace rf {
constexpr size_t MiB = 1024 * 1024;
Config configure(int w, int h, int bits, bool floating, bool mask, int scale, const std::vector<double> &sigma,
                 double blur, bool quick = false);
size_t cpuEssential(const Config &c);
class Cpu {
    struct Impl;
    std::unique_ptr<Impl> impl;

  public:
    explicit Cpu(const Config &c);
    ~Cpu();
    void enableCache(size_t budget);
    size_t bytes() const;
    bool cached() const;
    void process(const void *src, ptrdiff_t ss, void *dst, ptrdiff_t ds);
};
class Gpu {
  public:
    virtual ~Gpu() = default;
    virtual void process(const void *, ptrdiff_t, void *, ptrdiff_t) = 0;
    virtual size_t deviceBytes() const = 0;
    virtual size_t hostBytes() const = 0;
};
std::unique_ptr<Gpu> createGpu(const Config &c, size_t deviceBudget, size_t reserve, size_t hostBudget);
} // namespace rf
