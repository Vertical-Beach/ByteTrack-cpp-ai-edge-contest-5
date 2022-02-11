#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <iostream>

namespace byte_track
{
#ifndef RISCV
int lapjv_internal(const size_t n, double *cost[], int *x, int *y);
#else
int lapjv_internal(const size_t n, double *cost[], int *x, int *y, volatile int* riscv_dmem_base);
#endif
}