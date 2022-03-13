#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

// This is the public interface of our cuda function, called directly in main.cpp
void wrap_test_vectorAdd();