**cuMM: A CUDA Memory Manager**
================================

**Developed by XeTute, a Pakistani startup**

**Overview**
-----------

cuMM is a lightweight, production-ready CUDA memory management library that provides an easy-to-use interface for allocating, resizing, and copying memory on NVIDIA GPUs. This library is designed to be efficient, flexible, and easy to integrate into your CUDA-based projects.

**Features**
------------

* **Memory Allocation**: Allocate memory on the GPU with `malloc` and `resize` methods.
* **Memory Copy**: Copy data between host and device with `copy` method.
* **Memory Management**: Automatically free memory with the destructor or manually with `free` method.
* **Template-based**: Works with any data type, including custom structs and classes.

**Example Usage**
-----------------

The following example demonstrates how to use cuMM to allocate memory, resize, and copy data between host and device:
```cpp
#include "cuMM.cu"
#include "aos.hpp" // Include Array On Steroids (AOS) library

int main() {
    constexpr size_t elems = 32 * 1000;
    constexpr size_t elemsResized = 64 * 1000;

    AOS<short> number(elemsResized); // Init with 16k numbers
    cuMM<short> gpuManager;

    // Malloc
    gpuManager.malloc(elems); // Init with storage for 32k numbers

    // Resize
    gpuManager.resize(elemsResized); // Resize to storage for 64k numbers

    // Copy
    gpuManager.copy(number.data, number.size() * sizeof(short), 1); // Copy to GPU

    gpuManager.free(); // Free memory (optional)

    return 0;
}
```
**Building and Running**
-------------------------

To build and run the demo, follow these steps:

1. Clone the repository and navigate to the project directory.
2. Compile the demo using `nvcc` (e.g., `nvcc -o demo main.cpp aos.hpp cuMM.cu`).
3. Run the demo executable (e.g., `chmod +x ./demo && ./demo`).

**Note**
-----

cuMM is designed to be used in production environments. Although it is stable and efficient, it is recommended to thoroughly test the library in your specific use case.

**Importing AOS**
-----------------

The demo uses the Array On Steroids (AOS) library, which is developed by XeTute. You can use any data type or library you prefer. To use AOS, include the `aos.hpp` header file or link against the AOS.[https://github.com/N0CTRON/array-on-steroids].

**License**
---------

cuMM is licensed under the Apache 2.0 license.
