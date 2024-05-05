**cuMM: CUDA Memory Manager**

**Work in Progress**

cuMM is a C++ class designed to efficiently manage memory on CUDA devices. It provides a simple and efficient way to allocate, resize, and deallocate memory on the device.

**Features**

* `malloc` and `free` methods for allocating and deallocating memory on the device
* `resize` method for dynamically resizing allocated memory
* `size` and `sizeBytes` methods for querying allocated memory size

**Example Usage**

The provided `main.cpp` file demonstrates the usage of the cuMM class. It showcases the allocation, resizing, and deallocation of memory on the CUDA device.

**Building and Running**

To build and run the example, ensure you have a CUDA-compatible GPU and a C++ compiler installed on your system. Compile the code using `nvcc` and run the resulting executable.

**Note**

cuMM is still in production and may undergo changes. Your feedback and contributions are welcome.

**Importing AOS**

This project also includes the AOS (Array On Steroids) library, which is imported from the same company as cuMM. AOS is a template class that provides a flexible and efficient way to manage arrays on the host system.
