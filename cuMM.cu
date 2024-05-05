#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thread>

template <typename devType>
class cuMM
{
public:
	devType* data = nullptr;
	size_t currentSizeBytes = NULL;

	void malloc(size_t elems)
	{
		currentSizeBytes = sizeof(devType) * elems;
		cudaMalloc(&data, currentSizeBytes);
	}

	void free()
	{
		if (data != nullptr) //<-- Branching neccessary :(
		{
			cudaFree(data);
			data = nullptr;
		}
	}

	void resize(size_t newSize)
	{
		devType* newPtr;
		cudaMalloc(&newPtr, newSize);
		if (newPtr != nullptr)
		{
			cudaMemcpy(newPtr, data, currentSizeBytes < newSize ? currentSizeBytes : newSize, cudaMemcpyHostToHost);
			cudaFree(data);
		}
		data = newPtr;
	};

	cuMM() {};

	cuMM(size_t elems) //same as cuMalloc cuz overhead 'n stuff
	{
		currentSizeBytes = sizeof(devType) * elems;
		cudaMalloc(&data, currentSizeBytes);
	};

	~cuMM() { std::thread(free); }
};
