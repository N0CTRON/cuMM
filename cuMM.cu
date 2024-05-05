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

	void resize(size_t newElems)
	{
		if (newElems == 0)
		{
			free();
			return;
		}

		devType* newPtr = nullptr;
		size_t newSize = sizeof(devType) * newElems;
		cudaMalloc(&newPtr, newSize);

		if (newPtr != nullptr)
		{
			cudaMemcpy(newPtr, data, currentSizeBytes < newSize ? currentSizeBytes : newSize, cudaMemcpyDeviceToDevice);
			cudaFree(data);
			data = newPtr;
			currentSizeBytes = newSize;
		}
		else throw "CuMM couldn't resize: \"newPtr\" is nullPtr. Keeping old memory.\n";
	}

	size_t size() { return currentSizeBytes / sizeof(devType); }
	size_t sizeBytes() { return currentSizeBytes; }

	cuMM() {};
	cuMM(size_t elems) //same as cuMalloc cuz overhead 'n stuff
	{
		currentSizeBytes = sizeof(devType) * elems;
		cudaMalloc(&data, currentSizeBytes);
	};
	~cuMM() { std::thread(free); }

	devType& operator[] (const size_t& index) { return this->data[index]; }
};
