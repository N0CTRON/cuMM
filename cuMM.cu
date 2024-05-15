#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thread>

template <typename devType>
class cuMM
{
public:
	devType* data = nullptr;
	std::uint64_t currentSizeBytes = NULL;

	void malloc(std::uint64_t elems)
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

	void resize(std::uint64_t newElems)
	{
		if (newElems == 0)
		{
			free();
			return;
		}

		devType* newPtr = nullptr;
		std::uint64_t newSize = sizeof(devType) * newElems;
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

	template <typename devArray>
	void copy(devArray toCopy, std::uint64_t size, bool toOrFrom) //true = to / to CUDA, false = from / to host
	{
		if (toOrFrom && data != nullptr) cudaMemcpy(data, (void*)toCopy, std::min(currentSizeBytes, size), cudaMemcpyHostToDevice);
		else if (!toOrFrom && data != nullptr) cudaMemcpy((void*)toCopy, data, std::min(currentSizeBytes, size), cudaMemcpyDeviceToHost);
	}

	std::uint64_t size() { return currentSizeBytes / sizeof(devType); }
	std::uint64_t sizeBytes() { return currentSizeBytes; }

	cuMM() {};
	cuMM(std::uint64_t elems) //same as cuMalloc cuz overhead 'n stuff
	{
		currentSizeBytes = sizeof(devType) * elems;
		cudaMalloc(&data, currentSizeBytes);
	};
	~cuMM() { std::thread(free); }

	devType& operator[] (const std::uint64_t& index) { return this->data[index]; }
};
