#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <thread>

template <typename devType>
class cuMM
{
public:
	devType* data;

	void malloc(size_t elems) { cudaMalloc(&data, sizeof(devType) * elems); }
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
		return; //Next commit
	};

	cuMM() {};
	cuMM(size_t elems) { cudaMalloc(&data, sizeof(devType) * elems); }; //same as cuMalloc cuz overhead 'n stuff
	~cuMM() { std::thread(free); }
};