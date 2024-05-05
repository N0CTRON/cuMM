#include <iostream>
#include <chrono>

#include "cuMM.cu"
#include "aos.hpp" //<-- In this demo, we'll use the Array On Steroids library, but you can use any datatype you want

using namespace std::literals::chrono_literals;

typedef std::chrono::high_resolution_clock::time_point timePoint;
typedef short typeToAlloc;

int main()
{
	constexpr size_t elems = 32 * 1000;
	constexpr size_t elemsResized = 64 * 1000;

	AOS<typeToAlloc> number(elems); //Init with 16k numbers
	cuMM<typeToAlloc> gpuManager;

	timePoint start;
	timePoint end;
	
	//Malloc
	start = std::chrono::high_resolution_clock::now();
	gpuManager.malloc(elems); //Init with 32k numbers
	end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to allocate " << gpuManager.sizeBytes() << " bytes.\n";

	//Resize
	start = std::chrono::high_resolution_clock::now();
	gpuManager.resize(elemsResized); //Init with 32k numbers
	end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to resize to " << gpuManager.sizeBytes() << " bytes.\n";

	//manager.free(); <-- we don't need this cuz we have "~cuMM" :), you can still do this if you wanna be 100% save[would recommend on bigger projects].

	return 0;
}
