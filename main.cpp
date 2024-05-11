#include <iostream>
#include <chrono>

#include "cuMM.cu"
#include "aos.hpp" //<-- In this demo, we'll use the Array On Steroids library, but you can use any datatype you want

// This is a demo of the "cuMM" library, devoloped by XeTute. 
// It uses the "AOS" header-only libraries, which are also dev. by XeTute.
// XeTutes website: "https://xetute.neocities.org/"
// AOS GitHub: "https://www.github.com/N0CTRON/array-on-steriods/"

using namespace std::literals::chrono_literals;

typedef std::chrono::high_resolution_clock::time_point timePoint;
typedef short typeToAlloc;

int main()
{
	constexpr size_t elems = 32 * 1000;
	constexpr size_t elemsResized = 64 * 1000;

	AOS<typeToAlloc> number(elemsResized); //Init with 16k numbers
	cuMM<typeToAlloc> gpuManager;

	timePoint start;
	timePoint end;
	
	//Malloc
	start = std::chrono::high_resolution_clock::now();
	gpuManager.malloc(elems); //Init with storage for 32k numbers
	end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to allocate " << gpuManager.sizeBytes() << " bytes.\n";

	//Resize
	start = std::chrono::high_resolution_clock::now();
	gpuManager.resize(elemsResized); //Resize to storage for 64k numbers
	end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to resize to " << gpuManager.sizeBytes() << " bytes.\n";

	//Copy
	start = std::chrono::high_resolution_clock::now();
	gpuManager.copy(number.data, number.size() * sizeof(typeToAlloc), 1); //Init with 32k numbers and copy to GPU
	end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to copy to GPU.\n";

	gpuManager.free(); //<-- we don't necessarily need this cuz we have "~cuMM" :), you can still do this if you wanna be 100% save[would recommend on bigger projects].

	return 0;
}
