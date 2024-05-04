#include <iostream>
#include <chrono>

#include "cuMM.cu"
#include "aos.hpp" //<-- In this demo, we'll use the Array On Steroids library, but you can use any datatype you want

using namespace std::literals::chrono_literals;

typedef std::chrono::high_resolution_clock::time_point timePoint;

int main()
{
	constexpr size_t elems = 32 * 500;

	AOS<short> number(elems); //Init with 32k numbers
	
	timePoint start = std::chrono::high_resolution_clock::now();
	cuMM<short> gpuManager(elems); //Init with 32k numbers
	timePoint end = std::chrono::high_resolution_clock::now();
	std::cout << "cuMM took " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms to allocate " << elems * sizeof(short) << " bytes.\n";

	//manager.free(); <-- we don't need this cuz we have "~cuMM" :), you can still do this if you wanna be 100% save[would recommend on bigger projects].

	return 0;
}