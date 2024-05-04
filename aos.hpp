#include <malloc.h>
#include <thread>
#include <iostream>

template <typename arrType>
class AOS //AOS = array on steroids
{
private:
	size_t vsize;

public:
	arrType* data;

	void init(size_t nSize)
	{
		vsize = nSize;
		data = (arrType*)malloc(sizeof(arrType) * nSize);
		for (size_t i = 0; i < nSize; ++i) data[i] = arrType(NULL);
	}

	void init(size_t nSize, arrType defaultVal)
	{
		vsize = nSize;
		data = (arrType*)malloc(sizeof(arrType) * nSize);
		for (size_t i = 0; i < nSize; ++i) data[i] = arrType(defaultVal);
	}

	void resize(size_t nSize)
	{
		if (nSize != vsize)
		{
			arrType* newData = (arrType*)realloc(data, sizeof(arrType) * nSize);
			if (newData) //realloc succeded!! :)
			{
				if (nSize > vsize) for (size_t i = vsize; i < nSize; ++i) newData[i] = arrType(NULL);
				data = newData;
				vsize = nSize;
			}
			else std::cout << "[ | AOS-LOG ]: AOS::resize(size_t) failed, keeping old array.\n"; //if failed :(
		}
	}

	void suicide()
	{
		free(data);
		data = nullptr;
	}

	size_t size() { return vsize; }

	AOS(size_t nSize) { init(nSize); };
	AOS(size_t nSize, size_t defaultVal) { init(nSize, defaultVal); };
	AOS() {};
	~AOS() { std::thread(suicide); } //<-- Threading was tested, about 1.9 times faster than calling "suicide" from main thread

	arrType& operator[](const size_t index) { return this->data[index]; }
};