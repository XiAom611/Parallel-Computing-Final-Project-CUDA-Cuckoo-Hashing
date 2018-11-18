#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>



unsigned myrand() {
	unsigned a = rand()<<10;
	unsigned b = rand();
	return a + b;
}

__global__ void cuckooHash(
	unsigned* hashTable,
	unsigned* a, unsigned* b,
	unsigned* entry,
	unsigned* function,
	unsigned* collision,
	unsigned n_function, unsigned n, unsigned p)
{
	unsigned k = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned num = function[k];
	unsigned hashValue = ((a[num] * entry[k] + b[num]) % p) % n;
	if (collision[k] == 1 || hashTable[hashValue] == 0xffffffff) {
		hashTable[hashValue] = entry[k];
		function[k] = (num + 1) % n_function;
	}
}

__global__ void detectCollision(
	unsigned* hashTable,
	unsigned* a, unsigned* b,
	unsigned* entry,
	unsigned* function,
	unsigned* collision,
	unsigned n_function, unsigned n, unsigned p)
{
	unsigned k = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned num = (function[k] - 1) % n_function;
	unsigned hashValue = ((a[num] * entry[k] + b[num]) % p) % n;
	if (hashTable[hashValue] != entry[k]) {
		collision[k] = 1;
	} else {
		collision[k] = 0;
	}
}

__global__ void lookup(
	unsigned* hashTable,
	unsigned* a, unsigned* b,
	unsigned* searchEntry,
	unsigned* dict,
	unsigned n_function, unsigned n, unsigned p)
{
	unsigned k = blockDim.x * blockIdx.x + threadIdx.x;
	for (unsigned i = 0; i < n_function; i++) {
		unsigned hashValue = ((a[i] * searchEntry[k] + b[i]) % p) % n;
		if (hashTable[hashValue] == searchEntry[k]) {
			dict[k] = 1;
			break;
		}
	}
}

void generate_a_b(unsigned n_function, unsigned* a, unsigned* b) {
	for (unsigned i = 0; i < n_function; i++) {
		a[i] = rand() % 10;
		b[i] = rand() % 10;
		if (i != 0) {
			while (a[i] == a[i - 1] || b[i] == b[i - 1]) {
				a[i] = rand() % 10;
				b[i] = rand() % 10;
			}
		}
	}

	/////////////For task 5
	//a[0] = 232;
	//b[0] = 0;

	//////////DEBUG
	std::cout << "a: ";
	for (unsigned i = 0; i < n_function; i++) {
		std::cout << a[i] << " ";
	}
	std::cout << std::endl;
	std::cout << "b: ";
	for (unsigned i = 0; i < n_function; i++) {
		std::cout << b[i] << " ";
	}
	std::cout << std::endl;
}

int main() {
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;
	//std::cout << myrand() << std::endl;

	cudaError_t err = cudaSuccess;

	unsigned task, n_function;
	unsigned N;
	unsigned entryLength;
	unsigned p = 62353171; // a large prime

	unsigned *entry;
	unsigned *collision;
	unsigned *hashTable;
	unsigned *function;
	unsigned *a, *b;
	unsigned *dict;
	unsigned *searchEntry;

	unsigned *d_entry = NULL;
	unsigned *d_collision = NULL;
	unsigned *d_hashTable = NULL;
	unsigned *d_function = NULL;
	unsigned *d_a = NULL;
	unsigned *d_b = NULL;
	unsigned *d_dict = NULL;
	unsigned *d_searchEntry = NULL;

	unsigned limit;
	unsigned blockNum;
	unsigned blockSize = 512;
	unsigned flag = 0;

	unsigned iteration = 0;

	unsigned test;
	unsigned testNum;
	unsigned testHashValue;

	clock_t startTime;
	clock_t endTime;

	unsigned sum = 0;

	std::cout << "How many hash functions?" << std::endl;
	std::cin >> n_function;
	std::cout << "Which task?" << std::endl;
	std::cin >> task;



	srand(time(NULL));

	switch (task)
	{
	case 1:
		N = pow(2, 25); //33554432
		limit = ceil(4 * log10((double)N));
		unsigned s;
		std::cout << "Input s:";
		std::cin >> s;
		entryLength = pow(2, s);

		for (unsigned z = 0; z < 5; z++) {
			blockNum = ceil((double)entryLength / blockSize);
			entry = new unsigned[entryLength];

			std::cout << "Generating random numbers between 0~10000000..." << std::endl;
			for (unsigned i = 0; i < entryLength; i++) {
				entry[i] = myrand() % 10000000;
			}

			std::cout << "Generating a,b..." << std::endl;
			a = new unsigned[n_function];
			b = new unsigned[n_function];
			generate_a_b(n_function, a, b);
			//for (unsigned i = 0; i < n_function; i++) {
			//	a[i] = rand() % 10;
			//	b[i] = rand() % 10;
			//	if (i != 0) {
			//		while (a[i]==a[i-1] || b[i]==b[i-1]){
			//			a[i] = rand() % 10;
			//			b[i] = rand() % 10;
			//		}
			//	}
			//}

			std::cout << "Initilizing hashTable..." << std::endl;
			hashTable = new unsigned[N];
			memset(hashTable, 0xffffffff, N * sizeof(unsigned));

			std::cout << "Initilizing collisionTable..." << std::endl;
			collision = new unsigned[entryLength];
			memset(collision, 0, entryLength * sizeof(unsigned));

			std::cout << "Initilizing functionIndex..." << std::endl;
			function = new unsigned[entryLength];
			memset(function, 0, entryLength * sizeof(unsigned));

			std::cout << "Allocating device memory..." << std::endl;

			err = cudaMalloc((void**)&d_entry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating entry[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_a, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating a[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_b, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating b[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_collision, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating collision[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_function, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating functionIndex[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_hashTable, N * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating hashTable[] failed" << std::endl;
				goto Error;
			}

			std::cout << "Copying memory from host to device..." << std::endl;

			err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy hashTable" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy a[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy b[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy entry[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy functionIndex[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy collision[]" << std::endl;
				goto Error;
			}
			iteration = 0;
			startTime = clock();
			do {
				flag = 0;
				//Restarting hash
				if (iteration == limit) {
					iteration = 0;
					std::cout << ".........Rehash........." << std::endl;
					generate_a_b(n_function, a, b);
					memset(hashTable, 0xffffffff, N * sizeof(unsigned));
					memset(function, 0, entryLength * sizeof(unsigned));
					memset(collision, 0, entryLength * sizeof(unsigned));

					std::cout << "Recopying memory from host to device..." << std::endl;

					err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy hashTable" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy a[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy b[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy entry[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy functionIndex[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy collision[]" << std::endl;
						goto Error;
					}
				}



				iteration++;
				cuckooHash << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				detectCollision << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				std::cout << "Finish hash " << iteration << " times" << std::endl;

				// Copy collison back
				err = cudaMemcpy(collision, d_collision, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					std::cout << "Copy collison failed from device to host" << std::endl;
					std::cout << cudaGetErrorString(err) << std::endl;
					goto Error;
				}

				for (unsigned i = 0; i < entryLength; i++) {
					flag += collision[i];
				}
				std::cout << flag << " collisions" << std::endl;

			} while (flag != 0);
			endTime = clock();
			std::cout << "Hash Done!" << std::endl;

			std::cout << "time:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;

			//////////test
			//cudaMemcpy(function, d_function, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//cudaMemcpy(hashTable, d_hashTable, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//std::cout << "Checking Hash..." << std::endl;
			//for (unsigned i = 0; i < 10; i++) {
			//	test = myrand() % entryLength;
			//	testNum = (function[test] - 1) % n_function;
			//	testHashValue = ((a[testNum] * entry[test] + b[testNum]) % p) % N;
			//	std::cout << "--" << entry[test] << " " << testNum << " ";
			//	if (hashTable[testHashValue] == entry[test]) {
			//		std::cout << "correct" << std::endl;
			//	} else {
			//		std::cout << "incorrect" << std::endl;
			//	}
			//}
		}
		break;

	case 2:
		N = pow(2, 25); //33554432
		limit = ceil(4 * log10((double)N));
		unsigned ii;
		std::cout << "Input i:";
		std::cin >> ii;
		entryLength = pow(2, 24);

		for (unsigned z = 0; z < 5; z++) {
			blockNum = ceil((double)entryLength / blockSize);
			entry = new unsigned[entryLength];

			std::cout << "Generating random numbers between 0~10000000..." << std::endl;
			for (unsigned i = 0; i < entryLength; i++) {
				entry[i] = myrand() % 10000000;
			}

			std::cout << "Generating a,b..." << std::endl;
			a = new unsigned[n_function];
			b = new unsigned[n_function];
			generate_a_b(n_function, a, b);
			//for (unsigned i = 0; i < n_function; i++) {
			//	a[i] = rand() % 10;
			//	b[i] = rand() % 10;
			//	if (i != 0) {
			//		while (a[i]==a[i-1] || b[i]==b[i-1]){
			//			a[i] = rand() % 10;
			//			b[i] = rand() % 10;
			//		}
			//	}
			//}

			std::cout << "Initilizing hashTable..." << std::endl;
			hashTable = new unsigned[N];
			memset(hashTable, 0xffffffff, N * sizeof(unsigned));

			std::cout << "Initilizing collisionTable..." << std::endl;
			collision = new unsigned[entryLength];
			memset(collision, 0, entryLength * sizeof(unsigned));

			std::cout << "Initilizing functionIndex..." << std::endl;
			function = new unsigned[entryLength];
			memset(function, 0, entryLength * sizeof(unsigned));

			std::cout << "Allocating device memory..." << std::endl;

			err = cudaMalloc((void**)&d_entry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating entry[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_a, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating a[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_b, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating b[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_collision, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating collision[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_function, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating functionIndex[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_hashTable, N * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating hashTable[] failed" << std::endl;
				goto Error;
			}

			std::cout << "Copying memory from host to device..." << std::endl;

			err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy hashTable" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy a[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy b[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy entry[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy functionIndex[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy collision[]" << std::endl;
				goto Error;
			}
			iteration = 0;
			do {
				flag = 0;
				//Restarting hash
				if (iteration == limit) {
					iteration = 0;
					std::cout << ".........Rehash........." << std::endl;
					generate_a_b(n_function, a, b);
					memset(hashTable, 0xffffffff, N * sizeof(unsigned));
					memset(function, 0, entryLength * sizeof(unsigned));
					memset(collision, 0, entryLength * sizeof(unsigned));

					std::cout << "Recopying memory from host to device..." << std::endl;

					err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy hashTable" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy a[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy b[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy entry[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy functionIndex[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy collision[]" << std::endl;
						goto Error;
					}
				}

				iteration++;
				cuckooHash <<< blockNum, blockSize >>> (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				detectCollision <<< blockNum, blockSize >>> (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				std::cout << "Finish hash " << iteration << " times" << std::endl;

				// Copy collison back
				err = cudaMemcpy(collision, d_collision, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					std::cout << "Copy collison failed from device to host" << std::endl;
					goto Error;
				}

				for (unsigned i = 0; i < entryLength; i++) {
					flag += collision[i];
				}
				std::cout << flag << " collisions" << std::endl;

			} while (flag != 0);
			std::cout << "Hash Done!" << std::endl;

			//////////////////////////////////
			///////// Look up part
			std::cout << "Initilizing lookup part..." << std::endl;

			cudaMemcpy(hashTable, d_hashTable, N * sizeof(unsigned), cudaMemcpyDeviceToHost);

			searchEntry = new unsigned[entryLength];
			dict = new unsigned[entryLength];

			//add random key in to search entry
			for (unsigned i = 0; i < entryLength; i++) {
				//std::cout << myrand() << std::endl;
				if (i < entryLength*(1 - ii*0.1)) {
					unsigned randIdx = myrand() % entryLength;
					//std::cout << randIdx << " ";
					searchEntry[i] = entry[randIdx];
				} else {
					searchEntry[i] = myrand();
				}
			}
			//std::cout << std::endl;

			//store if find
			memset(dict, 0, entryLength * sizeof(unsigned));

			//allocate cuda memory for search
			err = cudaMalloc((void**)&d_searchEntry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "allocate searchEntry fail" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_dict, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "allocate dict fail" << std::endl;
				goto Error;
			}

			//copy data from host to device
			err = cudaMemcpy(d_searchEntry, searchEntry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "Fail to copy search Entry to device" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_dict, dict, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "Fail to copy dict to device" << std::endl;
				goto Error;
			}

			std::cout << "Start lookup keys..." << std::endl;

			///////////////DEBUG
			//sum = 0;
			//for (unsigned i = 0; i < entryLength; i++) {
			//	for (unsigned j = 0; j < n_function; j++) {
			//		unsigned hashValue = ((a[j] * searchEntry[i] + b[j]) % p) % N;
			//		if (hashTable[hashValue] == searchEntry[i]) {
			//			sum++;
			//			break;
			//		}
			//	}
			//}
			//std::cout << "--------------" << sum/entryLength << std::endl;

			startTime = clock();
			lookup <<< blockNum, blockSize >>> (d_hashTable,
				d_a, d_b,
				d_searchEntry,
				d_dict,
				n_function, N, p);
			endTime = clock();
			std::cout << "Lookup done" << std::endl;

			//Copy dict back to host
			err = cudaMemcpy(dict, d_dict, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) {
				std::cout << "Fail to copy dict back to device" << std::endl;
				goto Error;
			}

			sum = 0;
			for (unsigned i = 0; i < entryLength; i++) {
				sum += dict[i];
			}

			std::cout << "Hash Hit: " << (double)sum / (double)entryLength * 100 << "%";
			std::cout << " with " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;
		}
		break;

	case 3:
		//N = pow(2, 25); //33554432
		float alpha;
		std::cout << "input alpha: ";
		std::cin >> alpha;

		entryLength = pow(2, 24);
		std::cout << entryLength << " ";

		N = alpha*entryLength;
		std::cout << N << " ";
		limit = ceil(4 * log10((double)N));
		std::cout << limit << std::endl;


		for (unsigned z = 0; z < 5; z++) {
			blockNum = ceil((double)entryLength / blockSize);
			entry = new unsigned[entryLength];

			std::cout << "Generating random numbers between 0~10000000..." << std::endl;
			for (unsigned i = 0; i < entryLength; i++) {
				entry[i] = myrand() % 10000000;
			}

			std::cout << "Generating a,b..." << std::endl;
			a = new unsigned[n_function];
			b = new unsigned[n_function];
			generate_a_b(n_function, a, b);
			//for (unsigned i = 0; i < n_function; i++) {
			//	a[i] = rand() % 10;
			//	b[i] = rand() % 10;
			//	if (i != 0) {
			//		while (a[i]==a[i-1] || b[i]==b[i-1]){
			//			a[i] = rand() % 10;
			//			b[i] = rand() % 10;
			//		}
			//	}
			//}

			std::cout << "Initilizing hashTable..." << std::endl;
			hashTable = new unsigned[N];
			memset(hashTable, 0xffffffff, N * sizeof(unsigned));

			std::cout << "Initilizing collisionTable..." << std::endl;
			collision = new unsigned[entryLength];
			memset(collision, 0, entryLength * sizeof(unsigned));

			std::cout << "Initilizing functionIndex..." << std::endl;
			function = new unsigned[entryLength];
			memset(function, 0, entryLength * sizeof(unsigned));

			std::cout << "Allocating device memory..." << std::endl;

			err = cudaMalloc((void**)&d_entry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating entry[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_a, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating a[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_b, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating b[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_collision, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating collision[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_function, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating functionIndex[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_hashTable, N * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating hashTable[] failed" << std::endl;
				goto Error;
			}

			std::cout << "Copying memory from host to device..." << std::endl;

			err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy hashTable" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy a[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy b[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy entry[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy functionIndex[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy collision[]" << std::endl;
				goto Error;
			}

			iteration = 0;
			startTime = clock();
			do {
				flag = 0;
				//Restarting hash
				if (iteration == limit) {
					iteration = 0;
					std::cout << ".........Rehash........." << std::endl;
					generate_a_b(n_function, a, b);
					memset(hashTable, 0xffffffff, N * sizeof(unsigned));
					memset(function, 0, entryLength * sizeof(unsigned));
					memset(collision, 0, entryLength * sizeof(unsigned));

					std::cout << "Recopying memory from host to device..." << std::endl;

					err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy hashTable" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy a[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy b[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy entry[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy functionIndex[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy collision[]" << std::endl;
						goto Error;
					}
				}



				iteration++;
				cuckooHash << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				detectCollision << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				std::cout << "Finish hash " << iteration << " times" << std::endl;

				// Copy collison back
				err = cudaMemcpy(collision, d_collision, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					std::cout << "Copy collison failed from device to host" << std::endl;
					std::cout << cudaGetErrorString(err) << std::endl;
					goto Error;
				}

				for (unsigned i = 0; i < entryLength; i++) {
					flag += collision[i];
				}
				std::cout << flag << " collisions" << std::endl;

			} while (flag != 0);
			endTime = clock();
			std::cout << "Hash Done!" << std::endl;

			std::cout << "time:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;

			////////////test
			//cudaMemcpy(function, d_function, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//cudaMemcpy(hashTable, d_hashTable, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//std::cout << "Checking Hash..." << std::endl;
			//for (unsigned i = 0; i < 10; i++) {
			//	test = myrand() % entryLength;
			//	testNum = (function[test] - 1) % n_function;
			//	testHashValue = ((a[testNum] * entry[test] + b[testNum]) % p) % N;
			//	std::cout << "--" << entry[test] << " " << testNum << " ";
			//	if (hashTable[testHashValue] == entry[test]) {
			//		std::cout << "correct" << std::endl;
			//	} else {
			//		std::cout << "incorrect" << std::endl;
			//	}
			//}
		}
		break;

	case 4:
		float beta;
		std::cout << "input bound coefficient: ";
		std::cin >> beta;

		entryLength = pow(2, 24);
		std::cout << entryLength << " ";

		N = 1.2*entryLength;
		std::cout << N << " ";
		limit = ceil(beta * log10((double)N));
		std::cout << limit << std::endl;


		for (unsigned z = 0; z < 5; z++) {
			blockNum = ceil((double)entryLength / blockSize);
			entry = new unsigned[entryLength];

			std::cout << "Generating random numbers between 0~10000000..." << std::endl;
			for (unsigned i = 0; i < entryLength; i++) {
				entry[i] = myrand() % 10000000;
			}

			std::cout << "Generating a,b..." << std::endl;
			a = new unsigned[n_function];
			b = new unsigned[n_function];
			generate_a_b(n_function, a, b);
			//for (unsigned i = 0; i < n_function; i++) {
			//	a[i] = rand() % 10;
			//	b[i] = rand() % 10;
			//	if (i != 0) {
			//		while (a[i]==a[i-1] || b[i]==b[i-1]){
			//			a[i] = rand() % 10;
			//			b[i] = rand() % 10;
			//		}
			//	}
			//}

			std::cout << "Initilizing hashTable..." << std::endl;
			hashTable = new unsigned[N];
			memset(hashTable, 0xffffffff, N * sizeof(unsigned));

			std::cout << "Initilizing collisionTable..." << std::endl;
			collision = new unsigned[entryLength];
			memset(collision, 0, entryLength * sizeof(unsigned));

			std::cout << "Initilizing functionIndex..." << std::endl;
			function = new unsigned[entryLength];
			memset(function, 0, entryLength * sizeof(unsigned));

			std::cout << "Allocating device memory..." << std::endl;

			err = cudaMalloc((void**)&d_entry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating entry[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_a, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating a[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_b, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating b[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_collision, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating collision[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_function, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating functionIndex[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_hashTable, N * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating hashTable[] failed" << std::endl;
				goto Error;
			}

			std::cout << "Copying memory from host to device..." << std::endl;

			err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy hashTable" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy a[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy b[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy entry[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy functionIndex[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy collision[]" << std::endl;
				goto Error;
			}

			iteration = 0;
			startTime = clock();
			do {
				flag = 0;
				//Restarting hash
				if (iteration == limit) {
					iteration = 0;
					std::cout << ".........Rehash........." << std::endl;
					generate_a_b(n_function, a, b);
					memset(hashTable, 0xffffffff, N * sizeof(unsigned));
					memset(function, 0, entryLength * sizeof(unsigned));
					memset(collision, 0, entryLength * sizeof(unsigned));

					std::cout << "Recopying memory from host to device..." << std::endl;

					err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy hashTable" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy a[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy b[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy entry[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy functionIndex[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy collision[]" << std::endl;
						goto Error;
					}
				}



				iteration++;
				cuckooHash << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				detectCollision << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				std::cout << "Finish hash " << iteration << " times" << std::endl;

				// Copy collison back
				err = cudaMemcpy(collision, d_collision, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					std::cout << "Copy collison failed from device to host" << std::endl;
					std::cout << cudaGetErrorString(err) << std::endl;
					goto Error;
				}

				for (unsigned i = 0; i < entryLength; i++) {
					flag += collision[i];
				}
				std::cout << flag << " collisions" << std::endl;

			} while (flag != 0);
			endTime = clock();
			std::cout << "Hash Done!" << std::endl;

			std::cout << "time:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;

			//////////test
			//cudaMemcpy(function, d_function, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//cudaMemcpy(hashTable, d_hashTable, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//std::cout << "Checking Hash..." << std::endl;
			//for (unsigned i = 0; i < 10; i++) {
			//	test = myrand() % entryLength;
			//	testNum = (function[test] - 1) % n_function;
			//	testHashValue = ((a[testNum] * entry[test] + b[testNum]) % p) % N;
			//	std::cout << "--" << entry[test] << " " << testNum << " ";
			//	if (hashTable[testHashValue] == entry[test]) {
			//		std::cout << "correct" << std::endl;
			//	} else {
			//		std::cout << "incorrect" << std::endl;
			//	}
			//}
		}
		break;
	case 5:
		entryLength = pow(2, 24);
		std::cout << entryLength << " ";

		N = 1.2*entryLength;
		std::cout << N << " ";
		limit = ceil(6 * log10((double)N));
		std::cout << limit << std::endl;


		for (unsigned z = 0; z < 5; z++) {
			blockNum = ceil((double)entryLength / blockSize);
			entry = new unsigned[entryLength];

			std::cout << "Generating random numbers between 0~10000000..." << std::endl;
			for (unsigned i = 0; i < entryLength; i++) {
				entry[i] = myrand() % 10000000;
			}

			std::cout << "Generating a,b..." << std::endl;
			a = new unsigned[n_function];
			b = new unsigned[n_function];
			generate_a_b(n_function, a, b);
			//for (unsigned i = 0; i < n_function; i++) {
			//	a[i] = rand() % 10;
			//	b[i] = rand() % 10;
			//	if (i != 0) {
			//		while (a[i]==a[i-1] || b[i]==b[i-1]){
			//			a[i] = rand() % 10;
			//			b[i] = rand() % 10;
			//		}
			//	}
			//}

			std::cout << "Initilizing hashTable..." << std::endl;
			hashTable = new unsigned[N];
			memset(hashTable, 0xffffffff, N * sizeof(unsigned));

			std::cout << "Initilizing collisionTable..." << std::endl;
			collision = new unsigned[entryLength];
			memset(collision, 0, entryLength * sizeof(unsigned));

			std::cout << "Initilizing functionIndex..." << std::endl;
			function = new unsigned[entryLength];
			memset(function, 0, entryLength * sizeof(unsigned));

			std::cout << "Allocating device memory..." << std::endl;

			err = cudaMalloc((void**)&d_entry, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating entry[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_a, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating a[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_b, n_function * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating b[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_collision, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating collision[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_function, entryLength * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating functionIndex[] failed" << std::endl;
				goto Error;
			}

			err = cudaMalloc((void**)&d_hashTable, N * sizeof(unsigned));
			if (err != cudaSuccess) {
				std::cout << "-->Allocating hashTable[] failed" << std::endl;
				goto Error;
			}

			std::cout << "Copying memory from host to device..." << std::endl;

			err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy hashTable" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy a[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy b[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy entry[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy functionIndex[]" << std::endl;
				goto Error;
			}

			err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
			if (err != cudaSuccess) {
				std::cout << "-->Fail to copy collision[]" << std::endl;
				goto Error;
			}

			iteration = 0;
			startTime = clock();
			do {
				flag = 0;
				//Restarting hash
				if (iteration == limit) {
					iteration = 0;
					std::cout << ".........Rehash........." << std::endl;
					generate_a_b(n_function, a, b);
					memset(hashTable, 0xffffffff, N * sizeof(unsigned));
					memset(function, 0, entryLength * sizeof(unsigned));
					memset(collision, 0, entryLength * sizeof(unsigned));

					std::cout << "Recopying memory from host to device..." << std::endl;

					err = cudaMemcpy(d_hashTable, hashTable, N * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy hashTable" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_a, a, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy a[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_b, b, n_function * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy b[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_entry, entry, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy entry[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_function, function, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy functionIndex[]" << std::endl;
						goto Error;
					}

					err = cudaMemcpy(d_collision, collision, entryLength * sizeof(unsigned), cudaMemcpyHostToDevice);
					if (err != cudaSuccess) {
						std::cout << "-->Fail to copy collision[]" << std::endl;
						goto Error;
					}
				}



				iteration++;
				cuckooHash << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				detectCollision << < blockNum, blockSize >> > (d_hashTable,
					d_a, d_b,
					d_entry,
					d_function,
					d_collision,
					n_function, N, p);

				std::cout << "Finish hash " << iteration << " times" << std::endl;

				// Copy collison back
				err = cudaMemcpy(collision, d_collision, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
				if (err != cudaSuccess) {
					std::cout << "Copy collison failed from device to host" << std::endl;
					std::cout << cudaGetErrorString(err) << std::endl;
					goto Error;
				}

				for (unsigned i = 0; i < entryLength; i++) {
					flag += collision[i];
				}
				std::cout << flag << " collisions" << std::endl;

			} while (flag != 0);
			endTime = clock();
			std::cout << "Hash Done!" << std::endl;

			std::cout << "time:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;

			//////////test
			//cudaMemcpy(function, d_function, entryLength * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//cudaMemcpy(hashTable, d_hashTable, N * sizeof(unsigned), cudaMemcpyDeviceToHost);
			//std::cout << "Checking Hash..." << std::endl;
			//for (unsigned i = 0; i < 10; i++) {
			//	test = myrand() % entryLength;
			//	testNum = (function[test] - 1) % n_function;
			//	testHashValue = ((a[testNum] * entry[test] + b[testNum]) % p) % N;
			//	std::cout << "--" << entry[test] << " " << testNum << " ";
			//	if (hashTable[testHashValue] == entry[test]) {
			//		std::cout << "correct" << std::endl;
			//	} else {
			//		std::cout << "incorrect" << std::endl;
			//	}
			//}
		}
		break;

	default:
		std::cout << "No such task. Please run again." << std::endl;
		goto Error;
		break;
	}



Error:

	cudaFree((void**)&d_a);
	cudaFree((void**)&d_b);
	cudaFree((void**)&d_collision);
	cudaFree((void**)&d_dict);
	cudaFree((void**)&d_entry);
	cudaFree((void**)&d_function);
	cudaFree((void**)&d_searchEntry);
	cudaFree((void**)&d_hashTable);
	system("pause");
	return 0;
}
