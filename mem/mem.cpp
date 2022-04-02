#include <iostream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>
using namespace std;

void rand_fill(uint8_t* a, const uint64_t len) {
    // #pragma omp parallel for
    const int block_size = 1024;
    for (uint64_t i = 0; i < len / block_size; i++) {
        // a[i] = rand();
        memset(a + block_size * i, rand(), block_size);
    }
}

const int warmup_size = (1 << 20) * 100; // 100 MB

void int8_index_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    for (uint64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

void int8_pt_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    for (uint64_t i = 0; i < len; i++) {
        *dst = *src;
        dst++;
        src++;
    }
}

void int64_index_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    uint64_t *u64_dst = (uint64_t*)dst;
    uint64_t *u64_src = (uint64_t*)src;
    const uint64_t n_u64 = len / sizeof(uint64_t);
    for (uint64_t i = 0; i < n_u64; i++) {
        u64_dst[i] = u64_src[i];
    }
}

void int64_pt_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    uint64_t *u64_dst = (uint64_t*)dst;
    uint64_t *u64_src = (uint64_t*)src;
    const uint64_t n_u64 = len / sizeof(uint64_t);
    for (uint64_t i = 0; i < n_u64; i++) {
        *u64_dst = *u64_src;
        u64_dst++;
        u64_src++;
    }
}

void unroll_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    uint64_t *u64_dst = (uint64_t*)dst;
    uint64_t *u64_src = (uint64_t*)src;
    const uint64_t n_u64 = len / sizeof(uint64_t);
    const uint64_t n_unroll = 16;
    for (uint64_t i = 0; i < n_u64 / n_unroll; i++) {
        *u64_dst = *u64_src;
        *(u64_dst + 1) = *(u64_src + 1);
        *(u64_dst + 2) = *(u64_src + 2);
        *(u64_dst + 3) = *(u64_src + 3);
        *(u64_dst + 4) = *(u64_src + 4);
        *(u64_dst + 5) = *(u64_src + 5);
        *(u64_dst + 6) = *(u64_src + 6);
        *(u64_dst + 7) = *(u64_src + 7);
        *(u64_dst + 8) = *(u64_src + 8);
        *(u64_dst + 9) = *(u64_src + 9);
        *(u64_dst + 10) = *(u64_src + 10);
        *(u64_dst + 11) = *(u64_src + 11);
        *(u64_dst + 12) = *(u64_src + 12);
        *(u64_dst + 13) = *(u64_src + 13);
        *(u64_dst + 14) = *(u64_src + 14);
        *(u64_dst + 15) = *(u64_src + 15);
        u64_dst += n_unroll;
        u64_src += n_unroll;
    }
}

void para_unroll_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    const uint64_t n_u64 = len / sizeof(uint64_t);
    const uint64_t n_unroll = 16;
    #pragma omp parallel for num_threads(4)
    for (uint64_t i = 0; i < n_u64 / n_unroll; i++) {
        uint64_t *u64_dst = (uint64_t*)dst + i * n_unroll;
        uint64_t *u64_src = (uint64_t*)src + i * n_unroll;
        *u64_dst = *u64_src;
        *(u64_dst + 1) = *(u64_src + 1);
        *(u64_dst + 2) = *(u64_src + 2);
        *(u64_dst + 3) = *(u64_src + 3);
        *(u64_dst + 4) = *(u64_src + 4);
        *(u64_dst + 5) = *(u64_src + 5);
        *(u64_dst + 6) = *(u64_src + 6);
        *(u64_dst + 7) = *(u64_src + 7);
        *(u64_dst + 8) = *(u64_src + 8);
        *(u64_dst + 9) = *(u64_src + 9);
        *(u64_dst + 10) = *(u64_src + 10);
        *(u64_dst + 11) = *(u64_src + 11);
        *(u64_dst + 12) = *(u64_src + 12);
        *(u64_dst + 13) = *(u64_src + 13);
        *(u64_dst + 14) = *(u64_src + 14);
        *(u64_dst + 15) = *(u64_src + 15);
    }
}

void para_int64_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    uint64_t *u64_dst = (uint64_t*)dst;
    uint64_t *u64_src = (uint64_t*)src;
    const uint64_t n_u64 = len / sizeof(uint64_t);
    #pragma omp parallel for num_threads(4)
    for (uint64_t i = 0; i < n_u64; i++) {
        u64_dst[i] = u64_src[i];
    }
}

void para_int8_copy(uint8_t* src, uint8_t* dst, const uint64_t len) {
    #pragma omp parallel for num_threads(28)
    for (uint64_t i = 0; i < len; i++) {
        dst[i] = src[i];
    }
}

void copy_data_256(uint8_t* src, uint8_t* dst, uint64_t size)
{
	assert(size % 32 == 0);
	while (size) {
		_mm256_store_si256 ((__m256i*)dst, _mm256_load_si256((__m256i*)src));
		src += 32;
		dst += 32;
		size -= 32;
	}
}

// void * memcpy_512bit_u(void *dest, const void *src, uint64_t len)
// {
//   len /= (512 / 8);
//   const __m512i_u* s = (__m512i_u*)src;
//   __m512i_u* d = (__m512i_u*)dest;

//   while (len--)
//   {
//     _mm512_storeu_si512(d++, _mm512_loadu_si512(s++));
//   }

//   return dest;
// }

void * memcpy_256bit_u(void *dest, const void *src, uint64_t len)
{
  len /= (256 / 8);
  const __m256i_u* s = (__m256i_u*)src;
  __m256i_u* d = (__m256i_u*)dest;

  while (len--)
  {
    _mm256_storeu_si256(d++, _mm256_loadu_si256(s++));
  }

  return dest;
}

double shared_mem_test(
    const uint64_t n_MB,
    uint8_t* src, uint8_t* dst,
    uint8_t* warmup_src, uint8_t* warmup_dst
) {
    const uint64_t n_bytes = n_MB * 1024 * 1024;
    cout << "---- Arr size in MB: " << n_MB << endl;
    
    cout << "  Rand filling... " << n_bytes << endl;
    rand_fill(warmup_src, warmup_size);
    cout << "  Rand filling2... " << n_bytes << endl;    
    rand_fill(src, n_bytes);

    cout << "  Warming up..." << endl;
    memcpy(warmup_dst, warmup_src, warmup_size);

    cout << "  Go go go!" << endl;

    auto start_t = chrono::high_resolution_clock::now();
    
    // memcpy(dst, src, n_bytes);
    // int8_index_copy(src, dst, n_bytes);
    // int8_pt_copy(src, dst, n_bytes);
    // int64_index_copy(src, dst, n_bytes);
    // int64_pt_copy(src, dst, n_bytes);
    // para_unroll_copy(src, dst, n_bytes);
    // para_int8_copy(src, dst, n_bytes);
    memcpy_256bit_u(src, dst, n_bytes);
    
    auto end_t = chrono::high_resolution_clock::now();

    for (int i = 0; i < 100; i++) {
        const uint64_t test_loc = rand() % n_bytes;
        assert(dst[test_loc] == src[test_loc]);
    }

    auto duration = chrono::duration_cast<chrono::microseconds>(end_t - start_t).count();
    double MBps = static_cast<double>(n_MB) / duration * 1000 * 1000;
    cout << "---- Time: " << duration << " us  Rate: " << MBps << " MB/s" << endl;

    return MBps;
}

int main(int argc, char* argv[]) {
    srand(time(nullptr));

    #pragma omp parallel
    {
        #pragma omp master
        {
            cout << "omp default thread num: " << omp_get_num_threads() << endl;
        }
    }

    vector<uint64_t> n_MB;
    for (uint64_t i = 128; i <= 1024 * 32; i <<= 1) {
        n_MB.emplace_back(i);
    }
    
    uint8_t* warmup_src = new uint8_t[warmup_size];
    uint8_t* warmup_dst = new uint8_t[warmup_size];

    const uint64_t max_size = n_MB.back() * 1024 * 1024;
    uint8_t* src = static_cast<uint8_t*>(aligned_alloc(64, max_size));
    uint8_t* dst = static_cast<uint8_t*>(aligned_alloc(64, max_size));

    vector<double> MBps;
    for (const auto n : n_MB) {
        MBps.emplace_back(shared_mem_test(n, src, dst, warmup_src, warmup_dst));
    }

    cout << "**** Mem Test Result ****" << endl;
    cout << "size / MB" << "\t" << "rate / MBps" << endl;
    for (size_t i = 0; i < n_MB.size(); i++) {
        cout << n_MB[i] << "\t" << MBps[i] << endl;
    }

    delete[] warmup_src;
    delete[] warmup_dst;
    free(src);
    free(dst);

    return 0;
}