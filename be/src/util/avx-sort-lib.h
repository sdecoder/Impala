#include <math.h>
#include <immintrin.h>
#include <algorithm>
#include <stdint.h>
#include "runtime/tuple.h"
#include "runtime/tuple-row.h"

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#ifndef L2_CACHE_SIZE
#define L2_CACHE_SIZE (256 * 1024)
#endif

#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE (20 * 1024 * 1024)
#endif

/** Number of tuples that can fit into L2 cache divided by 2 */
#ifndef BLOCKSIZE
#define BLOCKSIZE (L2_CACHE_SIZE / (2 * sizeof(int64_t)))
#endif

#ifndef L3BLOCKSIZE
#define L3BLOCKSIZE (L3_CACHE_SIZE / sizeof(tuple_t))
#endif

#ifndef LOG2_BLOCKSIZE
#define LOG2_BLOCKSIZE (log2(BLOCKSIZE))
#endif

#ifndef MERGEBITONICWIDTH
#define MERGEBITONICWIDTH 16
#endif

#ifndef UPPER
#define UPPER 0x100000000
#endif


#ifndef PTR_MASK
#define PTR_MASK 0xFFFFF
#endif


#ifndef LO32F_MASK
#define LO32F_MASK 0xFFFFFFFF
#endif

#ifndef LO64F_MASK
#define LO64F_MASK 0xFFFFFFFFFFFFFFFF
#endif




typedef int64_t tuple_t;
typedef __int128_t int128_t;
typedef struct relation_t relation_t;
struct relation_t {
  tuple_t* tuples;
  uint64_t num_tuples;
};

typedef struct merge_node_t merge_node_t;
struct merge_node_t {
  tuple_t* buffer;
  volatile uint32_t count;
  volatile uint32_t head;
  volatile uint32_t tail;
} __attribute__((packed));

typedef struct block4 { int64_t val[4]; } block4;
typedef struct block8 { int64_t val[8]; } block8;
typedef struct block16 { int64_t val[16]; } block16;

/** L3 Cache size of the system in bytes. */
#ifndef L3_CACHE_SIZE
#define L3_CACHE_SIZE (20 * 1024 * 1024)
#endif

#ifndef L3BLOCKSIZE
#define L3BLOCKSIZE (L3_CACHE_SIZE / sizeof(tuple_t))
#endif

inline void SetPtr(int64_t& carrier, int ptr) {
  int _ptr = ptr & PTR_MASK;
  carrier = carrier | _ptr;
}

inline int GetPtr(int64_t& carrier) { return (int)(carrier & PTR_MASK); }

inline void SetInfPos(int64_t& carrier) {
  // 9214364837600034817;
  carrier = carrier | 0x7FE8000000000000;
}

inline void SetInfNeg(int64_t& carrier) {
  //-9007199254740991;
  carrier = carrier | 0xFFE8000000000000;
}

inline void SetKeyInt(int64_t& carrier, int64_t key) {
  if (key >= 0)
    carrier = carrier | (key << 20);
  else {
    carrier = carrier | (((int64_t)1) << 63);
    int64_t reg = -key;
    carrier = carrier | (reg << 20);
  }
}

inline void SetKeyFloat(int64_t& carrier, float key) {
  if (key >= 0) {
    int kbits = *reinterpret_cast<int*>(&key);
    carrier = carrier | ((int64_t)kbits << 20);
  } else {
    carrier = carrier | (((int64_t)1) << 63);
    key = -key;
    int kbits = *reinterpret_cast<int*>(&key);
    carrier = carrier | ((int64_t)kbits << 20);
  }
}

inline int GetKeyInt(int64_t& carrier) {
  int64_t sig = carrier & (((int64_t)1) << 63);
  if (sig < 0) {
    int reg_ = (int)((carrier & 0x000FFFFFFFF00000) >> 20);
    return -reg_;
  } else {
    return (int)((carrier & 0x000FFFFFFFF00000) >> 20);
  }
}

inline float GetKeyFloat(int64_t& carrier) {
  int64_t sig = carrier & (((int64_t)1) << 63);
  if (sig < 0) {
    int value = (int)((carrier & 0x000FFFFFFFF00000) >> 20);
    return -*reinterpret_cast<float*>(&value);
  } else {
    int value = (int)((carrier & 0x000FFFFFFFF00000) >> 20);
    return *reinterpret_cast<float*>(&value);
  }
}

inline void set_ptrkey_i64_hi32(int64_t& carrier, int ptr, int64_t source) {
  carrier = 0;
  if (source >= 0) {
    source = (source >> 32) & LO32F_MASK;
    carrier = carrier | (source << 20);
    carrier = carrier | ((int64_t)ptr & PTR_MASK);
  } else {
    carrier = carrier | (((int64_t)1) << 63);
    source = -source;
    source = (source >> 32) & LO32F_MASK;
    carrier = carrier | (source << 20);
    carrier = carrier | ((int64_t)ptr & PTR_MASK);
  }
}

inline void set_ptrkey_i64_lo32(int64_t& carrier, int ptr, int64_t source) {
  carrier = 0;
  source = source & LO32F_MASK;
  carrier = carrier | (source << 20);
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

// i64
inline bool compare_i64_hi32bit(int64_t x, int64_t y) {
  int64_t x_hi32bit = x & 0xFFFFFFFF00000000;
  int64_t y_hi32bit = y & 0xFFFFFFFF00000000;
  if (x_hi32bit == y_hi32bit) return true;
  return false;
}



// 128bit

inline void set_ptrkey_i128_127_96(int64_t& carrier, int ptr, int128_t source) {
  carrier = 0;

  if (source >= 0) {
    source = (source >> 96) & LO32F_MASK;
    carrier = carrier | (source << 20);
    carrier = carrier | ((int64_t)ptr & PTR_MASK);
  } else {
    carrier = carrier | (((int64_t)1) << 63);
    source = -source;
    source = (source >> 96) & LO32F_MASK;
    carrier = carrier | (source << 20);
    carrier = carrier | ((int64_t)ptr & PTR_MASK);
  }

}

inline void set_ptrkey_i128_95_64(int64_t& carrier, int ptr, int128_t source) {
  carrier = 0;
  source = (source >> 64) & LO32F_MASK;
  carrier = carrier | (source << 20);
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

inline void set_ptrkey_i128_63_32(int64_t& carrier, int ptr, int128_t source) {
  carrier = 0;
  source = (source >> 32) & LO32F_MASK;
  carrier = carrier | (source << 20);
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

inline void set_ptrkey_i128_31_0(int64_t& carrier, int ptr, int128_t source) {
  carrier = 0;
  source = source & LO32F_MASK;
  carrier = carrier | (source << 20);
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

inline bool compare_i128_hi127_96bit(int128_t x, int128_t y) {
  // compare the 127 to 96
  int128_t mask = LO32F_MASK;
  mask <<= 96; // 0xFFFFFFFF000000000000000000000000;

  int128_t val_x = x & mask;
  int128_t val_y = y & mask;
  if (val_x == val_y) return true;
  return false;
}

inline bool compare_i128_hi127_64bit(int128_t x, int128_t y) {
  // compare the 127 to 64
  // 0xFFFFFFFFFFFFFFFF0000000000000000
  int128_t mask = LO64F_MASK;
  mask <<= 64;

  int128_t val_x = x & mask;
  int128_t val_y = y & mask;
  if (val_x == val_y) return true;
  return false;
}


inline bool compare_i128_hi127_32bit(int128_t x, int128_t y) {

  // compare the bits from 128 to 32
  //0xFFFFFFFFFFFFFFFFFFFFFFFF00000000
  int128_t mask = LO64F_MASK;
  mask <<= 64;
  mask |= 0xFFFFFFFF00000000;

  int128_t val_x = x & mask;
  int128_t val_y = y & mask;
  if (val_x == val_y) return true;
  return false;
}



inline void set_ptrkey_for_d64_hi32bit(int64_t& carrier, int ptr,
                                       double source) {
  carrier = *reinterpret_cast<int64_t*>(&source);
  carrier = carrier & 0xFFFFFFFFFFF00000;
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

inline void set_ptrkey_for_d64_lo20bit(int64_t& carrier, int ptr,
                                       double source) {
  carrier = *reinterpret_cast<int64_t*>(&source);
  // clear the higher 32bits in the fraction part;
  carrier = carrier & 0xFFF00000000FFFFF;
  int64_t low20bit = carrier & PTR_MASK;
  carrier = carrier | (low20bit << 20);

  carrier = carrier & 0xFFFFFFFFFFF00000;
  carrier = carrier | ((int64_t)ptr & PTR_MASK);
}

inline int get_ptr_for_d64(double data) {
  int64_t i64_pre = *reinterpret_cast<int64_t*>(&data);
  return (int)(i64_pre & PTR_MASK);
}

inline bool compare_d64_hi32bit(double d1, double d2) {
  int64_t i64v1 = *(int64_t*)&d1;
  int64_t i64v2 = *(int64_t*)&d2;
  int64_t hi32_1 = i64v1 & 0x000FFFFFFFF00000;
  int64_t hi32_2 = i64v2 & 0x000FFFFFFFF00000;

  if (hi32_1 == hi32_2) return true;
  return false;
}

inline int trans_4char_to_int(const char* str, int len) {
  int result = 0;
  if (len == 1)
    return (int64_t)str[0] << 24;
  else if (len == 2) {
    result |= (int64_t)str[0] << 24;
    result |= (int64_t)str[1] << 16;
    return result;
  } else if (len == 3) {
    result |= (int64_t)str[0] << 24;
    result |= (int64_t)str[1] << 16;
    result |= (int64_t)str[2] << 8;
    return result;

  } else {
    // (len == 4)
    result |= (int64_t)str[0] << 24;
    result |= (int64_t)str[1] << 16;
    result |= (int64_t)str[2] << 8;
    result |= (int64_t)str[3];
    return result;
  }
}

// AVX Sort Interface
void avxsort_int64(int64_t** inputptr, int64_t** outputptr, uint64_t nitems);
void avxsort_tuples(tuple_t** inputptr, tuple_t** outputptr, uint64_t nitems);
void avxsortmultiway_tuples(tuple_t** inputptr, tuple_t** outputptr,
                            uint64_t nitems);
void avxsortmultiway_int64(int64_t** inputptr, int64_t** outputptr,
                           uint64_t nitems);
