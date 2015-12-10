#include <immintrin.h>
#include <math.h>
#include <stdlib.h>
#include <pthread.h>
#include <algorithm>
#include <iostream>

#include "runtime/tuple.h"
#include "runtime/tuple-row.h"
#include "avx-sort-lib.h"
using namespace std;

/*
struct relation_t {
  tuple_t * tuples;
  uint64_t  num_tuples;
};
typedef struct relation_t relation_t;*/

void avxsortmultiway_tuples(tuple_t **inputptr, tuple_t **outputptr,
                            uint64_t nitems);
void avxsort_tuples(tuple_t **inputptr, tuple_t **outputptr, uint64_t nitems);

void avxsort_int64(int64_t **inputptr, int64_t **outputptr, uint64_t nitems);
void avxsort_int32(int32_t **inputptr, int32_t **outputptr, uint64_t nitems);

void avxsort_aligned(int64_t **inputptr, int64_t **outputptr, uint64_t nitems);
void avxsort_unaligned(int64_t **inputptr, int64_t **outputptr,
                       uint64_t nitems);

void avxsort_block_aligned(int64_t **inputptr, int64_t **outputptr,
                           int BLOCK_SIZE);
void avxsort_block(int64_t **inputptr, int64_t **outputptr, int BLOCK_SIZE);

void avxsort_rem(int64_t **inputptr, int64_t **outputptr, uint32_t nitems);
void avxsort_rem_aligned(int64_t **inputptr, int64_t **outputptr,
                         uint32_t nitems);

void merge4_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
                  const uint32_t len);
void merge4_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                          int64_t *const out, const uint32_t len);

void merge8_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
                  const uint32_t len);
void merge8_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                          int64_t *const out, const uint32_t len);
void merge8_varlen(int64_t *inpA, int64_t *inpB, int64_t *Out,
                   const uint32_t lenA, const uint32_t lenB);
void merge8_varlen_aligned(int64_t *inpA, int64_t *inpB, int64_t *Out,
                           const uint32_t lenA, const uint32_t lenB);

void merge16_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
                   const uint32_t len);
void merge16_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                           int64_t *const out, const uint32_t len);
void merge16_varlen(int64_t *inpA, int64_t *inpB, int64_t *Out,
                    const uint32_t lenA, const uint32_t lenB);
void merge16_varlen_aligned(int64_t *inpA, int64_t *inpB, int64_t *Out,
                            const uint32_t lenA, const uint32_t lenB);

void avxsortmultiway_int64(int64_t **inputptr, int64_t **outputptr,
                           uint64_t nitems);
uint64_t avx_merge_int64(int64_t *const inpA, int64_t *const inpB,
                         int64_t *const out, const uint64_t lenA,
                         const uint64_t lenB);
static void *malloc_aligned(size_t size);
uint64_t avx_multiway_merge(tuple_t *output, relation_t **parts,
                            uint32_t nparts, tuple_t *fifobuffer,
                            uint32_t bufntuples);
uint32_t readmerge_parallel_decomposed(merge_node_t *node, tuple_t **inA,
                                       tuple_t **inB, uint32_t lenA,
                                       uint32_t lenB, uint32_t fifosize);
uint64_t mergestore_parallel_decomposed(merge_node_t *right, merge_node_t *left,
                                        tuple_t **output, uint32_t fifosize,
                                        uint8_t rightdone, uint8_t leftdone);
void merge_parallel_decomposed(merge_node_t *node, merge_node_t *right,
                               merge_node_t *left, uint32_t fifosize,
                               uint8_t rightdone, uint8_t leftdone);
uint32_t direct_copy_to_output_avx(tuple_t *dest, merge_node_t *src,
                                   uint32_t fifosize);
void direct_copy_avx(merge_node_t *dest, merge_node_t *src, uint32_t fifosize);

void mergestore16kernel(tuple_t *A, tuple_t *B, tuple_t **Out, uint32_t *ri,
                        uint32_t *li, uint32_t rend, uint32_t lend);
void merge16kernel(tuple_t *A, tuple_t *B, tuple_t *Out, uint32_t *ri,
                   uint32_t *li, uint32_t *oi, uint32_t *outnslots,
                   uint32_t rend, uint32_t lend);

void parallel_read(tuple_t **A, tuple_t **B, tuple_t **Out, uint32_t *ri,
                   uint32_t *li, uint32_t *oi, uint32_t *outnslots,
                   uint32_t lenA, uint32_t lenB);

void simd_memcpy(void *dst, void *src, size_t sz);

void inregister_sort_keyval32(int64_t *items, int64_t *output);
void inregister_sort_keyval32_aligned(int64_t *items, int64_t *output);

inline __attribute__((__always_inline__)) void swap(int64_t **A, int64_t **B) {
  int64_t *tmp = *A;
  *A = *B;
  *B = tmp;
}
inline __attribute__((__always_inline__)) uint32_t asm_log2(const uint32_t n) {
  register uint32_t res;
  __asm__("\tbsr %1, %0\n" : "=r"(res) : "r"(n));
  return res;
}

#define IFELSECONDMOVE(NXT, INA, INB, INCR)           \
  do {                                                \
    int8_t cmp = *((double *)INA) < *((double *)INB); \
    NXT = cmp ? INA : INB;                            \
    INA += cmp;                                       \
    INB += !cmp;                                      \
  } while (0)

/** Load 2 AVX 256-bit registers from the given address */
#define LOAD8(REGL, REGH, ADDR)                                    \
  do {                                                             \
    REGL = _mm256_load_pd((double const *)ADDR);                   \
    REGH = _mm256_load_pd((double const *)(((block4 *)ADDR) + 1)); \
  } while (0)

/** Load unaligned 2 AVX 256-bit registers from the given address */
#define LOAD8U(REGL, REGH, ADDR)                                    \
  do {                                                              \
    REGL = _mm256_loadu_pd((double const *)ADDR);                   \
    REGH = _mm256_loadu_pd((double const *)(((block4 *)ADDR) + 1)); \
  } while (0)

/** Store 2 AVX 256-bit registers to the given address */
#define STORE8(ADDR, REGL, REGH)                             \
  do {                                                       \
    _mm256_store_pd((double *)ADDR, REGL);                   \
    _mm256_store_pd((double *)(((block4 *)ADDR) + 1), REGH); \
  } while (0)

/** Store unaligned 2 AVX 256-bit registers to the given address */
#define STORE8U(ADDR, REGL, REGH)                             \
  do {                                                        \
    _mm256_storeu_pd((double *)ADDR, REGL);                   \
    _mm256_storeu_pd((double *)(((block4 *)ADDR) + 1), REGH); \
  } while (0)

/**
 * @note Reversing 64-bit values in an AVX register.
 */
#define REVERSE(REG)                        \
  do {                                      \
    REG = _mm256_permute4x64_pd(REG, 0x1B); \
  } while (0)

/** Bitonic merge kernel for 2 x 4 elements after the reversing step. */
#define BITONIC4(O1, O2, A, B)                                  \
  do {                                                          \
    /* Level-1 comparisons */                                   \
    __m256d l1 = _mm256_min_pd(A, B);                           \
    __m256d h1 = _mm256_max_pd(A, B);                           \
                                                                \
    /* Level-1 shuffles */                                      \
    __m256d l1p = _mm256_permute2f128_pd(l1, h1, 0x31);         \
    __m256d h1p = _mm256_permute2f128_pd(l1, h1, 0x20);         \
                                                                \
    /* Level-2 comparisons */                                   \
    __m256d l2 = _mm256_min_pd(l1p, h1p);                       \
    __m256d h2 = _mm256_max_pd(l1p, h1p);                       \
                                                                \
    /* Level-2 shuffles */                                      \
    __m256d l2p = _mm256_shuffle_pd(l2, h2, 0x0);               \
    __m256d h2p = _mm256_shuffle_pd(l2, h2, 0xF);               \
                                                                \
    /* Level-3 comparisons */                                   \
    __m256d l3 = _mm256_min_pd(l2p, h2p);                       \
    __m256d h3 = _mm256_max_pd(l2p, h2p);                       \
                                                                \
    /* Level-3 shuffles implemented with unpcklps unpckhps */   \
    /* AVX cannot shuffle both inputs from same 128-bit lane */ \
    /* so we need 2 more instructions for this operation. */    \
    __m256d l4 = _mm256_unpacklo_pd(l3, h3);                    \
    __m256d h4 = _mm256_unpackhi_pd(l3, h3);                    \
    O1 = _mm256_permute2f128_pd(l4, h4, 0x20);                  \
    O2 = _mm256_permute2f128_pd(l4, h4, 0x31);                  \
  } while (0)

/** Bitonic merge network for 2 x 8 elements without reversing B */
#define BITONIC8(O1, O2, O3, O4, A1, A2, B1, B2) \
  do {                                           \
    /* Level-0 comparisons */                    \
    __m256d l11 = _mm256_min_pd(A1, B1);         \
    __m256d l12 = _mm256_min_pd(A2, B2);         \
    __m256d h11 = _mm256_max_pd(A1, B1);         \
    __m256d h12 = _mm256_max_pd(A2, B2);         \
                                                 \
    BITONIC4(O1, O2, l11, l12);                  \
    BITONIC4(O3, O4, h11, h12);                  \
  } while (0)

/** Bitonic merge kernel for 2 x 4 elements */
#define BITONIC_MERGE4(O1, O2, A, B)            \
  do {                                          \
    /* reverse the order of input register B */ \
    REVERSE(B);                                 \
    BITONIC4(O1, O2, A, B);                     \
  } while (0)

/** Bitonic merge kernel for 2 x 8 elements */
#define BITONIC_MERGE8(O1, O2, O3, O4, A1, A2, B1, B2) \
  do {                                                 \
    /* reverse the order of input B */                 \
    REVERSE(B1);                                       \
    REVERSE(B2);                                       \
                                                       \
    /* Level-0 comparisons */                          \
    __m256d l11 = _mm256_min_pd(A1, B2);               \
    __m256d l12 = _mm256_min_pd(A2, B1);               \
    __m256d h11 = _mm256_max_pd(A1, B2);               \
    __m256d h12 = _mm256_max_pd(A2, B1);               \
                                                       \
    BITONIC4(O1, O2, l11, l12);                        \
    BITONIC4(O3, O4, h11, h12);                        \
  } while (0)

/** Bitonic merge kernel for 2 x 16 elements */
#define BITONIC_MERGE16(O1, O2, O3, O4, O5, O6, O7, O8, A1, A2, A3, A4, B1, \
                        B2, B3, B4)                                         \
  do {                                                                      \
    /** Bitonic merge kernel for 2 x 16 elemenets */                        \
    /* reverse the order of input B */                                      \
    REVERSE(B1);                                                            \
    REVERSE(B2);                                                            \
    REVERSE(B3);                                                            \
    REVERSE(B4);                                                            \
                                                                            \
    /* Level-0 comparisons */                                               \
    __m256d l01 = _mm256_min_pd(A1, B4);                                    \
    __m256d l02 = _mm256_min_pd(A2, B3);                                    \
    __m256d l03 = _mm256_min_pd(A3, B2);                                    \
    __m256d l04 = _mm256_min_pd(A4, B1);                                    \
    __m256d h01 = _mm256_max_pd(A1, B4);                                    \
    __m256d h02 = _mm256_max_pd(A2, B3);                                    \
    __m256d h03 = _mm256_max_pd(A3, B2);                                    \
    __m256d h04 = _mm256_max_pd(A4, B1);                                    \
                                                                            \
    BITONIC8(O1, O2, O3, O4, l01, l02, l03, l04);                           \
    BITONIC8(O5, O6, O7, O8, h01, h02, h03, h04);                           \
  } while (0)

void __attribute__((target("avx2")))
avxsort_tuples(tuple_t **inputptr, tuple_t **outputptr, uint64_t nitems) {
  int64_t *input = (int64_t *)(*inputptr);
  int64_t *output = (int64_t *)(*outputptr);

  /* choose actual implementation depending on the input alignment */
  if (((uintptr_t)input % CACHE_LINE_SIZE) == 0 &&
      ((uintptr_t)output % CACHE_LINE_SIZE) == 0)
    avxsort_aligned(&input, &output, nitems);
  else
    avxsort_unaligned(&input, &output, nitems);

  *inputptr = (tuple_t *)(input);
  *outputptr = (tuple_t *)(output);
}

/*
void avxsort_int64(int64_t ** inputptr, int64_t ** outputptr, uint64_t nitems){
#ifdef __AVX2__
  int64_t * input  = (int64_t*)(*inputptr);
  int64_t * output = (int64_t*)(*outputptr);
  /* choose actual implementation depending on the input alignment */
/* if(((uintptr_t)input % CACHE_LINE_SIZE) == 0
    && ((uintptr_t)output % CACHE_LINE_SIZE) == 0)
     avxsort_aligned(&input, &output, nitems);
 else
     avxsort_unaligned(&input, &output, nitems);

 *inputptr = (int64_t *)(input);
 *outputptr = (int64_t *)(output);
#else
   sort(*inputptr, *inputptr + nitems);
   *outputptr = *inputptr;
#endif

}*/

void __attribute__((target("avx2")))
avxsort_int64(int64_t **inputptr, int64_t **outputptr, uint64_t nitems) {
  /* \todo: implement */
  int64_t *input = (int64_t *)(*inputptr);
  int64_t *output = (int64_t *)(*outputptr);

  /* choose actual implementation depending on the input alignment */
  if (((uintptr_t)input % CACHE_LINE_SIZE) == 0 &&
      ((uintptr_t)output % CACHE_LINE_SIZE) == 0)
    avxsort_aligned(&input, &output, nitems);
  else
    avxsort_unaligned(&input, &output, nitems);

  *inputptr = (int64_t *)(input);
  *outputptr = (int64_t *)(output);
}

void __attribute__((target("avx2")))
avxsort_aligned(int64_t **inputptr, int64_t **outputptr, uint64_t nitems) {
  if (nitems <= 0) return;
  int64_t *input = *inputptr;
  int64_t *output = *outputptr;

  uint64_t i;
  uint64_t nchunks = (nitems / BLOCKSIZE);
  int rem = (nitems % BLOCKSIZE);
  /* printf("nchunks = %d, nitems = %d, rem = %d\n", nchunks, nitems, rem); */
  /* each chunk keeps track of its temporary memory offset */
  int64_t *ptrs[nchunks + 1][2]; /* [chunk-in, chunk-out-tmp] */
  uint32_t sizes[nchunks + 1];

  for (i = 0; i <= nchunks; i++) {
    ptrs[i][0] = input + i * BLOCKSIZE;
    ptrs[i][1] = output + i * BLOCKSIZE;
    sizes[i] = BLOCKSIZE;
  }

  /** 1) Divide the input into chunks fitting into L2 cache. */
  /* one more chunk if not divisible */
  for (i = 0; i < nchunks; i++) {
    //      LOG(INFO) << "[dbg] " << __FILE__  << __LINE__ ;
    avxsort_block_aligned(&ptrs[i][0], &ptrs[i][1], BLOCKSIZE);
    swap(&ptrs[i][0], &ptrs[i][1]);
  }

  if (rem) {
    /* sort the last chunk which is less than BLOCKSIZE */
    //  LOG(INFO) << "[dbg] " << __FILE__  << __LINE__ ;
    avxsort_rem_aligned(&ptrs[i][0], &ptrs[i][1], rem);
    swap(&ptrs[i][0], &ptrs[i][1]);
    sizes[i] = rem;
  }

  //===================================================================

  /**
   * 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
   * obtain sorted sequences of length 2^{itr+1}.
   */
  nchunks += (rem > 0);
  /* printf("Merge chunks = %d\n", nchunks); */
  const uint64_t logN = ceil(log2(nitems));
  for (i = LOG2_BLOCKSIZE; i < logN; i++) {
    uint64_t k = 0;
    for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
      int64_t *inpA = ptrs[j][0];
      int64_t *inpB = ptrs[j + 1][0];
      int64_t *out = ptrs[j][1];
      uint32_t sizeA = sizes[j];
      uint32_t sizeB = sizes[j + 1];

      merge8_varlen_aligned(inpA, inpB, out, sizeA, sizeB);

      /* setup new pointers */
      ptrs[k][0] = out;
      ptrs[k][1] = inpA;
      sizes[k] = sizeA + sizeB;
      k++;
    }

    if ((nchunks % 2)) {
      /* just move the pointers */
      ptrs[k][0] = ptrs[nchunks - 1][0];
      ptrs[k][1] = ptrs[nchunks - 1][1];
      sizes[k] = sizes[nchunks - 1];
      k++;
    }

    nchunks = k;
  }

  /* finally swap input/output pointers, where output holds the sorted list */
  *outputptr = ptrs[0][0];
  *inputptr = ptrs[0][1];
}

void __attribute__((target("avx2")))
avxsort_unaligned(int64_t **inputptr, int64_t **outputptr, uint64_t nitems) {
  if (nitems <= 0) return;

  int64_t *input = *inputptr;
  int64_t *output = *outputptr;

  uint64_t i;
  uint64_t nchunks = (nitems / BLOCKSIZE);
  int rem = (nitems % BLOCKSIZE);

  /* each chunk keeps track of its temporary memory offset */
  int64_t *ptrs[nchunks + 1][2]; /* [chunk-in, chunk-out-tmp] */
  uint32_t sizes[nchunks + 1];

  for (i = 0; i <= nchunks; i++) {
    ptrs[i][0] = input + i * BLOCKSIZE;
    ptrs[i][1] = output + i * BLOCKSIZE;
    sizes[i] = BLOCKSIZE;
  }

  /** 1) Divide the input into chunks fitting into L2 cache. */
  /* one more chunk if not divisible */
  for (i = 0; i < nchunks; i++) {
    avxsort_block(&ptrs[i][0], &ptrs[i][1], BLOCKSIZE);
    swap(&ptrs[i][0], &ptrs[i][1]);
  }

  if (rem) {
    /* sort the last chunk which is less than BLOCKSIZE */
    avxsort_rem(&ptrs[i][0], &ptrs[i][1], rem);
    swap(&ptrs[i][0], &ptrs[i][1]);
    sizes[i] = rem;
  }

  /**
   * 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
   * obtain sorted sequences of length 2^{itr+1}.
   */
  nchunks += (rem > 0);
  /* printf("Merge chunks = %d\n", nchunks); */
  const uint64_t logN = ceil(log2(nitems));
  for (i = LOG2_BLOCKSIZE; i < logN; i++) {
    uint64_t k = 0;
    for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
      int64_t *inpA = ptrs[j][0];
      int64_t *inpB = ptrs[j + 1][0];
      int64_t *out = ptrs[j][1];
      uint32_t sizeA = sizes[j];
      uint32_t sizeB = sizes[j + 1];

      merge16_varlen(inpA, inpB, out, sizeA, sizeB);

      /* setup new pointers */
      ptrs[k][0] = out;
      ptrs[k][1] = inpA;
      sizes[k] = sizeA + sizeB;
      k++;
    }

    if ((nchunks % 2)) {
      /* just move the pointers */
      ptrs[k][0] = ptrs[nchunks - 1][0];
      ptrs[k][1] = ptrs[nchunks - 1][1];
      sizes[k] = sizes[nchunks - 1];
      k++;
    }

    nchunks = k;
  }

  /* finally swap input/output pointers, where output holds the sorted list */
  *outputptr = ptrs[0][0];
  *inputptr = ptrs[0][1];
}

/**
 * Merge two sorted arrays to a final output using 8-way AVX bitonic merge.
 *
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline, target("avx2")))
merge8_varlen_aligned(int64_t *inpA, int64_t *inpB, int64_t *Out,
                      const uint32_t lenA, const uint32_t lenB) {
  uint32_t lenA8 = lenA & ~0x7, lenB8 = lenB & ~0x7;
  uint32_t ai = 0, bi = 0;
  int64_t *out = Out;
  if (lenA8 > 8 && lenB8 > 8) {
    register block8 *inA = (block8 *)inpA;
    register block8 *inB = (block8 *)inpB;
    block8 *const endA = (block8 *)(inpA + lenA) - 1;
    block8 *const endB = (block8 *)(inpB + lenB) - 1;

    block8 *outp = (block8 *)out;
    register block8 *next = inB;

    register __m256d outreg1l, outreg1h;
    register __m256d outreg2l, outreg2h;

    register __m256d regAl, regAh;
    register __m256d regBl, regBh;

    LOAD8(regAl, regAh, inA);
    LOAD8(regBl, regBh, next);

    inA++;
    inB++;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l, outreg1h);
    outp++;

    while (inA < endA && inB < endB) {
      /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
      IFELSECONDMOVE(next, inA, inB, 64);

      regAl = outreg2l;
      regAh = outreg2h;
      LOAD8(regBl, regBh, next);

      BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh,
                     regBl, regBh);

      /* store outreg1 */
      STORE8(outp, outreg1l, outreg1h);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(32)));
    _mm256_store_pd((double *)hireg, outreg2h);
    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      STORE8(inA, outreg2l, outreg2h);
    } else {
      /* store the last remaining register values to B */
      inB--;
      STORE8(inB, outreg2l, outreg2h);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8_ = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8_);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8_ = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8_);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}

/**
 * Merge two sorted arrays to a final output using 4-way AVX bitonic merge.
 *
 * @param inpA input array A
 * @param inpB input array B
 * @param Out  output array
 * @param lenA size of A
 * @param lenB size of B
 */
inline void __attribute__((always_inline, target("avx2")))
merge4_varlen(int64_t *inpA, int64_t *inpB, int64_t *Out, const uint32_t lenA,
              const uint32_t lenB) {
  uint32_t lenA4 = lenA & ~0x3, lenB4 = lenB & ~0x3;
  uint32_t ai = 0, bi = 0;
  int64_t *out = Out;
  if (lenA4 > 4 && lenB4 > 4) {
    register block4 *inA = (block4 *)inpA;
    register block4 *inB = (block4 *)inpB;
    block4 *const endA = (block4 *)(inpA + lenA) - 1;
    block4 *const endB = (block4 *)(inpB + lenB) - 1;
    block4 *outp = (block4 *)out;
    register block4 *next = inB;
    register __m256d outreg1;
    register __m256d outreg2;

    register __m256d regA = _mm256_loadu_pd((double const *)inA);
    register __m256d regB = _mm256_loadu_pd((double const *)next);

    inA++;
    inB++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    /* store outreg1 */
    _mm256_storeu_pd((double *)outp, outreg1);
    outp++;

    while (inA < endA && inB < endB) {
      /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
      IFELSECONDMOVE(next, inA, inB, 32);

      regA = outreg2;
      regB = _mm256_loadu_pd((double const *)next);

      BITONIC_MERGE4(outreg1, outreg2, regA, regB);

      /* store outreg1 */
      _mm256_storeu_pd((double *)outp, outreg1);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(16)));
    _mm256_store_pd((double *)hireg, outreg2);

    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      _mm256_storeu_pd((double *)inA, outreg2);
    } else {
      /* store the last remaining register values to B */
      inB--;
      _mm256_storeu_pd((double *)inB, outreg2);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8 = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8 = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}

/** aligned version */
inline void __attribute__((always_inline, target("avx2")))
merge4_varlen_aligned(int64_t *inpA, int64_t *inpB, int64_t *Out,
                      const uint32_t lenA, const uint32_t lenB) {
  uint32_t lenA4 = lenA & ~0x3, lenB4 = lenB & ~0x3;
  uint32_t ai = 0, bi = 0;
  int64_t *out = Out;

  if (lenA4 > 4 && lenB4 > 4) {
    register block4 *inA = (block4 *)inpA;
    register block4 *inB = (block4 *)inpB;
    block4 *const endA = (block4 *)(inpA + lenA) - 1;
    block4 *const endB = (block4 *)(inpB + lenB) - 1;

    block4 *outp = (block4 *)out;

    register block4 *next = inB;
    register __m256d outreg1;
    register __m256d outreg2;

    register __m256d regA = _mm256_load_pd((double const *)inA);
    register __m256d regB = _mm256_load_pd((double const *)next);

    inA++;
    inB++;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    /* store outreg1 */
    _mm256_store_pd((double *)outp, outreg1);
    outp++;

    while (inA < endA && inB < endB) {
      /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
      IFELSECONDMOVE(next, inA, inB, 32);

      regA = outreg2;
      regB = _mm256_load_pd((double const *)next);

      BITONIC_MERGE4(outreg1, outreg2, regA, regB);

      /* store outreg1 */
      _mm256_store_pd((double *)outp, outreg1);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(32)));
    _mm256_store_pd((double *)hireg, outreg2);

    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      _mm256_store_pd((double *)inA, outreg2);
    } else {
      /* store the last remaining register values to B */
      inB--;
      _mm256_store_pd((double *)inB, outreg2);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8 = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8 = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}

inline void __attribute__((always_inline, target("avx2")))
inregister_sort_keyval32(int64_t *items, int64_t *output) {
  __m256d ra = _mm256_loadu_pd((double const *)(items));
  __m256d rb = _mm256_loadu_pd((double const *)(items + 4));
  __m256d rc = _mm256_loadu_pd((double const *)(items + 8));
  __m256d rd = _mm256_loadu_pd((double const *)(items + 12));

  /* odd-even sorting network begins */
  /* 1st level of comparisons */
  __m256d ra1 = _mm256_min_pd(ra, rb);
  __m256d rb1 = _mm256_max_pd(ra, rb);

  __m256d rc1 = _mm256_min_pd(rc, rd);
  __m256d rd1 = _mm256_max_pd(rc, rd);

  /* 2nd level of comparisons */
  rb = _mm256_min_pd(rb1, rd1);
  rd = _mm256_max_pd(rb1, rd1);

  /* 3rd level of comparisons */
  __m256d ra2 = _mm256_min_pd(ra1, rc1);
  __m256d rc2 = _mm256_max_pd(ra1, rc1);

  /* 4th level of comparisons */
  __m256d rb3 = _mm256_min_pd(rb, rc2);
  __m256d rc3 = _mm256_max_pd(rb, rc2);

  /* results are in ra2, rb3, rc3, rd */
  /**
   * Initial data and transposed data looks like following:
   *  a2={ x1  x2  x3  x4  }                      a4={ x1 x5 x9  x13 }
   *  b3={ x5  x6  x7  x8  }  === Transpose ===>  b5={ x2 x6 x10 x14 }
   *  c3={ x9  x10 x11 x12 }                      c5={ x3 x7 x11 x15 }
   *  d2={ x13 x14 x15 x16 }                      d4={ x4 x8 x12 x16 }
   */
  /* shuffle x2 and x5 - shuffle x4 and x7 */
  __m256d ra3 = _mm256_unpacklo_pd(ra2, rb3);
  __m256d rb4 = _mm256_unpackhi_pd(ra2, rb3);

  /* shuffle x10 and x13 - shuffle x12 and x15 */
  __m256d rc4 = _mm256_unpacklo_pd(rc3, rd);
  __m256d rd3 = _mm256_unpackhi_pd(rc3, rd);

  /* shuffle (x3,x7) and (x9,x13) pairs */
  __m256d ra4 = _mm256_permute2f128_pd(ra3, rc4, 0x20);
  __m256d rc5 = _mm256_permute2f128_pd(ra3, rc4, 0x31);

  /* shuffle (x4,x8) and (x10,x14) pairs */
  __m256d rb5 = _mm256_permute2f128_pd(rb4, rd3, 0x20);
  __m256d rd4 = _mm256_permute2f128_pd(rb4, rd3, 0x31);

  /* after this, results are in ra4, rb5, rc5, rd4 */

  /* store */
  _mm256_storeu_pd((double *)output, ra4);
  _mm256_storeu_pd((double *)(output + 4), rb5);
  _mm256_storeu_pd((double *)(output + 8), rc5);
  _mm256_storeu_pd((double *)(output + 12), rd4);
}

inline void __attribute__((always_inline, target("avx2")))
avxsort_rem(int64_t **inputptr, int64_t **outputptr, uint32_t nitems) {
  int64_t *inp = *inputptr;
  int64_t *out = *outputptr;

#if 1 /* sort using AVX */
  /* each chunk keeps track of its temporary memory offset */
  int64_t *ptrs[8][2]; /* [chunk-in, chunk-out-tmp] */

  uint32_t n = nitems, pos = 0, i = 0;
  uint32_t nxtpow = 8192;
  uint32_t sizes[6];

  while (n < nxtpow) {
    nxtpow >>= 1;
  }

  while (nxtpow > 128) {
    ptrs[i][0] = inp + pos;
    ptrs[i][1] = out + pos;
    sizes[i] = nxtpow;

    avxsort_block(&ptrs[i][0], &ptrs[i][1], nxtpow);
    pos += nxtpow;
    n -= nxtpow;
    swap(&ptrs[i][0], &ptrs[i][1]);
    i++;

    while (n < nxtpow) {
      nxtpow >>= 1;
    }
  }

  if (n > 0) {
    /* sort last n < 128 items using scalar sort */
    ptrs[i][0] = inp + pos;
    ptrs[i][1] = out + pos;
    sizes[i] = n;

#ifdef __cplusplus
    std::sort(reinterpret_cast<double *>(ptrs[i][0]),
              reinterpret_cast<double *>(ptrs[i][0] + n));
//    mend_negative_part(ptrs[i][0], n);
#else
    qsort((double *)ptrs[i][0], n, sizeof(int64_t), keycmp);
#endif
    /* no need to swap */
    i++;
  }

  uint32_t nchunks = i;

  /* merge sorted blocks */
  while (nchunks > 1) {
    uint64_t k = 0;
    for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
      int64_t *inpA = ptrs[j][0];
      int64_t *inpB = ptrs[j + 1][0];
      int64_t *out = ptrs[j][1];
      uint32_t sizeA = sizes[j];
      uint32_t sizeB = sizes[j + 1];

      merge16_varlen(inpA, inpB, out, sizeA, sizeB);

      /* setup new pointers */
      ptrs[k][0] = out;
      ptrs[k][1] = inpA;
      sizes[k] = sizeA + sizeB;
      k++;
    }

    if ((nchunks % 2)) {
      /* just move the pointers */
      ptrs[k][0] = ptrs[nchunks - 1][0];
      ptrs[k][1] = ptrs[nchunks - 1][1];
      sizes[k] = sizes[nchunks - 1];
      k++;
    }

    nchunks = k;
  }

  /* finally swap input/output pointers, where output holds the sorted list */
  *outputptr = ptrs[0][0];
  *inputptr = ptrs[0][1];

#else /* sort using scalar */

#ifdef __cplusplus
  std::sort(reinterpret_cast<double *>(inp),
            reinterpret_cast<double *>(inp + nitems));
#else
  qsort(inp, nitems, sizeof(int64_t), keycmp);
#endif

  *outputptr = inp;
  *inputptr = out;

#endif
}

inline void __attribute__((always_inline, target("avx2")))
avxsort_block(int64_t **inputptr, int64_t **outputptr, int BLOCK_SIZE) {
  int64_t *ptrs[2];
  const uint64_t logBSZ = log2(BLOCK_SIZE);

  ptrs[0] = *inputptr;
  ptrs[1] = *outputptr;

  /** 1.a) Perform in-register sort to get sorted seq of K(K=4)*/
  block16 *inptr = (block16 *)ptrs[0];
  block16 *const end = (block16 *)(ptrs[0] + BLOCK_SIZE);
  while (inptr < end) {
    inregister_sort_keyval32((int64_t *)inptr, (int64_t *)inptr);
    inptr++;
  }

  /**
   * 1.b) for itr <- [(logK) .. (logM - 3)]
   *  - Simultaneously merge 4 sequences (using a K by K
   *  network) of length 2^itr to obtain sorted seq. of 2^{itr+1}
   */
  uint64_t j;
  const uint64_t jend = logBSZ - 2;

  j = 2;
  {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge4_eqlen(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;
    }
  }
  j = 3;
  {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge8_eqlen(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;
    }
  }
  for (j = 4; j < jend; j++) {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge16_eqlen(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;

      /* TODO: Try following. */
      /* simultaneous merge of 4 list pairs */
      /* merge 4 seqs simultaneously (always >= 4) */
      /* merge 2 seqs simultaneously (always >= 2) */
    }
  }

  /**
   * 1.c) for itr = (logM - 2), simultaneously merge 2 sequences
   *  (using a 2K by 2K network) of length M/4 to obtain sorted
   *  sequences of M/2.
   */
  uint64_t inlen = (1 << j);
  int64_t *inp;
  int64_t *out;
  int ptridx = j & 1;

  inp = ptrs[ptridx];
  out = ptrs[ptridx ^ 1];

  merge16_eqlen(inp, inp + inlen, out, inlen);
  merge16_eqlen(inp + 2 * inlen, inp + 3 * inlen, out + 2 * inlen, inlen);

  /* TODO: simultaneous merge of 2 list pairs */
  /**
   * 1.d) for itr = (logM - 1), merge 2 final sequences (using a
   * 4K by 4K network) of length M/2 to get sorted seq. of M.
   */
  j++; /* j=(LOG2_BLOCK_SIZE-1); inputsize M/2 --> outputsize M*/
  inlen = (1 << j);
  /* now we know that input is out from the last pass */
  merge16_eqlen(out, out + inlen, inp, inlen);

  /* finally swap input/output ptrs, output is the sorted list */
  *outputptr = inp;
  *inputptr = out;
}

inline void __attribute__((always_inline, target("avx2")))
avxsort_rem_aligned(int64_t **inputptr, int64_t **outputptr, uint32_t nitems) {
  int64_t *inp = *inputptr;
  int64_t *out = *outputptr;

#if 1 /* sort using AVX */
  /* each chunk keeps track of its temporary memory offset */
  int64_t *ptrs[8][2]; /* [chunk-in, chunk-out-tmp] */

  uint32_t n = nitems, pos = 0, i = 0;
  uint32_t nxtpow = 8192; /* TODO: infer from nitems, nearest pow2 to nitems */
  uint32_t sizes[6];

  while (n < nxtpow) {
    nxtpow >>= 1;
  }

  while (nxtpow > 128) {
    ptrs[i][0] = inp + pos;
    ptrs[i][1] = out + pos;
    sizes[i] = nxtpow;

    avxsort_block_aligned(&ptrs[i][0], &ptrs[i][1], nxtpow);
    pos += nxtpow;
    n -= nxtpow;
    swap(&ptrs[i][0], &ptrs[i][1]);
    i++;

    while (n < nxtpow) {
      nxtpow >>= 1;
    }
  }

  if (n > 0) {
    /* sort last n < 128 items using scalar sort */

    ptrs[i][0] = inp + pos;
    ptrs[i][1] = out + pos;
    sizes[i] = n;

#ifdef __cplusplus
    std::sort(reinterpret_cast<double *>(ptrs[i][0]),
              reinterpret_cast<double *>(ptrs[i][0] + n));
//  mend_negative_part(ptrs[i][0], n);
#else
    qsort((double *)ptrs[i][0], n, sizeof(int64_t), keycmp);
#endif
    /* no need to swap */
    i++;
  }

  uint32_t nchunks = i;

  /* merge sorted blocks */
  while (nchunks > 1) {
    uint64_t k = 0;
    for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
      int64_t *inpA = ptrs[j][0];
      int64_t *inpB = ptrs[j + 1][0];
      int64_t *out = ptrs[j][1];
      uint32_t sizeA = sizes[j];
      uint32_t sizeB = sizes[j + 1];

      merge16_varlen_aligned(inpA, inpB, out, sizeA, sizeB);

      /* setup new pointers */
      ptrs[k][0] = out;
      ptrs[k][1] = inpA;
      sizes[k] = sizeA + sizeB;
      k++;
    }

    if ((nchunks % 2)) {
      /* just move the pointers */
      ptrs[k][0] = ptrs[nchunks - 1][0];
      ptrs[k][1] = ptrs[nchunks - 1][1];
      sizes[k] = sizes[nchunks - 1];
      k++;
    }

    nchunks = k;
  }

  /* finally swap input/output pointers, where output holds the sorted list */
  *outputptr = ptrs[0][0];
  *inputptr = ptrs[0][1];

#else /* sort using scalar */

#ifdef __cplusplus
  std::sort(reinterpret_cast<double *>(inp),
            reinterpret_cast<double *>(inp + nitems));
#else
  qsort(inp, nitems, sizeof(int64_t), keycmp);
#endif

  *outputptr = inp;
  *inputptr = out;

#endif
}

inline void __attribute__((always_inline, target("avx2")))
merge16_varlen(int64_t *inpA, int64_t *inpB, int64_t *Out, const uint32_t lenA,
               const uint32_t lenB) {
  uint32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;
  uint32_t ai = 0, bi = 0;

  int64_t *out = Out;

  if (lenA16 > 16 && lenB16 > 16) {
    register block16 *inA = (block16 *)inpA;
    register block16 *inB = (block16 *)inpB;
    block16 *const endA = (block16 *)(inpA + lenA) - 1;
    block16 *const endB = (block16 *)(inpB + lenB) - 1;

    block16 *outp = (block16 *)out;

    register block16 *next = inB;

    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8U(regAl1, regAl2, inA);
    LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    LOAD8U(regBl1, regBl2, inB);
    LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;

    while (inA < endA && inB < endB) {
      /** The inline assembly below does exactly the following code: */
      /* Option 3: with assembly */
      IFELSECONDMOVE(next, inA, inB, 128);

      regAl1 = outreg2l1;
      regAl2 = outreg2l2;
      regAh1 = outreg2h1;
      regAh2 = outreg2h2;

      LOAD8U(regBl1, regBl2, next);
      LOAD8U(regBh1, regBh2, ((block8 *)next + 1));

      BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                      outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                      regAh2, regBl1, regBl2, regBh1, regBh2);

      /* store outreg1 */
      STORE8U(outp, outreg1l1, outreg1l2);
      STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(16)));
    _mm256_store_pd((double *)hireg, outreg2h2);

    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      STORE8U(inA, outreg2l1, outreg2l2);
      STORE8U(((block8 *)inA + 1), outreg2h1, outreg2h2);
    } else {
      /* store the last remaining register values to B */
      inB--;
      STORE8U(inB, outreg2l1, outreg2l2);
      STORE8U(((block8 *)inB + 1), outreg2h1, outreg2h2);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8 = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8 = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}

inline void __attribute__((always_inline, target("avx2")))
avxsort_block_aligned(int64_t **inputptr, int64_t **outputptr, int BLOCK_SIZE) {
  int64_t *ptrs[2];
  const uint64_t logBSZ = asm_log2(BLOCK_SIZE);

  ptrs[0] = *inputptr;
  ptrs[1] = *outputptr;

  /** 1.a) Perform in-register sort to get sorted seq of K(K=4)*/
  block16 *inptr = (block16 *)ptrs[0];
  block16 *const end = (block16 *)(ptrs[0] + BLOCK_SIZE);
  while (inptr < end) {
    inregister_sort_keyval32_aligned((int64_t *)inptr, (int64_t *)inptr);
    inptr++;
  }

  /**
   * 1.b) for itr <- [(logK) .. (logM - 3)]
   *  - Simultaneously merge 4 sequences (using a K by K
   *  network) of length 2^itr to obtain sorted seq. of 2^{itr+1}
   */
  uint64_t j;
  const uint64_t jend = logBSZ - 2;

  j = 2;
  {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge4_eqlen_aligned(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;
    }
  }

  j = 3;
  {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge8_eqlen_aligned(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;
    }
  }
  for (j = 4; j < jend; j++) {
    int ptridx = j & 1;
    int64_t *inp = (int64_t *)ptrs[ptridx];
    int64_t *out = (int64_t *)ptrs[ptridx ^ 1];
    int64_t *const end = (int64_t *)(inp + BLOCK_SIZE);

    /**
     *  merge length 2^j lists beginnig at inp and output a
     *  sorted list of length 2^(j+1) starting at out
     */
    const uint64_t inlen = (1 << j);
    const uint64_t outlen = (inlen << 1);

    while (inp < end) {
      merge16_eqlen_aligned(inp, inp + inlen, out, inlen);
      inp += outlen;
      out += outlen;

      /* TODO: Try following. */
      /* simultaneous merge of 4 list pairs */
      /* merge 4 seqs simultaneously (always >= 4) */
      /* merge 2 seqs simultaneously (always >= 2) */
    }
  }

  /**
   * 1.c) for itr = (logM - 2), simultaneously merge 2 sequences
   *  (using a 2K by 2K network) of length M/4 to obtain sorted
   *  sequences of M/2.
   */
  uint64_t inlen = (1 << j);
  int64_t *inp;
  int64_t *out;
  int ptridx = j & 1;

  inp = ptrs[ptridx];
  out = ptrs[ptridx ^ 1];

  merge16_eqlen_aligned(inp, inp + inlen, out, inlen);
  merge16_eqlen_aligned(inp + 2 * inlen, inp + 3 * inlen, out + 2 * inlen,
                        inlen);

  /* TODO: simultaneous merge of 2 list pairs */
  /**
   * 1.d) for itr = (logM - 1), merge 2 final sequences (using a
   * 4K by 4K network) of length M/2 to get sorted seq. of M.
   */
  j++; /* j=(LOG2_BLOCK_SIZE-1); inputsize M/2 --> outputsize M*/
  inlen = (1 << j);
  /* now we know that input is out from the last pass */
  merge16_eqlen_aligned(out, out + inlen, inp, inlen);

  /* finally swap input/output ptrs, output is the sorted list */
  *outputptr = inp;
  *inputptr = out;
}

inline void __attribute__((always_inline, target("avx2")))
merge16_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                      int64_t *const out, const uint32_t len) {
  register block16 *inA = (block16 *)inpA;
  register block16 *inB = (block16 *)inpB;
  block16 *const endA = (block16 *)(inpA + len);
  block16 *const endB = (block16 *)(inpB + len);

  block16 *outp = (block16 *)out;

  register block16 *next = inB;

  __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
  __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

  __m256d regAl1, regAl2, regAh1, regAh2;
  __m256d regBl1, regBl2, regBh1, regBh2;

  LOAD8(regAl1, regAl2, inA);
  LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
  inA++;

  LOAD8(regBl1, regBl2, inB);
  LOAD8(regBh1, regBh2, ((block8 *)(inB) + 1));
  inB++;

  BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                  outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                  regAh2, regBl1, regBl2, regBh1, regBh2);

  /* store outreg1 */
  STORE8(outp, outreg1l1, outreg1l2);
  STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
  outp++;

  while (inA < endA && inB < endB) {
    /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
    IFELSECONDMOVE(next, inA, inB, 128);

    regAl1 = outreg2l1;
    regAl2 = outreg2l2;
    regAh1 = outreg2h1;
    regAh2 = outreg2h2;

    LOAD8(regBl1, regBl2, next);
    LOAD8(regBh1, regBh2, ((block8 *)next + 1));

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8(outp, outreg1l1, outreg1l2);
    STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1 = outreg2l1;
    __m256d regBl2 = outreg2l2;
    __m256d regBh1 = outreg2h1;
    __m256d regBh2 = outreg2h2;

    LOAD8(regAl1, regAl2, inA);
    LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8(outp, outreg1l1, outreg1l2);
    STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  while (inB < endB) {
    __m256d regBl1, regBl2, regBh1, regBh2;
    __m256d regAl1 = outreg2l1;
    __m256d regAl2 = outreg2l2;
    __m256d regAh1 = outreg2h1;
    __m256d regAh2 = outreg2h2;

    LOAD8(regBl1, regBl2, inB);
    LOAD8(regBh1, regBh2, ((block8 *)inB + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8(outp, outreg1l1, outreg1l2);
    STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  /* store the last remaining register values */
  STORE8(outp, outreg2l1, outreg2l2);
  STORE8(((block8 *)outp + 1), outreg2h1, outreg2h2);
}

inline void __attribute__((always_inline, target("avx2")))
merge8_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                     int64_t *const out, const uint32_t len) {
  register block8 *inA = (block8 *)inpA;
  register block8 *inB = (block8 *)inpB;
  block8 *const endA = (block8 *)(inpA + len);
  block8 *const endB = (block8 *)(inpB + len);

  block8 *outp = (block8 *)out;

  register block8 *next = inB;

  register __m256d outreg1l, outreg1h;
  register __m256d outreg2l, outreg2h;

  register __m256d regAl, regAh;
  register __m256d regBl, regBh;

  LOAD8(regAl, regAh, inA);
  LOAD8(regBl, regBh, next);

  inA++;
  inB++;

  BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                 regBh);

  /* store outreg1 */
  STORE8(outp, outreg1l, outreg1h);
  outp++;

  while (inA < endA && inB < endB) {
    /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
    IFELSECONDMOVE(next, inA, inB, 64);

    regAl = outreg2l;
    regAh = outreg2h;
    LOAD8(regBl, regBh, next);

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l, outreg1h);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regAl, regAh;
    LOAD8(regAl, regAh, inA);

    __m256d regBl = outreg2l;
    __m256d regBh = outreg2h;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l, outreg1h);
    outp++;
    inA++;
  }

  while (inB < endB) {
    __m256d regAl = outreg2l;
    __m256d regAh = outreg2h;
    __m256d regBl, regBh;

    LOAD8(regBl, regBh, inB);

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8(outp, outreg1l, outreg1h);
    outp++;
    inB++;
  }

  /* store the last remaining register values */
  STORE8(outp, outreg2l, outreg2h);
}

inline void __attribute__((always_inline, target("avx2")))
merge4_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
             const uint32_t len) {
  register block4 *inA = (block4 *)inpA;
  register block4 *inB = (block4 *)inpB;
  block4 *const endA = (block4 *)(inpA + len);
  block4 *const endB = (block4 *)(inpB + len);

  block4 *outp = (block4 *)out;

  register block4 *next = inB;

  register __m256d outreg1;
  register __m256d outreg2;

  register __m256d regA = _mm256_loadu_pd((double const *)inA);
  register __m256d regB = _mm256_loadu_pd((double const *)next);

  inA++;
  inB++;

  BITONIC_MERGE4(outreg1, outreg2, regA, regB);

  /* store outreg1 */
  _mm256_storeu_pd((double *)outp, outreg1);
  outp++;

  while (inA < endA && inB < endB) {
    /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
    IFELSECONDMOVE(next, inA, inB, 32);
    regA = outreg2;
    regB = _mm256_loadu_pd((double const *)next);
    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
    /* store outreg1 */
    _mm256_storeu_pd((double *)outp, outreg1);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regA = _mm256_loadu_pd((double const *)inA);
    __m256d regB = outreg2;
    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
    _mm256_storeu_pd((double *)outp, outreg1);
    inA++;
    outp++;
  }

  while (inB < endB) {
    __m256d regA = outreg2;
    __m256d regB = _mm256_loadu_pd((double const *)inB);
    BITONIC_MERGE4(outreg1, outreg2, regA, regB);
    _mm256_storeu_pd((double *)outp, outreg1);
    inB++;
    outp++;
  }

  /* store the last remaining register values */
  _mm256_storeu_pd((double *)outp, outreg2);
}

inline void __attribute__((always_inline, target("avx2")))
inregister_sort_keyval32_aligned(int64_t *items, int64_t *output) {
  /* IACA_START */
  __m256d ra = _mm256_load_pd((double const *)(items));
  __m256d rb = _mm256_load_pd((double const *)(items + 4));
  __m256d rc = _mm256_load_pd((double const *)(items + 8));
  __m256d rd = _mm256_load_pd((double const *)(items + 12));

  /* odd-even sorting network begins */
  /* 1st level of comparisons */
  __m256d ra1 = _mm256_min_pd(ra, rb);
  __m256d rb1 = _mm256_max_pd(ra, rb);

  __m256d rc1 = _mm256_min_pd(rc, rd);
  __m256d rd1 = _mm256_max_pd(rc, rd);

  /* 2nd level of comparisons */
  rb = _mm256_min_pd(rb1, rd1);
  rd = _mm256_max_pd(rb1, rd1);

  /* 3rd level of comparisons */
  __m256d ra2 = _mm256_min_pd(ra1, rc1);
  __m256d rc2 = _mm256_max_pd(ra1, rc1);

  /* 4th level of comparisons */
  __m256d rb3 = _mm256_min_pd(rb, rc2);
  __m256d rc3 = _mm256_max_pd(rb, rc2);

  /* results are in ra2, rb3, rc3, rd */
  /**
   * Initial data and transposed data looks like following:
   *  a2={ x1  x2  x3  x4  }                      a4={ x1 x5 x9  x13 }
   *  b3={ x5  x6  x7  x8  }  === Transpose ===>  b5={ x2 x6 x10 x14 }
   *  c3={ x9  x10 x11 x12 }                      c5={ x3 x7 x11 x15 }
   *  d2={ x13 x14 x15 x16 }                      d4={ x4 x8 x12 x16 }
   */
  /* shuffle x2 and x5 - shuffle x4 and x7 */
  __m256d ra3 = _mm256_unpacklo_pd(ra2, rb3);
  __m256d rb4 = _mm256_unpackhi_pd(ra2, rb3);

  /* shuffle x10 and x13 - shuffle x12 and x15 */
  __m256d rc4 = _mm256_unpacklo_pd(rc3, rd);
  __m256d rd3 = _mm256_unpackhi_pd(rc3, rd);

  /* shuffle (x3,x7) and (x9,x13) pairs */
  __m256d ra4 = _mm256_permute2f128_pd(ra3, rc4, 0x20);
  __m256d rc5 = _mm256_permute2f128_pd(ra3, rc4, 0x31);

  /* shuffle (x4,x8) and (x10,x14) pairs */
  __m256d rb5 = _mm256_permute2f128_pd(rb4, rd3, 0x20);
  __m256d rd4 = _mm256_permute2f128_pd(rb4, rd3, 0x31);

  /* after this, results are in ra4, rb5, rc5, rd4 */
  /* IACA_END */
  /* store */
  _mm256_store_pd((double *)output, ra4);
  _mm256_store_pd((double *)(output + 4), rb5);
  _mm256_store_pd((double *)(output + 8), rc5);
  _mm256_store_pd((double *)(output + 12), rd4);
}

inline void __attribute__((always_inline, target("avx2")))
merge4_eqlen_aligned(int64_t *const inpA, int64_t *const inpB,
                     int64_t *const out, const uint32_t len) {
  register block4 *inA = (block4 *)inpA;
  register block4 *inB = (block4 *)inpB;
  block4 *const endA = (block4 *)(inpA + len);
  block4 *const endB = (block4 *)(inpB + len);
  block4 *outp = (block4 *)out;
  register block4 *next = inB;

  register __m256d outreg1;
  register __m256d outreg2;
  register __m256d regA = _mm256_load_pd((double const *)inA);
  register __m256d regB = _mm256_load_pd((double const *)next);

  inA++;
  inB++;

  BITONIC_MERGE4(outreg1, outreg2, regA, regB);

  /* store outreg1 */
  _mm256_store_pd((double *)outp, outreg1);
  outp++;

  while (inA < endA && inB < endB) {
    /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
    IFELSECONDMOVE(next, inA, inB, 32);

    regA = outreg2;
    regB = _mm256_load_pd((double const *)next);

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    /* store outreg1 */
    _mm256_store_pd((double *)outp, outreg1);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regA = _mm256_load_pd((double const *)inA);
    __m256d regB = outreg2;

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    _mm256_store_pd((double *)outp, outreg1);
    inA++;
    outp++;
  }

  while (inB < endB) {
    __m256d regA = outreg2;
    __m256d regB = _mm256_load_pd((double const *)inB);

    BITONIC_MERGE4(outreg1, outreg2, regA, regB);

    _mm256_store_pd((double *)outp, outreg1);
    inB++;
    outp++;
  }

  /* store the last remaining register values */
  _mm256_store_pd((double *)outp, outreg2);
}

inline void __attribute__((always_inline, target("avx2")))
merge16_varlen_aligned(int64_t *inpA, int64_t *inpB, int64_t *Out,
                       const uint32_t lenA, const uint32_t lenB) {
  uint32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;
  uint32_t ai = 0, bi = 0;
  int64_t *out = Out;
  if (lenA16 > 16 && lenB16 > 16) {
    register block16 *inA = (block16 *)inpA;
    register block16 *inB = (block16 *)inpB;
    block16 *const endA = (block16 *)(inpA + lenA) - 1;
    block16 *const endB = (block16 *)(inpB + lenB) - 1;

    block16 *outp = (block16 *)out;

    register block16 *next = inB;

    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8(regAl1, regAl2, inA);
    LOAD8(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    LOAD8(regBl1, regBl2, inB);
    LOAD8(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8(outp, outreg1l1, outreg1l2);
    STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;

    while (inA < endA && inB < endB) {
      /** The inline assembly below does exactly the following code: */
      /* Option 3: with assembly */
      IFELSECONDMOVE(next, inA, inB, 128);

      regAl1 = outreg2l1;
      regAl2 = outreg2l2;
      regAh1 = outreg2h1;
      regAh2 = outreg2h2;

      LOAD8(regBl1, regBl2, next);
      LOAD8(regBh1, regBh2, ((block8 *)next + 1));

      BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                      outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                      regAh2, regBl1, regBl2, regBh1, regBh2);

      /* store outreg1 */
      STORE8(outp, outreg1l1, outreg1l2);
      STORE8(((block8 *)outp + 1), outreg1h1, outreg1h2);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(16)));
    _mm256_store_pd((double *)hireg, outreg2h2);

    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      STORE8(inA, outreg2l1, outreg2l2);
      STORE8(((block8 *)inA + 1), outreg2h1, outreg2h2);
    } else {
      /* store the last remaining register values to B */
      inB--;
      STORE8(inB, outreg2l1, outreg2l2);
      STORE8(((block8 *)inB + 1), outreg2h1, outreg2h2);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8 = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8 = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}

inline void __attribute__((always_inline, target("avx2")))
merge8_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
             const uint32_t len) {
  register block8 *inA = (block8 *)inpA;
  register block8 *inB = (block8 *)inpB;
  block8 *const endA = (block8 *)(inpA + len);
  block8 *const endB = (block8 *)(inpB + len);

  block8 *outp = (block8 *)out;

  register block8 *next = inB;

  register __m256d outreg1l, outreg1h;
  register __m256d outreg2l, outreg2h;

  register __m256d regAl, regAh;
  register __m256d regBl, regBh;

  LOAD8U(regAl, regAh, inA);
  LOAD8U(regBl, regBh, next);

  inA++;
  inB++;

  BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                 regBh);

  /* store outreg1 */
  STORE8U(outp, outreg1l, outreg1h);
  outp++;

  while (inA < endA && inB < endB) {
    /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
    IFELSECONDMOVE(next, inA, inB, 64);

    regAl = outreg2l;
    regAh = outreg2h;
    LOAD8U(regBl, regBh, next);

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l, outreg1h);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regAl, regAh;
    LOAD8U(regAl, regAh, inA);

    __m256d regBl = outreg2l;
    __m256d regBh = outreg2h;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l, outreg1h);
    outp++;
    inA++;
  }

  while (inB < endB) {
    __m256d regAl = outreg2l;
    __m256d regAh = outreg2h;
    __m256d regBl, regBh;

    LOAD8U(regBl, regBh, inB);

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l, outreg1h);
    outp++;
    inB++;
  }

  /* store the last remaining register values */
  STORE8U(outp, outreg2l, outreg2h);
}

inline void __attribute__((always_inline, target("avx2")))
merge16_eqlen(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
              const uint32_t len) {
  register block16 *inA = (block16 *)inpA;
  register block16 *inB = (block16 *)inpB;
  block16 *const endA = (block16 *)(inpA + len);
  block16 *const endB = (block16 *)(inpB + len);

  block16 *outp = (block16 *)out;

  register block16 *next = inB;

  __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
  __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

  __m256d regAl1, regAl2, regAh1, regAh2;
  __m256d regBl1, regBl2, regBh1, regBh2;

  LOAD8U(regAl1, regAl2, inA);
  LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
  inA++;

  LOAD8U(regBl1, regBl2, inB);
  LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
  inB++;

  BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                  outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                  regAh2, regBl1, regBl2, regBh1, regBh2);

  /* store outreg1 */
  STORE8U(outp, outreg1l1, outreg1l2);
  STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
  outp++;

  while (inA < endA && inB < endB) {
    /** The inline assembly below does exactly the following code: */
    /* Option 3: with assembly */
    IFELSECONDMOVE(next, inA, inB, 128);

    regAl1 = outreg2l1;
    regAl2 = outreg2l2;
    regAh1 = outreg2h1;
    regAh2 = outreg2h2;

    LOAD8U(regBl1, regBl2, next);
    LOAD8U(regBh1, regBh2, ((block8 *)next + 1));

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  /* handle remaining items */
  while (inA < endA) {
    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1 = outreg2l1;
    __m256d regBl2 = outreg2l2;
    __m256d regBh1 = outreg2h1;
    __m256d regBh2 = outreg2h2;

    LOAD8U(regAl1, regAl2, inA);
    LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  while (inB < endB) {
    __m256d regBl1, regBl2, regBh1, regBh2;
    __m256d regAl1 = outreg2l1;
    __m256d regAl2 = outreg2l2;
    __m256d regAh1 = outreg2h1;
    __m256d regAh2 = outreg2h2;

    LOAD8U(regBl1, regBl2, inB);
    LOAD8U(regBh1, regBh2, ((block8 *)inB + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;
  }

  /* store the last remaining register values */
  STORE8U(outp, outreg2l1, outreg2l2);
  STORE8U(((block8 *)outp + 1), outreg2h1, outreg2h2);
}

void __attribute__((target("avx2")))
avxsortmultiway_tuples(tuple_t **inputptr, tuple_t **outputptr,
                       uint64_t nitems) {
  int64_t *inp = (int64_t *)*inputptr;
  int64_t *out = (int64_t *)*outputptr;
  avxsortmultiway_int64(&inp, &out, nitems);
  *inputptr = (tuple_t *)inp;
  *outputptr = (tuple_t *)out;
}

void __attribute__((target("avx2")))
avxsortmultiway_int64(int64_t **inputptr, int64_t **outputptr,
                      uint64_t nitems) {
  int64_t *input = (int64_t *)(*inputptr);
  int64_t *output = (int64_t *)(*outputptr);

  uint64_t i;
  uint64_t nblocks = nitems / L3BLOCKSIZE;
  /* each chunk keeps track of its temporary memory offset */
  int64_t *ptrs[nblocks + 1][2]; /* [block-in, block-out-tmp] */
  uint32_t sizes[nblocks + 1];

  uint64_t remsize = (nitems % L3BLOCKSIZE);
  for (i = 0; i < nblocks; i++) {
    ptrs[i][0] = input + i * L3BLOCKSIZE;
    ptrs[i][1] = output + i * L3BLOCKSIZE;
    sizes[i] = L3BLOCKSIZE;
  }

  /** 1) Divide the input into chunks fitting into L3 cache. */
  for (i = 0; i < nblocks; i++) {
    avxsort_int64(&ptrs[i][0], &ptrs[i][1], L3BLOCKSIZE);
    swap(&ptrs[i][0], &ptrs[i][1]);
  }

  /* one more chunk if not divisible */
  if (remsize) {
    ptrs[i][0] = input + i * L3BLOCKSIZE;
    ptrs[i][1] = output + i * L3BLOCKSIZE;
    sizes[i] = remsize;
    /* sort the last chunk which is less than BLOCKSIZE */
    avxsort_int64(&ptrs[i][0], &ptrs[i][1], remsize);
    swap(&ptrs[i][0], &ptrs[i][1]);
    nblocks++;
  }

  /** 2) Apply a multi-way merge if number of blocks are > 3 */
  if (nblocks > 3) {
    /* multi-way merge */
    if ((nblocks % 2) == 1) {
      /* merge the remainder chunk into the last one */
      int nc1 = nblocks - 2;
      int nc2 = nblocks - 1;
      int64_t *inpA = ptrs[nc1][0];
      int64_t *inpB = ptrs[nc2][0];
      int64_t *out = ptrs[nc1][1];
      uint32_t sizeA = sizes[nc1];
      uint32_t sizeB = sizes[nc2];

      /* merge16_varlen(inpA, inpB, out, sizeA, sizeB); */
      avx_merge_int64(inpA, inpB, out, sizeA, sizeB);

      /* setup new pointers */
      ptrs[nc1][0] = out;
      ptrs[nc1][1] = inpA;
      sizes[nc1] = sizeA + sizeB;
      nblocks--;
    }

    /* now setup a multi-way merge. */
    /*
     * IMPORTANT NOTE: nblocks must be padded to pow2! If nblocks is not a power
     * of 2, then we introduce additional blocks with 0-tuples to achieve a pow2
     * multi-way merging.
     */
    uint64_t nblockspow2 = 1 << (int)(ceil(log2(nblocks)));
    /* printf("Merge nblocks = %"PRId64" -- nblocks(pow2) = %"PRId64"\n",
            nblocks, nblockspow2); */
    relation_t rels[nblockspow2];
    relation_t *chunks[nblockspow2];

    for (i = 0; i < nblocks; i++) {
      rels[i].tuples = (tuple_t *)ptrs[i][0];
      rels[i].num_tuples = sizes[i];
      chunks[i] = &rels[i];
    }
    for (; i < nblockspow2; i++) {
      rels[i].tuples = 0;
      rels[i].num_tuples = 0;
      chunks[i] = &rels[i];
    }

    uint32_t bufntuples = (L3_CACHE_SIZE) / sizeof(tuple_t);
    tuple_t *outptr = (tuple_t *)ptrs[0][1];

    tuple_t *fifobuffer = (tuple_t *)malloc_aligned(L3_CACHE_SIZE);

    avx_multiway_merge(outptr, chunks, nblockspow2, fifobuffer, bufntuples);
    free(fifobuffer);
    /* finally swap input/output pointers, where output holds the sorted list */
    *outputptr = (int64_t *)outptr;
    *inputptr = (int64_t *)ptrs[0][0];
  } else {
    /* apply 2-way merge */
    const uint64_t logN = ceil(log2(nitems));
    for (i = log2(L3BLOCKSIZE); i < logN; i++) {
      uint64_t k = 0;
      for (uint64_t j = 0; j < (nblocks - 1); j += 2) {
        int64_t *inpA = ptrs[j][0];
        int64_t *inpB = ptrs[j + 1][0];
        int64_t *out = ptrs[j][1];
        uint32_t sizeA = sizes[j];
        uint32_t sizeB = sizes[j + 1];

        avx_merge_int64(inpA, inpB, out, sizeA, sizeB);

        /* setup new pointers */
        ptrs[k][0] = out;
        ptrs[k][1] = inpA;
        sizes[k] = sizeA + sizeB;
        k++;
      }

      if ((nblocks % 2)) {
        /* just move the pointers */
        ptrs[k][0] = ptrs[nblocks - 1][0];
        ptrs[k][1] = ptrs[nblocks - 1][1];
        sizes[k] = sizes[nblocks - 1];
        k++;
      }

      nblocks = k;
    }

    /* finally swap input/output pointers, where output holds the sorted list */
    *outputptr = ptrs[0][0];
    *inputptr = ptrs[0][1];
  }
}

uint64_t __attribute__((target("avx2")))
avx_merge_int64(int64_t *const inpA, int64_t *const inpB, int64_t *const out,
                const uint64_t lenA, const uint64_t lenB) {
  int isaligned = 0, iseqlen = 0;

  /* is-aligned ? */
  isaligned = (((uintptr_t)inpA % CACHE_LINE_SIZE) == 0) &&
              (((uintptr_t)inpB % CACHE_LINE_SIZE) == 0) &&
              (((uintptr_t)out % CACHE_LINE_SIZE) == 0);

  /* is equal length? */
  /* iseqlen = (lenA == lenB); */
  /* TODO: There is a problem when using merge-eqlen variants, because the
  merge routine does not consider that other lists begin where one list ends
  and might be overwriting a few tuples. */
  if (iseqlen) {
    if (isaligned) {
#if (MERGEBITONICWIDTH == 4)
      merge4_eqlen_aligned(inpA, inpB, out, lenA);
#elif(MERGEBITONICWIDTH == 8)
      merge8_eqlen_aligned(inpA, inpB, out, lenA);
#elif(MERGEBITONICWIDTH == 16)
      merge16_eqlen_aligned(inpA, inpB, out, lenA);
#endif
    } else {
#if (MERGEBITONICWIDTH == 4)
      merge4_eqlen(inpA, inpB, out, lenA);
#elif(MERGEBITONICWIDTH == 8)
      merge8_eqlen(inpA, inpB, out, lenA);
#elif(MERGEBITONICWIDTH == 16)
      merge16_eqlen(inpA, inpB, out, lenA);
#endif
    }
  } else {
    if (isaligned) {
#if (MERGEBITONICWIDTH == 4)
      merge4_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif(MERGEBITONICWIDTH == 8)
      merge8_varlen_aligned(inpA, inpB, out, lenA, lenB);
#elif(MERGEBITONICWIDTH == 16)
      merge16_varlen_aligned(inpA, inpB, out, lenA, lenB);
#endif
    } else {
#if (MERGEBITONICWIDTH == 4)
      merge4_varlen(inpA, inpB, out, lenA, lenB);
#elif(MERGEBITONICWIDTH == 8)
      merge8_varlen(inpA, inpB, out, lenA, lenB);
#elif(MERGEBITONICWIDTH == 16)
      merge16_varlen(inpA, inpB, out, lenA, lenB);
#endif
    }
  }

  return (lenA + lenB);
}

static void *malloc_aligned(size_t size) {
  void *ret;
  int rv = posix_memalign((void **)&ret, CACHE_LINE_SIZE, size);
  if (rv) {
    perror("[ERROR] malloc_aligned() failed: out of memory");
    return 0;
  }
  return ret;
}

uint64_t __attribute__((target("avx2")))
avx_multiway_merge(tuple_t *output, relation_t **parts, uint32_t nparts,
                   tuple_t *fifobuffer, uint32_t bufntuples) {
  uint64_t totalmerged = 0;
  uint32_t nfifos = nparts - 2;
  uint32_t totalfifosize =
      bufntuples - nparts -
      (nfifos * sizeof(merge_node_t) + nfifos * sizeof(uint8_t) +
       nparts * sizeof(relation_t) + sizeof(tuple_t) - 1) /
          sizeof(tuple_t);

  uint32_t fifosize = totalfifosize / nfifos;
  /* align ring-buffer size to be multiple of 64-Bytes */
  /* fifosize = ALIGNDOWN(fifosize); */
  /* totalfifosize = fifosize * nfifos; */

  merge_node_t *nodes = (merge_node_t *)(fifobuffer + totalfifosize);
  uint8_t *done = (uint8_t *)(nodes + nfifos);

  /* printf("[INFO ] fifosize = %d, totalfifosize = %d tuples, %.2lf KiB\n", */
  /*        fifosize, totalfifosize, totalfifosize*sizeof(tuple_t)/1024.0); */

  for (uint32_t i = 0; i < nfifos; i++) {
    nodes[i].buffer = fifobuffer + fifosize * i;
    nodes[i].count = 0;
    nodes[i].head = 0;
    nodes[i].tail = 0;
    done[i] = 0;
  }

  uint32_t finished = 0;
  const uint32_t readthreshold = fifosize / 2;

  while (!finished) {
    finished = 1;
    int m = nfifos - 1;

    /* first iterate through leafs and read as much data as possible */
    for (uint32_t c = 0; c < nparts; c += 2, m--) {
      if (!done[m] && (nodes[m].count < readthreshold)) {
        uint32_t A = c;
        uint32_t B = c + 1;
        tuple_t *inA = parts[A]->tuples;
        tuple_t *inB = parts[B]->tuples;

        uint32_t nread;

        /*
        if(!check_merge_node_sorted(&nodes[m], fifosize)){
             printf("before read:: Node not sorted\n");
             exit(0);
        }*/

        nread = readmerge_parallel_decomposed(&nodes[m], &inA, &inB,
                                              parts[A]->num_tuples,
                                              parts[B]->num_tuples, fifosize);

        /*
        if(!check_merge_node_sorted(&nodes[m], fifosize)){
             printf("after read:: Node not sorted\n");
             exit(0);
        }*/

        parts[A]->num_tuples -= (inA - parts[A]->tuples);
        parts[B]->num_tuples -= (inB - parts[B]->tuples);

        parts[A]->tuples = inA;
        parts[B]->tuples = inB;

        done[m] = (nread == 0 || ((parts[A]->num_tuples == 0) &&
                                  (parts[B]->num_tuples == 0)));

        finished &= done[m];
      }
    }

    /* now iterate inner nodes and do merge for ready nodes */
    for (; m >= 0; m--) {
      if (!done[m]) {
        int r = 2 * m + 2;
        int l = r + 1;
        merge_node_t *right = &nodes[r];
        merge_node_t *left = &nodes[l];

        uint8_t children_done = (done[r] | done[l]);

        if ((children_done || nodes[m].count < readthreshold) &&
            nodes[m].count < fifosize) {
          if (children_done ||
              (right->count >= readthreshold && left->count >= readthreshold)) {
            /* if(!check_node_sorted(right, fifosize)) */
            /*     printf("Right Node not sorted\n"); */

            /* if(!check_node_sorted(left, fifosize)) */
            /*     printf("Left Node not sorted\n"); */

            /* do a merge on right and left */
            /* TODO: FIXME: "(done[r] & done[l])" doesn't work for fan-in of 128
             */
            /* TODO: FIXME: "children_done" doesn't work for fan-in of 8 */
            merge_parallel_decomposed(
                &nodes[m], right, left, fifosize, done[r],
                done[l]  // children_done /* full-merge? */
                );

            /* if(!check_node_sorted(right, fifosize)) */
            /*     printf("After merge- Right Node not sorted\n"); */

            /* if(!check_node_sorted(left, fifosize)) */
            /*     printf("After merge- Left Node not sorted\n"); */

            /*
            if(!check_merge_node_sorted(&nodes[m], fifosize))
                 printf("After merge - Node not sorted\n");*/
          }
          done[m] =
              (done[r] & done[l]) && (right->count == 0 && left->count == 0);
        }

        finished &= done[m];
      }
    }

    totalmerged +=
        /* finally iterate for the root node and store data */
        mergestore_parallel_decomposed(&nodes[0], &nodes[1], &output, fifosize,
                                       done[0], done[1] /* full-merge? */
                                       );
  }

  /* free(fifobuffer); */
  return totalmerged;
}

uint64_t __attribute__((target("avx2")))
mergestore_parallel_decomposed(merge_node_t *right, merge_node_t *left,
                               tuple_t **output, uint32_t fifosize,
                               uint8_t rightdone, uint8_t leftdone) {
  /* directly copy tuples from right or left if one of them done but not the
   * other */
  if (rightdone && right->count == 0) {
    if (!leftdone && left->count > 0) {
      uint64_t numcopied = direct_copy_to_output_avx(*output, left, fifosize);
      /*
      if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
          printf("[ERROR] 1.\n");
      }
      */
      *output += numcopied;
      return numcopied;
    }
  } else if (leftdone && left->count == 0) {
    if (!rightdone && right->count > 0) {
      uint64_t numcopied = direct_copy_to_output_avx(*output, right, fifosize);
      /*
      if(is_sorted_helper((int64_t*)(*output), numcopied) == 0){
          printf("[ERROR] 2.\n");
      }
      */
      *output += numcopied;
      return numcopied;
    }
  }

  tuple_t *Out = *output;
  int rcases = 0, lcases = 0;

  uint32_t ri = right->head, rend;
  if (right->head >= right->tail) {
    rend = fifosize;
    rcases = 1;
  } else {
    rend = right->tail;
  }

  uint32_t li = left->head, lend;
  if (left->head >= left->tail) {
    lend = fifosize;
    lcases = 1;
  } else {
    lend = left->tail;
  }

  while (right->count > 0 && left->count > 0) {
    register tuple_t *R = right->buffer + ri;
    register tuple_t *L = left->buffer + li;

    /* serialmergestorekernel(R, L, &Out, &ri, &li, rend, lend); */
    mergestore16kernel(R, L, &Out, &ri, &li, rend, lend);

    right->count -= (ri - right->head);
    right->head = ((ri == fifosize) ? 0 : ri);
    left->count -= (li - left->head);
    left->head = ((li == fifosize) ? 0 : li);

    if (rcases > 0 && ri == rend) {
      ri = 0;
      rend = right->tail;
      rcases = 0;
    }

    if (lcases > 0 && li == lend) {
      li = 0;
      lend = left->tail;
      lcases = 0;
    }
  }

  /* not possible until we do not read new tuples anymore */
  uint8_t done = rightdone & leftdone;
  if (done) {
    if (right->count > 0) {
      tuple_t *R = right->buffer + right->head;

      ri = right->head;
      rend = right->head + right->count;
      if (rend > fifosize) rend = fifosize;

      do {
        // uint32_t sz = rend-ri;
        // memcpy((void*)Out, (void*)R, sz*sizeof(tuple_t));
        // Out += sz;
        // R += sz;
        // right->count -= sz;
        // right->head  += sz;
        // ri = rend;
        while (ri < rend) {
          ri++;
          *Out = *R;
          Out++;
          R++;
          right->count--;
          right->head++;
        }

        if (rcases > 0 && ri == rend) {
          ri = 0;
          rend = right->tail;
          rcases = 0;
          if (right->head >= fifosize) {
            right->head = 0;
            R = right->buffer;
          }
        }
      } while (right->count > 0);

    } else if (left->count > 0) {
      tuple_t *L = left->buffer + left->head;

      li = left->head;
      lend = left->head + left->count;
      if (lend > fifosize) lend = fifosize;

      do {
        // uint32_t sz = lend-li;
        // memcpy((void*)Out, (void*)L, sz*sizeof(tuple_t));
        // Out += sz;
        // L += sz;
        // left->count -= sz;
        // left->head  += sz;
        // li = lend;
        while (li < lend) {
          li++;
          *Out = *L;
          Out++;
          L++;
          left->count--;
          left->head++;
        }

        if (lcases > 0 && li == lend) {
          li = 0;
          lend = left->tail;
          lcases = 0;
          if (left->head >= fifosize) {
            left->head = 0;
            L = left->buffer;
          }
        }

      } while (left->count > 0);
    }
  }

  uint64_t numstored = (Out - *output);
  /*
  if(is_sorted_helper((int64_t*)(*output), numstored) == 0){
      printf("[ERROR] rightdone=%d leftdone=%d\n", rightdone, leftdone);
  }
  */
  *output = Out;
  return numstored;
}

uint32_t __attribute__((target("avx2")))
readmerge_parallel_decomposed(merge_node_t *node, tuple_t **inA, tuple_t **inB,
                              uint32_t lenA, uint32_t lenB, uint32_t fifosize) {
  uint32_t nodecount = node->count;
  uint32_t nodehead = node->head;
  uint32_t nodetail = node->tail;
  tuple_t *Out = node->buffer;

  tuple_t *A = *inA;
  tuple_t *B = *inB;

  /* size related variables */
  uint32_t ri = 0, li = 0, outnslots;

  uint32_t oi = nodetail, oend;
  uint32_t oi2 = 0, oend2 = 0;

  if (nodehead > nodetail) {
    oend = nodehead;
  } else {
    oend = fifosize;
    oi2 = 0;
    oend2 = nodehead;
  }
  outnslots = oend - oi;

  Out += oi;

  /* fill first chunk of the node buffer */
  parallel_read(&A, &B, &Out, &ri, &li, &oi, &outnslots, lenA, lenB);

  nodecount += (oi - nodetail);
  nodetail = ((oi == fifosize) ? 0 : oi);

  if (outnslots == 0 && oend2 != 0) {
    outnslots = oend2 - oi2;
    Out = node->buffer;

    /* fill second chunk of the node buffer */
    parallel_read(&A, &B, &Out, &ri, &li, &oi2, &outnslots, lenA, lenB);

    nodecount += oi2;
    nodetail = ((oi2 == fifosize) ? 0 : oi2);
  }

  if (nodecount < fifosize) {
    outnslots = fifosize - nodecount;
    oi = nodetail;
    oend = (nodetail + outnslots);
    if (oend > fifosize) oend = fifosize;
    outnslots = oend - oi;

    if (ri < lenA) {
      do {
        while (outnslots > 0 && ri < lenA) {
          outnslots--;
          oi++;
          *Out = *A;
          ri++;
          A++;
          Out++;
          nodecount++;
          nodetail++;
        }

        if (oi == oend) {
          oi = 0;
          oend = nodehead;
          if (nodetail >= fifosize) {
            nodetail = 0;
            Out = node->buffer;
          }
          outnslots = oend - oi;
        }

      } while (nodecount < fifosize && ri < lenA);
    } else if (li < lenB) {
      do {
        while (outnslots > 0 && li < lenB) {
          outnslots--;
          oi++;
          *Out = *B;
          li++;
          B++;
          Out++;
          nodecount++;
          nodetail++;
        }

        if (oi == oend) {
          oi = 0;
          oend = nodehead;
          if (nodetail >= fifosize) {
            nodetail = 0;
            Out = node->buffer;
          }
          outnslots = oend - oi;
        }
      } while (nodecount < fifosize && li < lenB);
    }
  }
  *inA = A;
  *inB = B;

  node->tail = nodetail;
  node->count = nodecount;

  /* if(!check_node_sorted(node, fifosize)) */
  /*     printf("in merge_read() - Node not sorted\n"); */

  return (ri + li);
}

inline void __attribute__((always_inline, target("avx2")))
parallel_read(tuple_t **A, tuple_t **B, tuple_t **Out, uint32_t *ri,
              uint32_t *li, uint32_t *oi, uint32_t *outnslots, uint32_t lenA,
              uint32_t lenB) {
  uint32_t _ri = *ri, _li = *li, _oi = *oi;

  merge16kernel(*A, *B, *Out, ri, li, oi, outnslots, lenA, lenB);

  *A += (*ri - _ri);
  *B += (*li - _li);
  *Out += (*oi - _oi);
}

uint32_t __attribute__((target("avx2")))
direct_copy_to_output_avx(tuple_t *dest, merge_node_t *src, uint32_t fifosize) {
  /* make sure dest has space and src has tuples */
  // assert(src->count > 0);

  /* Cases for the ring-buffer : 1) head < tail 2) head > tail */
  uint32_t src_block_start[2];
  uint32_t src_block_size[2];

  if (src->head >= src->tail) { /* Case 2) */
    /* src block-1 */
    src_block_start[0] = src->head;
    src_block_size[0] = fifosize - src->head;
    /* src block-2 */
    src_block_start[1] = 0;
    src_block_size[1] = src->tail;
  } else {
    /* Case 1) src-> head < src->tail */
    /* src block-1 */
    src_block_start[0] = src->head;
    src_block_size[0] = src->tail - src->head;
    /* no block-2 */
    src_block_size[1] = 0;
  }

  uint32_t copied = 0;
  for (int j = 0; j < 2; j++) {
    uint32_t copysize = src_block_size[j];

    if (copysize > 0) {
      simd_memcpy((void *)(dest + copied), src->buffer + src_block_start[j],
                  copysize * sizeof(tuple_t));

      copied += copysize;
    }
  }

  src->count -= copied;
  src->head = (src->head + copied) % fifosize;

  return copied;
}

void __attribute__((target("avx2")))
direct_copy_avx(merge_node_t *dest, merge_node_t *src, uint32_t fifosize) {
  /* make sure dest has space and src has tuples */
  // assert(dest->count < fifosize);
  // assert(src->count > 0);

  /* Cases for the ring-buffer : 1) head < tail 2) head > tail */

  uint32_t dest_block_start[2];
  uint32_t dest_block_size[2];

  uint32_t src_block_start[2];
  uint32_t src_block_size[2];

  if (dest->head <= dest->tail) { /* Case 1) */
    /* dest block-1 */
    dest_block_start[0] = dest->tail;
    dest_block_size[0] = fifosize - dest->tail;
    /* dest block-2 */
    dest_block_start[1] = 0;
    dest_block_size[1] = dest->head;
  } else {
    /* Case 2) dest-> head > dest->tail */
    /* dest block-1 */
    dest_block_start[0] = dest->tail;
    dest_block_size[0] = dest->head - dest->tail;
    /* no block-2 */
    dest_block_size[1] = 0;
  }

  if (src->head >= src->tail) { /* Case 2) */
    /* src block-1 */
    src_block_start[0] = src->head;
    src_block_size[0] = fifosize - src->head;
    /* src block-2 */
    src_block_start[1] = 0;
    src_block_size[1] = src->tail;
  } else {
    /* Case 1) src-> head < src->tail */
    /* src block-1 */
    src_block_start[0] = src->head;
    src_block_size[0] = src->tail - src->head;
    /* no block-2 */
    src_block_size[1] = 0;
  }

  uint32_t copied = 0;
  for (int i = 0, j = 0; i < 2 && j < 2;) {
    uint32_t copysize = min(dest_block_size[i], src_block_size[j]);

    if (copysize > 0) {
      simd_memcpy(dest->buffer + dest_block_start[i],
                  src->buffer + src_block_start[j], copysize * sizeof(tuple_t));

      dest_block_start[i] += copysize;
      src_block_start[j] += copysize;
      dest_block_size[i] -= copysize;
      src_block_size[j] -= copysize;
      copied += copysize;
    }

    if (dest_block_size[i] == 0) i++;

    if (src_block_size[j] == 0) j++;
  }

  dest->count += copied;
  dest->tail = (dest->tail + copied) % fifosize;
  src->count -= copied;
  src->head = (src->head + copied) % fifosize;
}

void __attribute__((target("avx2")))
merge_parallel_decomposed(merge_node_t *node, merge_node_t *right,
                          merge_node_t *left, uint32_t fifosize,
                          uint8_t rightdone, uint8_t leftdone) {
  /* directly copy tuples from right or left if one of them done but not the
   * other */
  if (rightdone && right->count == 0) {
    if (!leftdone && left->count > 0) {
      direct_copy_avx(node, left, fifosize);
      return;
    }
  } else if (leftdone && left->count == 0) {
    if (!rightdone && right->count > 0) {
      direct_copy_avx(node, right, fifosize);
      return;
    }
  }

  /* both done? */
  uint8_t done = rightdone & leftdone;

  uint32_t righttail = right->tail;
  uint32_t rightcount = right->count;
  uint32_t righthead = right->head;
  uint32_t lefttail = left->tail;
  uint32_t leftcount = left->count;
  uint32_t lefthead = left->head;

  int rcases = 0, lcases = 0;
  uint32_t outnslots;

  uint32_t oi = node->tail, oend;
  if (node->head > node->tail) {
    oend = node->head;
  } else {
    oend = fifosize;
  }

  outnslots = oend - oi;

  uint32_t ri = righthead, rend;
  if (righthead >= righttail) {
    rend = fifosize;
    rcases = 1;
  } else {
    rend = righttail;
  }

  uint32_t li = lefthead, lend;
  if (lefthead >= lefttail) {
    lend = fifosize;
    lcases = 1;
  } else {
    lend = lefttail;
  }

  while (node->count < fifosize && (rightcount > 0 && leftcount > 0)) {
    register tuple_t *R = right->buffer + ri;
    register tuple_t *L = left->buffer + li;
    register tuple_t *Out = node->buffer + oi;

    /* serialmergekernel(R, L, Out, &ri, &li, &oi, &outnslots, rend, lend); */
    merge16kernel(R, L, Out, &ri, &li, &oi, &outnslots, rend, lend);

    node->count += (oi - node->tail);
    node->tail = ((oi == fifosize) ? 0 : oi);
    rightcount -= (ri - righthead);
    righthead = ((ri == fifosize) ? 0 : ri);
    leftcount -= (li - lefthead);
    lefthead = ((li == fifosize) ? 0 : li);

    if (oi == oend) {
      oi = 0;
      oend = node->head;
      outnslots = oend - oi;
    }

    if (rcases > 0 && ri == rend) {
      ri = 0;
      rend = righttail;
      rcases = 0;
    }

    if (lcases > 0 && li == lend) {
      li = 0;
      lend = lefttail;
      lcases = 0;
    }
  }

  /* not possible until we do not read new tuples anymore */
  if (done && node->count < fifosize) {
    tuple_t *Out = node->buffer + node->tail;

    outnslots = fifosize - node->count;
    oi = node->tail;
    oend = (node->tail + outnslots);
    if (oend > fifosize) oend = fifosize;

    outnslots = oend - oi;

    if (rightcount > 0) {
      tuple_t *R = right->buffer + righthead;

      ri = righthead;
      rend = righthead + rightcount;
      if (rend > fifosize) rend = fifosize;

      do {
        while (outnslots > 0 && ri < rend) {
          outnslots--;
          oi++;
          ri++;
          *Out = *R;
          Out++;
          R++;
          node->count++;
          rightcount--;
          node->tail++;
          righthead++;
        }

        /* node->count  += (oi - node->tail); */
        /* node->tail = ((oi == fifosize) ? 0 : oi); */
        /* rightcount -= (ri - righthead); */
        /* righthead = ((ri == fifosize) ? 0 : ri); */

        if (oi == oend) {
          oi = 0;
          oend = node->head;
          if (node->tail >= fifosize) {
            node->tail = 0;
            Out = node->buffer;
          }
        }

        if (rcases > 0 && ri == rend) {
          ri = 0;
          rend = righttail;
          rcases = 0;
          if (righthead >= fifosize) {
            righthead = 0;
            R = right->buffer;
          }
        }
      } while (outnslots > 0 && rightcount > 0);

    } else if (leftcount > 0) {
      tuple_t *L = left->buffer + lefthead;

      li = lefthead;
      lend = lefthead + leftcount;
      if (lend > fifosize) lend = fifosize;

      do {
        while (outnslots > 0 && li < lend) {
          outnslots--;
          oi++;
          li++;
          *Out = *L;
          Out++;
          L++;
          node->count++;
          leftcount--;
          node->tail++;
          lefthead++;
        }

        /* node->count  += (oi - node->tail); */
        /* node->tail = ((oi == fifosize) ? 0 : oi); */
        /* leftcount -= (li - lefthead); */
        /* lefthead = ((li == fifosize) ? 0 : li); */

        if (oi == oend) {
          oi = 0;
          oend = node->head;
          if (node->tail >= fifosize) {
            node->tail = 0;
            Out = node->buffer;
          }
        }

        if (lcases > 0 && li == lend) {
          li = 0;
          lend = lefttail;
          lcases = 0;
          if (lefthead >= fifosize) {
            lefthead = 0;
            L = left->buffer;
          }
        }

      } while (outnslots > 0 && leftcount > 0);
    }
  }

  /* if(!check_node_sorted(node, fifosize)) */
  /*     printf("Node not sorted rsz=%d, lsz=%d\n", rsz, lsz); */

  right->count = rightcount;
  right->head = righthead;
  left->count = leftcount;
  left->head = lefthead;
}

inline void __attribute__((always_inline, target("avx2")))
mergestore16kernel(tuple_t *A, tuple_t *B, tuple_t **Out, uint32_t *ri,
                   uint32_t *li, uint32_t rend, uint32_t lend) {
  int32_t lenA = rend - *ri, lenB = lend - *li;
  int32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;
  uint32_t rii = *ri, lii = *li;
  tuple_t *out = *Out;

  // if((uintptr_t)A % 64 != 0)
  //     printf("**** A not aligned = %d\n", (uintptr_t)A % 64);
  // if((uintptr_t)B % 64 != 0)
  //     printf("**** B not aligned = %d\n", (uintptr_t)B % 64);
  // if((uintptr_t)Out % 64 != 0)
  //     printf("**** Out not aligned = %d\n", (uintptr_t)Out % 64);

  if (lenA16 > 16 && lenB16 > 16) {
    register block16 *inA = (block16 *)A;
    register block16 *inB = (block16 *)B;
    block16 *const endA = (block16 *)(A + lenA) - 1;
    block16 *const endB = (block16 *)(B + lenB) - 1;

    block16 *outp = (block16 *)out;

    register block16 *next = inB;

    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8U(regAl1, regAl2, inA);
    LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    LOAD8U(regBl1, regBl2, inB);
    LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    outp++;

    while (inA < endA && inB < endB) {
      /** The inline assembly below does exactly the following code: */
      /* if(*((int64_t *)inA) < *((int64_t *)inB)) { */
      /*     next = inA; */
      /*     inA ++; */
      /* } */
      /* else { */
      /*     next = inB; */
      /*     inB ++; */
      /* } */
      /* Option 3: with assembly */
      IFELSECONDMOVE(next, inA, inB, 128);

      regAl1 = outreg2l1;
      regAl2 = outreg2l2;
      regAh1 = outreg2h1;
      regAh2 = outreg2h2;

      LOAD8U(regBl1, regBl2, next);
      LOAD8U(regBh1, regBh2, ((block8 *)next + 1));

      BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                      outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                      regAh2, regBl1, regBl2, regBh1, regBh2);

      /* store outreg1 */
      STORE8U(outp, outreg1l1, outreg1l2);
      STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
      outp++;
    }

    {
      /* flush the register to one of the lists */
      int64_t /*tuple_t*/ hireg[4] __attribute__((aligned(16)));
      _mm256_store_pd((double *)hireg, outreg2h2);

      /*if(((tuple_t *)inA)->key >= hireg[3].key){*/
      if (*((double *)inA) >= hireg[3]) {
        /* store the last remaining register values to A */
        inA--;
        STORE8U(inA, outreg2l1, outreg2l2);
        STORE8U(((block8 *)inA + 1), outreg2h1, outreg2h2);
      } else {
        /* store the last remaining register values to B */
        inB--;
        STORE8U(inB, outreg2l1, outreg2l2);
        STORE8U(((block8 *)inB + 1), outreg2h1, outreg2h2);
      }
    }

    rii = *ri + ((tuple_t *)inA - A);
    lii = *li + ((tuple_t *)inB - B);

    A = (tuple_t *)inA;
    B = (tuple_t *)inB;
    out = (tuple_t *)outp;
  }

  /* serial-merge */
  {
    int64_t *in1 = (int64_t *)A;
    int64_t *in2 = (int64_t *)B;
    int64_t *pout = (int64_t *)out;
    while (rii < rend && lii < lend) {
      int64_t *in = in2;
      uint32_t cmp = (*(double *)in1 < *(double *)in2);
      uint32_t notcmp = !cmp;
      rii += cmp;
      lii += notcmp;
      if (cmp) in = in1;
      *pout = *in;
      pout++;
      in1 += cmp;
      in2 += notcmp;
      /*
      if(*in1 < *in2){
          *pout = *in1;
          in1++;
          rii++;
      }
      else {
          *pout = *in2;
          in2++;
          lii++;
      }
      pout ++;
      */
    }
    out = (tuple_t *)pout;
    /* just for tuples, comparison on keys.
while(rii < rend && lii < lend){
    tuple_t * in = B;
    //uint32_t cmp = (A->key < B->key);
    uint32_t cmp =
    uint32_t notcmp = !cmp;

    rii += cmp;
    lii += notcmp;

    if(cmp)
        in = A;

     *out = *in;
    out ++;
    A += cmp;
    B += notcmp;
}
     */
  }

  *ri = rii;
  *li = lii;
  *Out = out;
}

void __attribute__((target("avx2")))
simd_memcpy(void *dst, void *src, size_t sz) {
  char *src_ptr = (char *)src;
  char *dst_ptr = (char *)dst;
  char *src_end = (char *)src + sz - 64;
  /* further improvement with aligned load/store */
  for (; src_ptr <= src_end; src_ptr += 64, dst_ptr += 64) {
    __asm volatile(
        "movdqu 0(%0) , %%xmm0;  "
        "movdqu 16(%0), %%xmm1;  "
        "movdqu 32(%0), %%xmm2;  "
        "movdqu 48(%0), %%xmm3;  "
        "movdqu %%xmm0, 0(%1) ;  "
        "movdqu %%xmm1, 16(%1) ;  "
        "movdqu %%xmm2, 32(%1) ;  "
        "movdqu %%xmm3, 48(%1) ;  " ::"r"(src_ptr),
        "r"(dst_ptr));
  }

  /* copy remainders */
  src_end += 64;
  if (src_ptr < src_end) {
    memcpy(dst_ptr, src_ptr, (src_end - src_ptr));
  }
}

inline void __attribute__((always_inline, target("avx2")))
merge16kernel(tuple_t *A, tuple_t *B, tuple_t *Out, uint32_t *ri, uint32_t *li,
              uint32_t *oi, uint32_t *outnslots, uint32_t rend, uint32_t lend) {
  int32_t lenA = rend - *ri, lenB = lend - *li;
  int32_t nslots = *outnslots;
  int32_t remNslots = nslots & 0xF;
  int32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;

  uint32_t rii = *ri, lii = *li, oii = *oi;
  nslots -= remNslots;

  if (nslots > 0 && lenA16 > 16 && lenB16 > 16) {
    register block16 *inA = (block16 *)A;
    register block16 *inB = (block16 *)B;
    block16 *const endA = (block16 *)(A + lenA) - 1;
    block16 *const endB = (block16 *)(B + lenB) - 1;

    block16 *outp = (block16 *)Out;

    register block16 *next = inB;

    __m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
    __m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;

    __m256d regAl1, regAl2, regAh1, regAh2;
    __m256d regBl1, regBl2, regBh1, regBh2;

    LOAD8U(regAl1, regAl2, inA);
    LOAD8U(regAh1, regAh2, ((block8 *)(inA) + 1));
    inA++;

    LOAD8U(regBl1, regBl2, inB);
    LOAD8U(regBh1, regBh2, ((block8 *)(inB) + 1));
    inB++;

    BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                    outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                    regAh2, regBl1, regBl2, regBh1, regBh2);

    /* store outreg1 */
    STORE8U(outp, outreg1l1, outreg1l2);
    STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
    nslots -= 16;
    outp++;

    while (nslots > 0 && inA < endA && inB < endB) {
      nslots -= 16;
      /** The inline assembly below does exactly the following code: */
      /* Option 3: with assembly */
      IFELSECONDMOVE(next, inA, inB, 128);

      regAl1 = outreg2l1;
      regAl2 = outreg2l2;
      regAh1 = outreg2h1;
      regAh2 = outreg2h2;

      LOAD8U(regBl1, regBl2, next);
      LOAD8U(regBh1, regBh2, ((block8 *)next + 1));

      BITONIC_MERGE16(outreg1l1, outreg1l2, outreg1h1, outreg1h2, outreg2l1,
                      outreg2l2, outreg2h1, outreg2h2, regAl1, regAl2, regAh1,
                      regAh2, regBl1, regBl2, regBh1, regBh2);

      /* store outreg1 */
      STORE8U(outp, outreg1l1, outreg1l2);
      STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
      outp++;
    }

    {
      /* flush the register to one of the lists */
      int64_t /*tuple_t*/ hireg[4] __attribute__((aligned(32)));
      _mm256_store_pd((double *)hireg, outreg2h2);

      /* if(((tuple_t *)inA)->key >= hireg[3].key){*/
      if (*((double *)inA) >= hireg[3]) {
        /* store the last remaining register values to A */
        inA--;
        STORE8U(inA, outreg2l1, outreg2l2);
        STORE8U(((block8 *)inA + 1), outreg2h1, outreg2h2);
      } else {
        /* store the last remaining register values to B */
        inB--;
        STORE8U(inB, outreg2l1, outreg2l2);
        STORE8U(((block8 *)inB + 1), outreg2h1, outreg2h2);
      }
    }

    rii = *ri + ((tuple_t *)inA - A);
    lii = *li + ((tuple_t *)inB - B);
    oii = *oi + ((tuple_t *)outp - Out);

    A = (tuple_t *)inA;
    B = (tuple_t *)inB;
    Out = (tuple_t *)outp;
  }
  nslots += remNslots;

  /* serial-merge */
  while ((nslots > 0 && rii < rend && lii < lend)) {
    tuple_t *in = B;
    uint32_t cmp = *((double *)A) < *((double *)B); /*(A->key < B->key);*/
    uint32_t notcmp = !cmp;

    rii += cmp;
    lii += notcmp;

    if (cmp) in = A;

    nslots--;
    oii++;
    *Out = *in;
    Out++;
    A += cmp;
    B += notcmp;
  }

  *ri = rii;
  *li = lii;
  *oi = oii;
  *outnslots = nslots;
}

inline void __attribute__((always_inline, target("avx2")))
merge8_varlen(int64_t *inpA, int64_t *inpB, int64_t *Out, const uint32_t lenA,
              const uint32_t lenB) {
  uint32_t lenA8 = lenA & ~0x7, lenB8 = lenB & ~0x7;
  uint32_t ai = 0, bi = 0;

  int64_t *out = Out;

  if (lenA8 > 8 && lenB8 > 8) {
    register block8 *inA = (block8 *)inpA;
    register block8 *inB = (block8 *)inpB;
    block8 *const endA = (block8 *)(inpA + lenA) - 1;
    block8 *const endB = (block8 *)(inpB + lenB) - 1;

    block8 *outp = (block8 *)out;

    register block8 *next = inB;

    register __m256d outreg1l, outreg1h;
    register __m256d outreg2l, outreg2h;

    register __m256d regAl, regAh;
    register __m256d regBl, regBh;

    LOAD8U(regAl, regAh, inA);
    LOAD8U(regBl, regBh, next);

    inA++;
    inB++;

    BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh, regBl,
                   regBh);

    /* store outreg1 */
    STORE8U(outp, outreg1l, outreg1h);
    outp++;

    while (inA < endA && inB < endB) {
      /* 3 Options : normal-if, cmove-with-assembly, sw-predication */
      IFELSECONDMOVE(next, inA, inB, 64);

      regAl = outreg2l;
      regAh = outreg2h;
      LOAD8U(regBl, regBh, next);

      BITONIC_MERGE8(outreg1l, outreg1h, outreg2l, outreg2h, regAl, regAh,
                     regBl, regBh);

      /* store outreg1 */
      STORE8U(outp, outreg1l, outreg1h);
      outp++;
    }

    /* flush the register to one of the lists */
    int64_t hireg[4] __attribute__((aligned(16)));
    _mm256_store_pd((double *)hireg, outreg2h);

    if (*((double *)inA) >= *((double *)(hireg + 3))) {
      /* store the last remaining register values to A */
      inA--;
      STORE8U(inA, outreg2l, outreg2h);
    } else {
      /* store the last remaining register values to B */
      inB--;
      STORE8U(inB, outreg2l, outreg2h);
    }

    ai = ((int64_t *)inA - inpA);
    bi = ((int64_t *)inB - inpB);

    inpA = (int64_t *)inA;
    inpB = (int64_t *)inB;
    out = (int64_t *)outp;
  }

  /* serial-merge */
  while (ai < lenA && bi < lenB) {
    int64_t *in = inpB;
    uint32_t cmp = (*(double *)inpA < *(double *)inpB);
    uint32_t notcmp = !cmp;

    ai += cmp;
    bi += notcmp;

    if (cmp) in = inpA;

    *out = *in;
    out++;
    inpA += cmp;
    inpB += notcmp;
  }

  if (ai < lenA) {
    /* if A has any more items to be output */

    if ((lenA - ai) >= 8) {
      /* if A still has some times to be output with AVX */
      uint32_t lenA8_ = ((lenA - ai) & ~0x7);
      register block8 *inA = (block8 *)inpA;
      block8 *const endA = (block8 *)(inpA + lenA8_);
      block8 *outp = (block8 *)out;

      while (inA < endA) {
        __m256d regAl, regAh;
        LOAD8U(regAl, regAh, inA);
        STORE8U(outp, regAl, regAh);
        outp++;
        inA++;
      }

      ai += ((int64_t *)inA - inpA);
      inpA = (int64_t *)inA;
      out = (int64_t *)outp;
    }

    while (ai < lenA) {
      *out = *inpA;
      ai++;
      out++;
      inpA++;
    }
  } else if (bi < lenB) {
    /* if B has any more items to be output */

    if ((lenB - bi) >= 8) {
      /* if B still has some times to be output with AVX */
      uint32_t lenB8_ = ((lenB - bi) & ~0x7);
      register block8 *inB = (block8 *)inpB;
      block8 *const endB = (block8 *)(inpB + lenB8_);
      block8 *outp = (block8 *)out;

      while (inB < endB) {
        __m256d regBl, regBh;
        LOAD8U(regBl, regBh, inB);
        STORE8U(outp, regBl, regBh);
        outp++;
        inB++;
      }

      bi += ((int64_t *)inB - inpB);
      inpB = (int64_t *)inB;
      out = (int64_t *)outp;
    }

    while (bi < lenB) {
      *out = *inpB;
      bi++;
      out++;
      inpB++;
    }
  }
}
