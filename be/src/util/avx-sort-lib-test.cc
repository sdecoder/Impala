#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <sys/time.h>
#include <gtest/gtest.h>
#include <ctime>
#include <iostream>
#include <algorithm>
#include <limits.h>

#include "avx-sort-lib.h"
#include "common/names.h"

//#define FLT_MAX (9.999999999999999e999)
//#define FLT_MIN (1e-999)

//#define TEST_VALUE_FIELD bool_payload [PASS]
//#define TEST_VALUE_FIELD char_payload [FAIL]
//#define TEST_VALUE_FIELD tiny_int_payload  //tinyint slots [FAIL]
//#define TEST_VALUE_FIELD small_int_payload //smallint slots [FAIL]
//#define TEST_VALUE_FIELD int_payload//; // int slots
//#define TEST_VALUE_FIELD float_payload //; //float slots

namespace impala {
#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))

int gen_random_int() {
  const int value = rand();
  int sign = rand();
  return sign > RAND_MAX / 2 ? value : -value;
}

float gen_random_float() {
  const float fmin_ = 1.175494e-38, fmax_ = 3.402823e+38;
  float r3 = fmin_ +
             static_cast<float>(rand()) /
                 (static_cast<float>(RAND_MAX / (fmax_ - fmin_)));
  return r3;
  int sign = rand();
  return sign > RAND_MAX / 2 ? r3 : -r3;
}

/*
inline int partition_(tuple_t* array, int length){
  int start = 0;
  int end = length -1;
  while( start < end){
    while ( array[start] < 0 && start < length) start++;
    while ( array[end] > 0 && end >= 0 ) end --;

    if( start < end){
      //swap
      tuple_t  tmp = array[start];
      array[start] = array[end];
      array[end]   = tmp;
    }

  }
  return end;
}
/*
TEST(AVXDirectSort, IntValue) {
  srand((unsigned)time(0));
  const int nitems = 10;
  const bool ascend = true;
  tuple_t* i32data;
  posix_memalign((void**)&i32data, CACHE_LINE_SIZE, nitems* sizeof(tuple_t));
  srand((unsigned)time(0));
  for (int i = 0; i < nitems; ++i)
  {
    i32data[i] = 0;
    SetPtr(i32data[i], i);
    SetKeyInt(i32data[i], gen_random_int());
  }


  tuple_t* output = avxsort_keyval(i32data, nitems, ascend);
  LOG(INFO) << "[dbg] Debug output: ";
  for (int i = 0; i < nitems; ++i)
        LOG(INFO)<<"[dbg] PTR: " << GetPtr(output[i])  << " KEY[I32Value]: " <<
GetKeyInt(output[i]) ;


  int start = 0;
  int end = nitems;
  int mid = 0;
  while (start < end) {
        int mid = start + ((end - start) >> 1);
        if( GetKeyInt(output[mid]) >= 0) end = mid -1;
        else start = mid +1;
  }

  LOG(INFO)<<"[dbg] mid: " << mid ;


}*/

TEST(AVXSort, IntValue) {
  srand((unsigned)time(0));
  const int nitems = 1048576;  // 2 ^20 = 1048576
  const bool ascend = true;
  tuple_t* i32data;
  tuple_t* i32data_copy;
  posix_memalign((void**)&i32data, CACHE_LINE_SIZE, nitems * sizeof(tuple_t));
  posix_memalign((void**)&i32data_copy, CACHE_LINE_SIZE,
                 nitems * sizeof(tuple_t));
  srand((unsigned)time(0));
  for (int i = 0; i < nitems; ++i) {
    i32data[i] = 0;
    SetPtr(i32data[i], i);
    SetKeyInt(i32data[i], gen_random_int());
    i32data_copy[i] = i32data[i];
  }

  //  int bound_ = partition_(i32data, nitems);
  struct timeval tvstart, tvend;
  tuple_t* outputptr;
  posix_memalign((void**)&outputptr, CACHE_LINE_SIZE, nitems * sizeof(tuple_t));

  gettimeofday(&tvstart, NULL);
  avxsort_tuples(&i32data, &outputptr, nitems);
  //  tuple_t* i32output = avxsort_keyval(i32data , nitems, ascend);
  gettimeofday(&tvend, NULL);
  long avx_time_consumed = (tvend.tv_sec - tvstart.tv_sec) * 1000000 +
                           (tvend.tv_usec - tvstart.tv_usec);
  LOG(INFO) << "[dbg] time consumed using AVX2 sort: " << avx_time_consumed
            << " ms";

  gettimeofday(&tvstart, NULL);
  std::sort(i32data_copy, i32data_copy + nitems);
  gettimeofday(&tvend, NULL);
  long sort_time_consumed = (tvend.tv_sec - tvstart.tv_sec) * 1000000 +
                            (tvend.tv_usec - tvstart.tv_usec);
  LOG(INFO) << "[dbg] time consumed using std::sort sort: "
            << sort_time_consumed << " ms";
  LOG(INFO) << "[dbg] AVX Sort Acceleration Ratio: "
            << (double)sort_time_consumed / avx_time_consumed << "X";

  return;

  /*
   tuple_t * neg_input; //= new tuple_t[nitems];
   tuple_t * pos_input; //= new tuple_t[nitems];
   if( bound_ > 0 ) {
     //posix_memalign((void**)&i32output, CACHE_LINE_SIZE, (bound_ + 1)*
   sizeof(tuple_t));
     posix_memalign((void**)&i32output, CACHE_LINE_SIZE, nitems *
   sizeof(tuple_t));
     posix_memalign((void**)&neg_input, CACHE_LINE_SIZE, (bound_ + 1)*
   sizeof(tuple_t));
     for (int i = 0; i < bound_ + 1; ++i)
     {
       //neg_input[i] = i32data[i];
       SetPtr(neg_input[i], GetPtr(i32data[i]));
       SetKeyInt(neg_input[i], - GetKeyInt( i32data[i]));
     }
     tuple_t* neg_output = avxsort_keyval(neg_input, bound_ + 1, ascend);

     //LOG(INFO) << "[dbg] Sorted data: ";
     for (int i = 0; i < bound_ +1 ; ++i)
     {
     //  LOG(INFO) << "[dbg] INDEX: " << i << " " << get_key_f(neg_output[i]);
     }


     reverse(neg_output,  neg_output + bound_  + 1);
     //LOG(INFO) << "[dbg] Reversed data: ";
     for (int i = 0; i < bound_ +1 ; ++i)
     {
       SetPtr(i32output[i], GetPtr(neg_output[i]));
       SetKeyInt(i32output[i], -GetKeyInt(neg_output[i]));
     }

     posix_memalign((void**)&pos_input, CACHE_LINE_SIZE, (nitems -  bound_ - 1)*
   sizeof(tuple_t));
     memcpy(pos_input, i32data + (bound_ + 1), (nitems -  bound_ - 1)*
   sizeof(tuple_t));
     tuple_t* pos_output = avxsort_keyval(pos_input, (nitems -  bound_ - 1),
   ascend);
     memcpy(i32output + bound_ + 1, pos_output, (nitems -  bound_ - 1) *
   sizeof(tuple_t));

   }else {
     i32output = avxsort_keyval(i32data, nitems, ascend);
   }*/

  /*
  #ifdef 0
  LOG(INFO)<<"[dbg] INDEX: "  << 0 <<" PTR: " << GetPtr(i32output[0]) << "\tKEY:
  " << GetKeyInt(i32output[0]); ;
  bool successful = true;
  for (int i = 1; i < nitems; ++i)
  {
    //int pre_key = get_key(outputptr[i-1]);
    int pre_i32key = GetKeyInt(i32output[i-1]);
    int cur_i32key = GetKeyInt(i32output[i]);
    //cout << get_key_i(outputptr[i]) << endl;

    int ptr_ = GetPtr(i32output[i]);
    LOG(INFO)<<"[dbg] INDEX: "  << i <<" PTR: " << ptr_ << "\tKEY: " <<
  cur_i32key ;
    if( pre_i32key > cur_i32key ){
      LOG(INFO)<<"[dbg] failed at: " << i - 1  << " and " << i ;
      LOG(INFO)<<"[dbg] failed at: " << pre_i32key << " and " << cur_i32key ;
      successful = false;
      break;
    }

  }
  if( successful )  LOG(INFO)<<"GOOD";
  else LOG(INFO)<<"BAD" ;
  #endif*/
}
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  impala::CpuInfo::Init();
  return RUN_ALL_TESTS();
}