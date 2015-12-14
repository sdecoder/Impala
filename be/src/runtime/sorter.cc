// Copyright 2013 Cloudera Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/sorter.h"
#include <gutil/strings/substitute.h>

#include "runtime/buffered-block-mgr.h"
#include "runtime/row-batch.h"
#include "runtime/runtime-state.h"
#include "runtime/sorted-run-merger.h"
#include "util/runtime-profile.h"

#include "common/names.h"

#include "util/avx-sort-lib.h"
#include <queue>

using namespace strings;

namespace impala {

// Number of pinned blocks required for a merge.
const int BLOCKS_REQUIRED_FOR_MERGE = 3;

// Error message when pinning fixed or variable length blocks failed.
// TODO: Add the node id that iniated the sort
const string PIN_FAILED_ERROR_MSG = "Failed to pin block for $0-length data needed "
    "for sorting. Reducing query concurrency or increasing the memory limit may help "
    "this query to complete successfully.";

const string MEM_ALLOC_FAILED_ERROR_MSG = "Failed to allocate block for $0-length "
    "data needed for sorting. Reducing query concurrency or increasing the "
    "memory limit may help this query to complete successfully.";

/// A run is a sequence of blocks containing tuples that are or will eventually be in
/// sorted order.
/// A run may maintain two sequences of blocks - one containing the tuples themselves,
/// (i.e. fixed-len slots and ptrs to var-len data), and the other for the var-length
/// column data pointed to by those tuples.
/// Tuples in a run may be sorted in place (in-memory) and merged using a merger.
class Sorter::Run {
 public:
  /// materialize_slots is true for runs constructed from input rows. The input rows are
  /// materialized into single sort tuples using the expressions in
  /// sort_tuple_slot_expr_ctxs_. For intermediate merges, the tuples are already
  /// materialized so materialize_slots is false.
  Run(Sorter* parent, TupleDescriptor* sort_tuple_desc, bool materialize_slots);

  ~Run() { DeleteAllBlocks(); }

  /// Initialize the run for input rows by allocating the minimum number of required
  /// blocks - one block for fixed-len data added to fixed_len_blocks_, one for the
  /// initially unsorted var-len data added to var_len_blocks_, and one to copy sorted
  /// var-len data into (var_len_copy_block_).
  Status Init();

  /// Add a batch of input rows to the current run. Returns the number
  /// of rows actually added in num_processed. If the run is full (no more blocks can
  /// be allocated), num_processed may be less than the number of rows in the batch.
  /// If materialize_slots_ is true, materializes the input rows using the expressions
  /// in sorter_->sort_tuple_slot_expr_ctxs_, else just copies the input rows.
  template <bool has_var_len_data>
  Status AddBatch(RowBatch* batch, int start_index, int* num_processed);

  /// Attaches all fixed-len and var-len blocks to the given row batch.
  void TransferResources(RowBatch* row_batch);

  /// Unpins all the blocks in a sorted run. Var-length column data is copied into new
  /// blocks in sorted order. Pointers in the original tuples are converted to offsets
  /// from the beginning of the sequence of var-len data blocks.
  Status UnpinAllBlocks();

  /// Deletes all blocks.
  void DeleteAllBlocks();

  /// Interface for merger - get the next batch of rows from this run. The callee (Run)
  /// still owns the returned batch. Calls GetNext(RowBatch*, bool*).
  Status GetNextBatch(RowBatch** sorted_batch);

 private:
  friend class Sorter;
  friend class TupleSorter;

  /// Fill output_batch with rows from this run. If convert_offset_to_ptr is true, offsets
  /// in var-length slots are converted back to pointers. Only row pointers are copied
  /// into output_batch.
  /// If this run was unpinned, one block (2 if there are var-len slots) is pinned while
  /// rows are filled into output_batch. The block is unpinned before the next block is
  /// pinned. Atmost 1 (2) block(s) will be pinned at any time.
  /// If the run was pinned, the blocks are not unpinned (Sorter holds on to the memory).
  /// In either case, all rows in output_batch will have their fixed and var-len data from
  /// the same block.
  /// TODO: If we leave the last run to be merged in memory, the fixed-len blocks can be
  /// unpinned as they are consumed.
  template <bool convert_offset_to_ptr>
  Status GetNext(RowBatch* output_batch, bool* eos);

  /// Check if a run can be extended by allocating additional blocks from the block
  /// manager. Always true when building a sorted run in an intermediate merge, because
  /// the current block(s) can be unpinned before getting the next free block (so a block
  /// is always available)
  bool CanExtendRun() const;

  /// Collect the non-null var-len (e.g. STRING) slots from 'src' in var_slots and return
  /// the total length of all var_len slots in total_var_len.
  void CollectNonNullVarSlots(Tuple* src, vector<StringValue*>* var_len_values,
      int* total_var_len);

  /// Check if the current run can be extended by a block. Add the newly allocated block
  /// to block_sequence, or set added to false if the run could not be extended.
  /// If the run is sorted (produced by an intermediate merge), unpin the last block in
  /// block_sequence before allocating and adding a new block - the run can always be
  /// extended in this case. If the run is unsorted, check max_blocks_in_unsorted_run_
  /// to see if a block can be added to the run. Also updates the sort bytes counter.
  Status TryAddBlock(vector<BufferedBlockMgr::Block*>* block_sequence, bool* added);

  /// Prepare to read a sorted run. Pins the first block(s) in the run if the run was
  /// previously unpinned.
  Status PrepareRead();

  /// Copy the StringValue data in var_values to dest in order and update the StringValue
  /// ptrs to point to the copied data.
  void CopyVarLenData(char* dest, const vector<StringValue*>& var_values);

  /// Copy the StringValue in var_values to dest in order. Update the StringValue ptrs to
  /// contain an offset to the copied data. Parameter 'offset' is the offset for the first
  /// StringValue.
  void CopyVarLenDataConvertOffset(char* dest, int64_t offset,
      const vector<StringValue*>& var_values);

  /// Returns true if we have var-len slots and there are var-len blocks.
  inline bool HasVarLenBlocks() const {
    return has_var_len_slots_ && !var_len_blocks_.empty();
  }

  /// Parent sorter object.
  const Sorter* sorter_;

  /// Materialized sort tuple. Input rows are materialized into 1 tuple (with descriptor
  /// sort_tuple_desc_) before sorting.
  const TupleDescriptor* sort_tuple_desc_;

  /// Sizes of sort tuple and block.
  const int sort_tuple_size_;
  const int block_size_;

  const bool has_var_len_slots_;

  /// True if the sort tuple must be materialized from the input batch in AddBatch().
  /// materialize_slots_ is true for runs being constructed from input batches, and
  /// is false for runs being constructed from intermediate merges.
  const bool materialize_slots_;

  /// True if the run is sorted. Set to true after an in-memory sort, and initialized to
  /// true for runs resulting from merges.
  bool is_sorted_;

  /// True if all blocks in the run are pinned.
  bool is_pinned_;

  /// Sequence of blocks in this run containing the fixed-length portion of the sort
  /// tuples comprising this run. The data pointed to by the var-len slots are in
  /// var_len_blocks_.
  /// If is_sorted_ is true, the tuples in fixed_len_blocks_ will be in sorted order.
  /// fixed_len_blocks_[i] is NULL iff it has been deleted.
  vector<BufferedBlockMgr::Block*> fixed_len_blocks_;

  /// Sequence of blocks in this run containing the var-length data corresponding to the
  /// var-length columns from fixed_len_blocks_. These are reconstructed to be in sorted
  /// order in UnpinAllBlocks().
  /// var_len_blocks_[i] is NULL iff it has been deleted.
  vector<BufferedBlockMgr::Block*> var_len_blocks_;

  /// If there are var-len slots, an extra pinned block is used to copy out var-len data
  /// into a new sequence of blocks in sorted order. var_len_copy_block_ stores this
  /// extra allocated block.
  BufferedBlockMgr::Block* var_len_copy_block_;

  /// Number of tuples so far in this run.
  int64_t num_tuples_;

  /// Number of tuples returned via GetNext(), maintained for debug purposes.
  int64_t num_tuples_returned_;

  /// buffered_batch_ is used to return TupleRows to the merger when this run is being
  /// merged. buffered_batch_ is returned in calls to GetNextBatch().
  scoped_ptr<RowBatch> buffered_batch_;

  /// Members used when a run is read in GetNext().
  /// The index into the fixed_ and var_len_blocks_ vectors of the current blocks being
  /// processed in GetNext().
  int fixed_len_blocks_index_;
  int var_len_blocks_index_;

  /// If true, pin the next fixed and var-len blocks and delete the previous ones
  /// in the next call to GetNext(). Set during the previous call to GetNext().
  /// Not used if a run is already pinned.
  bool pin_next_fixed_len_block_;
  bool pin_next_var_len_block_;

  /// Offset into the current fixed length data block being processed.
  int fixed_len_block_offset_;
}; // class Sorter::Run


/// Sorts a sequence of tuples from a run in place using a provided tuple comparator.
/// Quick sort is used for sequences of tuples larger that 16 elements, and insertion sort
/// is used for smaller sequences. The TupleSorter is initialized with a RuntimeState
/// instance to check for cancellation during an in-memory sort.
class Sorter::TupleSorter {
 public:
  TupleSorter(const TupleRowComparator& less_than_comp, int64_t block_size,
      int tuple_size, RuntimeState* state);

  ~TupleSorter();

  /// Performs a quicksort for tuples in 'run' followed by an insertion sort to
  /// finish smaller blocks.
  /// Returns early if stste_->is_cancelled() is true. No status
  /// is returned - the caller must check for cancellation.
  void Sort(Run* run);
  // AVX Sort function collection:
//  bool CanSortUsingAVX(const ColumnType& type);
  //  void SetTuplePtrKey(int64_t& carrier_, const int64_t bulk_start, int ptr, int colidx);

  void* GetDataByRowIdxColIdx(const std::vector<ExprContext*>& context, int64_t row_idx, int col_idx);
  bool SortNthColumn(const int colidx, const int64_t bulk_start, const int bulk_size, const bool asc_sort, int64_t* const localptr);

//  const int64_t osd_itv_start, const int64_t osd_itv_size, int* localptr ){
  void SetNull(int64_t& carrier_, int colidx);
  void SortInt32(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr,
    const ColumnType& type);

  void SortInt64(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortFloat(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortDouble(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortTimestamp(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size,  int64_t* const localptr);

  void SortTime(const int colidx, const int64_t bulk_start,
    const int isd_offset, const int isd_size, int64_t* const localptr);

  void SortStringVarchar(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortChar(const int colidx, const ColumnType& type, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortDecimal(const int width, const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortDecimal4Value(const int width, const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size,  int64_t* const  localptr);

  void SortDecimal8Value(const int width, const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  void SortDecimal16Value(const int width, const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size,  int64_t* const localptr);

  void SorterCore(const int colidx, const int64_t bulk_start,
    const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr);

  class mycomparison {
    const TupleRowComparator* comparator;
    TupleSorter* tuplesorter;

   public:
    void set_comparator(const TupleRowComparator* input) { comparator = input; }
    void set_tuple_sorter(TupleSorter* input) { tuplesorter = input; }

    bool operator()(const pair<int64_t, int64_t>& lhs, const pair<int64_t, int64_t>& rhs) const {

      const std::vector<ExprContext*>& context = comparator->get_key_expr_ctxs_lhs_();
      const int32_t colidx = 0;
      TupleIterator _cur_a = TupleIterator(tuplesorter, lhs.first);
      TupleIterator _cur_b = TupleIterator(tuplesorter, rhs.first);
      TupleRow* row_a = reinterpret_cast<TupleRow*>(&_cur_a.current_tuple_);
      TupleRow* row_b = reinterpret_cast<TupleRow*>(&_cur_b.current_tuple_);


      void* dataptr_a = context[colidx]->GetValue(row_a);
      void* dataptr_b = context[colidx]->GetValue(row_b);
      const int32_t i32value_a = *reinterpret_cast<const int32_t*>(dataptr_a);
      const int32_t i32value_b = *reinterpret_cast<const int32_t*>(dataptr_b);

      return true;

      if(i32value_a < i32value_b) return false;
      //if ((*comparator)(row_a, row_b)) return false;
      return true;
    }
  };


 private:
  static const int INSERTION_THRESHOLD = 16;

  /// Helper class used to iterate over tuples in a run during quick sort and insertion
  /// sort.
  class TupleIterator {
   public:
    TupleIterator(TupleSorter* parent, int64_t index)
      : parent_(parent),
        index_(index),
        current_tuple_(NULL) {
      DCHECK_GE(index, 0);
      DCHECK_LE(index, parent_->run_->num_tuples_);
      // If the run is empty, only index_ and current_tuple_ are initialized.
      if (parent_->run_->num_tuples_ == 0) return;
      // If the iterator is initialized to past the end, set up buffer_start_ and
      // block_index_ as if it pointing to the last tuple. Add tuple_size_ bytes to
      // current_tuple_, so everything is correct when Prev() is invoked.
      int past_end_bytes = 0;
      if (UNLIKELY(index >= parent_->run_->num_tuples_)) {
        past_end_bytes = parent->tuple_size_;
        index_ = parent_->run_->num_tuples_;
        index = index_ - 1;
      }
      block_index_ = index / parent->block_capacity_;
      buffer_start_ = parent->run_->fixed_len_blocks_[block_index_]->buffer();
      int block_offset = (index % parent->block_capacity_) * parent->tuple_size_;
      current_tuple_ = buffer_start_ + block_offset + past_end_bytes;
    }

    /// Sets current_tuple_ to point to the next tuple in the run. Increments
    /// block_index and resets buffer if the next tuple is in the next block.
    void Next() {
      current_tuple_ += parent_->tuple_size_;
      ++index_;
      if (UNLIKELY(current_tuple_ > buffer_start_ + parent_->last_tuple_block_offset_ &&
          index_ < parent_->run_->num_tuples_)) {
       // Don't increment block index, etc. past the end.
       ++block_index_;
       DCHECK_LT(block_index_, parent_->run_->fixed_len_blocks_.size());
       buffer_start_ = parent_->run_->fixed_len_blocks_[block_index_]->buffer();
       current_tuple_ = buffer_start_;
      }
    }

    /// Sets current_tuple to point to the previous tuple in the run. Decrements
    /// block_index and resets buffer if the new tuple is in the previous block.
    void Prev() {
      current_tuple_ -= parent_->tuple_size_;
      --index_;
      if (UNLIKELY(current_tuple_ < buffer_start_ && index_ >= 0)) {
        --block_index_;
        DCHECK_GE(block_index_, 0);
        buffer_start_ = parent_->run_->fixed_len_blocks_[block_index_]->buffer();
        current_tuple_ = buffer_start_ + parent_->last_tuple_block_offset_;
      }
    }

   private:
    friend class TupleSorter;

    /// Pointer to the tuple sorter.
    TupleSorter* parent_;

    /// Index of the current tuple in the run.
    int64_t index_;

    /// Pointer to the current tuple.
    uint8_t* current_tuple_;

    /// Start of the buffer containing current tuple.
    uint8_t* buffer_start_;

    /// Index into run_.fixed_len_blocks_ of the block containing the current tuple.
    int block_index_;
  };

  /// Size of the tuples in memory.
  const int tuple_size_;

  /// Number of tuples per block in a run.
  const int block_capacity_;

  /// Offset in bytes of the last tuple in a block, calculated from block and tuple sizes.
  const int last_tuple_block_offset_;

  /// Tuple comparator that returns true if lhs < rhs.
  const TupleRowComparator less_than_comp_;

  /// Runtime state instance to check for cancellation. Not owned.
  RuntimeState* const state_;

  /// The run to be sorted.
  Run* run_;

  /// Temporarily allocated space to copy and swap tuples (Both are used in Partition()).
  /// temp_tuple_ points to temp_tuple_buffer_. Owned by this TupleSorter instance.
  TupleRow* temp_tuple_row_;
  uint8_t* temp_tuple_buffer_;
  uint8_t* swap_buffer_;

  /// Perform an insertion sort for rows in the range [first, last) in a run.
  void InsertionSort(const TupleIterator& first, const TupleIterator& last);

  /// Partitions the sequence of tuples in the range [first, last) in a run into two
  /// groups around the pivot tuple - i.e. tuples in first group are <= the pivot, and
  /// tuples in the second group are >= pivot. Tuples are swapped in place to create the
  /// groups and the index to the first element in the second group is returned.
  /// Checks state_->is_cancelled() and returns early with an invalid result if true.
  TupleIterator Partition(TupleIterator first, TupleIterator last, Tuple* pivot);

  /// Performs a quicksort of rows in the range [first, last) followed by insertion sort
  /// for smaller groups of elements.
  /// Checks state_->is_cancelled() and returns early if true.
  void SortHelper(TupleIterator first, TupleIterator last);

  /// Swaps tuples pointed to by left and right using the swap buffer.
  void Swap(uint8_t* left, uint8_t* right);
}; // class TupleSorter

// Sorter::Run methods
Sorter::Run::Run(Sorter* parent, TupleDescriptor* sort_tuple_desc,
    bool materialize_slots)
  : sorter_(parent),
    sort_tuple_desc_(sort_tuple_desc),
    sort_tuple_size_(sort_tuple_desc->byte_size()),
    block_size_(parent->block_mgr_->max_block_size()),
    has_var_len_slots_(sort_tuple_desc->HasVarlenSlots()),
    materialize_slots_(materialize_slots),
    is_sorted_(!materialize_slots),
    is_pinned_(true),
    var_len_copy_block_(NULL),
    num_tuples_(0) {
}

Status Sorter::Run::Init() {
  BufferedBlockMgr::Block* block = NULL;
  RETURN_IF_ERROR(
      sorter_->block_mgr_->GetNewBlock(sorter_->block_mgr_client_, NULL, &block));
  if (block == NULL) {
    Status status = Status::MemLimitExceeded();
    status.AddDetail(Substitute(MEM_ALLOC_FAILED_ERROR_MSG, "fixed"));
    return status;
  }
  fixed_len_blocks_.push_back(block);
  if (has_var_len_slots_) {
    RETURN_IF_ERROR(
        sorter_->block_mgr_->GetNewBlock(sorter_->block_mgr_client_, NULL, &block));
    if (block == NULL) {
      Status status = Status::MemLimitExceeded();
      status.AddDetail(Substitute(MEM_ALLOC_FAILED_ERROR_MSG, "variable"));
      return status;
    }
    var_len_blocks_.push_back(block);
    if (!is_sorted_) {
      RETURN_IF_ERROR(sorter_->block_mgr_->GetNewBlock(
          sorter_->block_mgr_client_, NULL, &var_len_copy_block_));
      if (var_len_copy_block_ == NULL) {
        Status status = Status::MemLimitExceeded();
        status.AddDetail(Substitute(MEM_ALLOC_FAILED_ERROR_MSG, "variable"));
        return status;
      }
    }
  }
  if (!is_sorted_) sorter_->initial_runs_counter_->Add(1);
  return Status::OK();
}

template <bool has_var_len_data>
Status Sorter::Run::AddBatch(RowBatch* batch, int start_index, int* num_processed) {
  DCHECK(!fixed_len_blocks_.empty());
  *num_processed = 0;
  BufferedBlockMgr::Block* cur_fixed_len_block = fixed_len_blocks_.back();

  DCHECK_EQ(materialize_slots_, !is_sorted_);
  if (!materialize_slots_) {
    // If materialize slots is false the run is being constructed for an
    // intermediate merge and the sort tuples have already been materialized.
    // The input row should have the same schema as the sort tuples.
    DCHECK_EQ(batch->row_desc().tuple_descriptors().size(), 1);
    DCHECK_EQ(batch->row_desc().tuple_descriptors()[0], sort_tuple_desc_);
  }

  // Input rows are copied/materialized into tuples allocated in fixed_len_blocks_.
  // The variable length column data are copied into blocks stored in var_len_blocks_.
  // Input row processing is split into two loops.
  // The inner loop processes as many input rows as will fit in cur_fixed_len_block.
  // The outer loop allocates a new block for fixed-len data if the input batch is
  // not exhausted.

  // cur_input_index is the index into the input 'batch' of the current input row being
  // processed.
  int cur_input_index = start_index;
  vector<StringValue*> string_values;
  string_values.reserve(sort_tuple_desc_->string_slots().size());
  while (cur_input_index < batch->num_rows()) {
    // tuples_remaining is the number of tuples to copy/materialize into
    // cur_fixed_len_block.
    int tuples_remaining = cur_fixed_len_block->BytesRemaining() / sort_tuple_size_;
    tuples_remaining = min(batch->num_rows() - cur_input_index, tuples_remaining);

    for (int i = 0; i < tuples_remaining; ++i) {
      int total_var_len = 0;
      TupleRow* input_row = batch->GetRow(cur_input_index);
      Tuple* new_tuple = cur_fixed_len_block->Allocate<Tuple>(sort_tuple_size_);
      if (materialize_slots_) {
        new_tuple->MaterializeExprs<has_var_len_data>(input_row, *sort_tuple_desc_,
            sorter_->sort_tuple_slot_expr_ctxs_, NULL, &string_values, &total_var_len);
        if (total_var_len > sorter_->block_mgr_->max_block_size()) {
          return Status(ErrorMsg(TErrorCode::INTERNAL_ERROR, Substitute(
              "Variable length data in a single tuple larger than block size $0 > $1",
              total_var_len, sorter_->block_mgr_->max_block_size())));
        }
      } else {
        memcpy(new_tuple, input_row->GetTuple(0), sort_tuple_size_);
        if (has_var_len_data) {
          CollectNonNullVarSlots(new_tuple, &string_values, &total_var_len);
        }
      }

      if (has_var_len_data) {
        DCHECK_GT(var_len_blocks_.size(), 0);
        BufferedBlockMgr::Block* cur_var_len_block = var_len_blocks_.back();
        if (cur_var_len_block->BytesRemaining() < total_var_len) {
          bool added;
          RETURN_IF_ERROR(TryAddBlock(&var_len_blocks_, &added));
          if (added) {
            cur_var_len_block = var_len_blocks_.back();
          } else {
            // There was not enough space in the last var-len block for this tuple, and
            // the run could not be extended. Return the fixed-len allocation and exit.
            cur_fixed_len_block->ReturnAllocation(sort_tuple_size_);
            return Status::OK();
          }
        }

        // Sorting of tuples containing array values is not implemented. The planner
        // combined with projection should guarantee that none are in each tuple.
        BOOST_FOREACH(const SlotDescriptor* collection_slot,
            sort_tuple_desc_->collection_slots()) {
          DCHECK(new_tuple->IsNull(collection_slot->null_indicator_offset()));
        }

        char* var_data_ptr = cur_var_len_block->Allocate<char>(total_var_len);
        if (materialize_slots_) {
          CopyVarLenData(var_data_ptr, string_values);
        } else {
          int64_t offset = (var_len_blocks_.size() - 1) * block_size_;
          offset += var_data_ptr - reinterpret_cast<char*>(cur_var_len_block->buffer());
          CopyVarLenDataConvertOffset(var_data_ptr, offset, string_values);
        }
      }
      ++num_tuples_;
      ++*num_processed;
      ++cur_input_index;
    }

    // If there are still rows left to process, get a new block for the fixed-length
    // tuples. If the run is already too long, return.
    if (cur_input_index < batch->num_rows()) {
      bool added;
      RETURN_IF_ERROR(TryAddBlock(&fixed_len_blocks_, &added));
      if (added) {
        cur_fixed_len_block = fixed_len_blocks_.back();
      } else {
        return Status::OK();
      }
    }
  }
  return Status::OK();
}

void Sorter::Run::TransferResources(RowBatch* row_batch) {
  DCHECK(row_batch != NULL);
  BOOST_FOREACH(BufferedBlockMgr::Block* block, fixed_len_blocks_) {
    if (block != NULL) row_batch->AddBlock(block);
  }
  fixed_len_blocks_.clear();
  BOOST_FOREACH(BufferedBlockMgr::Block* block, var_len_blocks_) {
    if (block != NULL) row_batch->AddBlock(block);
  }
  var_len_blocks_.clear();
  if (var_len_copy_block_ != NULL) {
    row_batch->AddBlock(var_len_copy_block_);
    var_len_copy_block_ = NULL;
  }
}

void Sorter::Run::DeleteAllBlocks() {
  BOOST_FOREACH(BufferedBlockMgr::Block* block, fixed_len_blocks_) {
    if (block != NULL) block->Delete();
  }
  fixed_len_blocks_.clear();
  BOOST_FOREACH(BufferedBlockMgr::Block* block, var_len_blocks_) {
    if (block != NULL) block->Delete();
  }
  var_len_blocks_.clear();
  if (var_len_copy_block_ != NULL) {
    var_len_copy_block_->Delete();
    var_len_copy_block_ = NULL;
  }
}

Status Sorter::Run::UnpinAllBlocks() {
  vector<BufferedBlockMgr::Block*> sorted_var_len_blocks;
  sorted_var_len_blocks.reserve(var_len_blocks_.size());
  vector<StringValue*> string_values;
  int64_t var_data_offset = 0;
  int total_var_len;
  string_values.reserve(sort_tuple_desc_->string_slots().size());
  BufferedBlockMgr::Block* cur_sorted_var_len_block = NULL;
  if (HasVarLenBlocks()) {
    DCHECK(var_len_copy_block_ != NULL);
    sorted_var_len_blocks.push_back(var_len_copy_block_);
    cur_sorted_var_len_block = var_len_copy_block_;
  } else {
    DCHECK(var_len_copy_block_ == NULL);
  }

  for (int i = 0; i < fixed_len_blocks_.size(); ++i) {
    BufferedBlockMgr::Block* cur_fixed_block = fixed_len_blocks_[i];
    if (HasVarLenBlocks()) {
      for (int block_offset = 0; block_offset < cur_fixed_block->valid_data_len();
          block_offset += sort_tuple_size_) {
        Tuple* cur_tuple =
            reinterpret_cast<Tuple*>(cur_fixed_block->buffer() + block_offset);
        CollectNonNullVarSlots(cur_tuple, &string_values, &total_var_len);
        DCHECK(cur_sorted_var_len_block != NULL);
        if (cur_sorted_var_len_block->BytesRemaining() < total_var_len) {
          bool added;
          RETURN_IF_ERROR(TryAddBlock(&sorted_var_len_blocks, &added));
          DCHECK(added);
          cur_sorted_var_len_block = sorted_var_len_blocks.back();
        }
        char* var_data_ptr = cur_sorted_var_len_block->Allocate<char>(total_var_len);
        var_data_offset = block_size_ * (sorted_var_len_blocks.size() - 1) +
            (var_data_ptr - reinterpret_cast<char*>(cur_sorted_var_len_block->buffer()));
        CopyVarLenDataConvertOffset(var_data_ptr, var_data_offset, string_values);
      }
    }
    RETURN_IF_ERROR(cur_fixed_block->Unpin());
  }

  if (has_var_len_slots_ && var_len_blocks_.size() > 0) {
    DCHECK_GT(sorted_var_len_blocks.back()->valid_data_len(), 0);
    RETURN_IF_ERROR(sorted_var_len_blocks.back()->Unpin());
  }

  // Clear var_len_blocks_ and replace with it with the contents of sorted_var_len_blocks
  BOOST_FOREACH(BufferedBlockMgr::Block* var_block, var_len_blocks_) {
    RETURN_IF_ERROR(var_block->Delete());
  }
  var_len_blocks_.clear();
  sorted_var_len_blocks.swap(var_len_blocks_);
  // Set var_len_copy_block_ to NULL since it's now in var_len_blocks_ and is no longer
  // needed.
  var_len_copy_block_ = NULL;
  is_pinned_ = false;
  return Status::OK();
}

Status Sorter::Run::PrepareRead() {
  fixed_len_blocks_index_ = 0;
  fixed_len_block_offset_ = 0;
  var_len_blocks_index_ = 0;
  pin_next_fixed_len_block_ = pin_next_var_len_block_ = false;
  num_tuples_returned_ = 0;

  buffered_batch_.reset(new RowBatch(*sorter_->output_row_desc_,
      sorter_->state_->batch_size(), sorter_->mem_tracker_));

  // If the run is pinned, merge is not invoked, so buffered_batch_ is not needed
  // and the individual blocks do not need to be pinned.
  if (is_pinned_) return Status::OK();

  // Attempt to pin the first fixed and var-length blocks. In either case, pinning may
  // fail if the number of reserved blocks is oversubscribed, see IMPALA-1590.
  if (fixed_len_blocks_.size() > 0) {
    bool pinned;
    RETURN_IF_ERROR(fixed_len_blocks_[0]->Pin(&pinned));
    // Temporary work-around for IMPALA-1868. Fail the query with OOM rather than
    // DCHECK in case block pin fails.
    if (!pinned) {
      Status status = Status::MemLimitExceeded();
      status.AddDetail(Substitute(PIN_FAILED_ERROR_MSG, "fixed"));
      return status;
    }
  }

  if (has_var_len_slots_ && var_len_blocks_.size() > 0) {
    bool pinned;
    RETURN_IF_ERROR(var_len_blocks_[0]->Pin(&pinned));
    // Temporary work-around for IMPALA-1590. Fail the query with OOM rather than
    // DCHECK in case block pin fails.
    if (!pinned) {
      Status status = Status::MemLimitExceeded();
      status.AddDetail(Substitute(PIN_FAILED_ERROR_MSG, "variable"));
      return status;
    }
  }
  return Status::OK();
}

Status Sorter::Run::GetNextBatch(RowBatch** output_batch) {
  if (buffered_batch_.get() != NULL) {
    buffered_batch_->Reset();
    // Fill more rows into buffered_batch_.
    bool eos;
    if (has_var_len_slots_ && !is_pinned_) {
      RETURN_IF_ERROR(GetNext<true>(buffered_batch_.get(), &eos));
      if (buffered_batch_->num_rows() == 0 && !eos) {
        // No rows were filled because GetNext() had to read the next var-len block
        // Call GetNext() again.
        RETURN_IF_ERROR(GetNext<true>(buffered_batch_.get(), &eos));
      }
    } else {
      RETURN_IF_ERROR(GetNext<false>(buffered_batch_.get(), &eos));
    }
    DCHECK(eos || buffered_batch_->num_rows() > 0);
    if (eos) {
      // No rows are filled in GetNext() on eos, so this is safe.
      DCHECK_EQ(buffered_batch_->num_rows(), 0);
      buffered_batch_.reset();
      // The merge is complete. Delete the last blocks in the run.
      RETURN_IF_ERROR(fixed_len_blocks_.back()->Delete());
      fixed_len_blocks_[fixed_len_blocks_.size() - 1] = NULL;
      if (HasVarLenBlocks()) {
        RETURN_IF_ERROR(var_len_blocks_.back()->Delete());
        var_len_blocks_[var_len_blocks_.size() - 1] = NULL;
      }
    }
  }

  // *output_batch == NULL indicates eos.
  *output_batch = buffered_batch_.get();
  return Status::OK();
}

template <bool convert_offset_to_ptr>
Status Sorter::Run::GetNext(RowBatch* output_batch, bool* eos) {
  if (fixed_len_blocks_index_ == fixed_len_blocks_.size()) {
    *eos = true;
    DCHECK_EQ(num_tuples_returned_, num_tuples_);
    return Status::OK();
  } else {
    *eos = false;
  }

  BufferedBlockMgr::Block* fixed_len_block = fixed_len_blocks_[fixed_len_blocks_index_];

  if (!is_pinned_) {
    // Pin the next block and delete the previous if set in the previous call to
    // GetNext().
    if (pin_next_fixed_len_block_) {
      RETURN_IF_ERROR(fixed_len_blocks_[fixed_len_blocks_index_ - 1]->Delete());
      fixed_len_blocks_[fixed_len_blocks_index_ - 1] = NULL;
      bool pinned;
      RETURN_IF_ERROR(fixed_len_block->Pin(&pinned));
      // Temporary work-around for IMPALA-2344. Fail the query with OOM rather than
      // DCHECK in case block pin fails.
      if (!pinned) {
        Status status = Status::MemLimitExceeded();
        status.AddDetail(Substitute(PIN_FAILED_ERROR_MSG, "fixed"));
        return status;
      }
      pin_next_fixed_len_block_ = false;
    }
    if (pin_next_var_len_block_) {
      RETURN_IF_ERROR(var_len_blocks_[var_len_blocks_index_ - 1]->Delete());
      var_len_blocks_[var_len_blocks_index_ - 1] = NULL;
      bool pinned;
      RETURN_IF_ERROR(var_len_blocks_[var_len_blocks_index_]->Pin(&pinned));
      // Temporary work-around for IMPALA-2344. Fail the query with OOM rather than
      // DCHECK in case block pin fails.
      if (!pinned) {
        Status status = Status::MemLimitExceeded();
        status.AddDetail(Substitute(PIN_FAILED_ERROR_MSG, "variable"));
        return status;
      }
      pin_next_var_len_block_ = false;
    }
  }

  // GetNext fills rows into the output batch until a block boundary is reached.
  DCHECK(fixed_len_block != NULL);
  while (!output_batch->AtCapacity() &&
      fixed_len_block_offset_ < fixed_len_block->valid_data_len()) {
    DCHECK(fixed_len_block != NULL);
    Tuple* input_tuple = reinterpret_cast<Tuple*>(
        fixed_len_block->buffer() + fixed_len_block_offset_);

    if (convert_offset_to_ptr) {
      // Convert the offsets in the var-len slots in input_tuple back to pointers.
      const vector<SlotDescriptor*>& string_slots = sort_tuple_desc_->string_slots();
      for (int i = 0; i < string_slots.size(); ++i) {
        SlotDescriptor* slot_desc = string_slots[i];
        if (input_tuple->IsNull(slot_desc->null_indicator_offset())) continue;

        DCHECK(slot_desc->type().IsVarLenStringType());
        StringValue* value = reinterpret_cast<StringValue*>(
            input_tuple->GetSlot(slot_desc->tuple_offset()));
        int64_t data_offset = reinterpret_cast<int64_t>(value->ptr);

        // data_offset is an offset in bytes from the beginning of the first block
        // in var_len_blocks_. Convert it into an index into var_len_blocks_ and an
        // offset within that block.
        int block_index = data_offset / block_size_;
        int block_offset = data_offset % block_size_;

        if (block_index > var_len_blocks_index_) {
          // We've reached the block boundary for the current var-len block.
          // This tuple will be returned in the next call to GetNext().
          DCHECK_EQ(block_index, var_len_blocks_index_ + 1);
          DCHECK_EQ(block_offset, 0);
          DCHECK_EQ(i, 0);
          var_len_blocks_index_ = block_index;
          pin_next_var_len_block_ = true;
          break;
        } else {
          DCHECK_EQ(block_index, var_len_blocks_index_);
          // Calculate the address implied by the offset and assign it.
          value->ptr = reinterpret_cast<char*>(
              var_len_blocks_[var_len_blocks_index_]->buffer() + block_offset);
        } // if (block_index > var_len_blocks_index_)
      } // for (int i = 0; i < string_slots.size(); ++i)

      // The var-len data is in the next block, so end this call to GetNext().
      if (pin_next_var_len_block_) break;
    } // if (convert_offset_to_ptr)

    int output_row_idx = output_batch->AddRow();
    output_batch->GetRow(output_row_idx)->SetTuple(0, input_tuple);
    output_batch->CommitLastRow();
    fixed_len_block_offset_ += sort_tuple_size_;
    ++num_tuples_returned_;
  }

  // Reached the block boundary, need to move to the next block.
  if (fixed_len_block_offset_ >= fixed_len_block->valid_data_len()) {
    pin_next_fixed_len_block_ = true;
    ++fixed_len_blocks_index_;
    fixed_len_block_offset_ = 0;
  }
  return Status::OK();
}

void Sorter::Run::CollectNonNullVarSlots(Tuple* src,
    vector<StringValue*>* string_values, int* total_var_len) {
  string_values->clear();
  *total_var_len = 0;
  BOOST_FOREACH(const SlotDescriptor* string_slot, sort_tuple_desc_->string_slots()) {
    if (!src->IsNull(string_slot->null_indicator_offset())) {
      StringValue* string_val =
          reinterpret_cast<StringValue*>(src->GetSlot(string_slot->tuple_offset()));
      string_values->push_back(string_val);
      *total_var_len += string_val->len;
    }
  }
}

Status Sorter::Run::TryAddBlock(vector<BufferedBlockMgr::Block*>* block_sequence,
    bool* added) {
  DCHECK(!block_sequence->empty());
  BufferedBlockMgr::Block* last_block = block_sequence->back();
  if (!is_sorted_) {
    sorter_->sorted_data_size_->Add(last_block->valid_data_len());
    last_block = NULL;
  } else {
    // If the run is sorted, we will unpin the last block and extend the run.
  }

  BufferedBlockMgr::Block* new_block;
  RETURN_IF_ERROR(sorter_->block_mgr_->GetNewBlock(
      sorter_->block_mgr_client_, last_block, &new_block));
  if (new_block != NULL) {
    *added = true;
    block_sequence->push_back(new_block);
  } else {
    *added = false;
  }
  return Status::OK();
}

void Sorter::Run::CopyVarLenData(char* dest, const vector<StringValue*>& string_values) {
  BOOST_FOREACH(StringValue* string_val, string_values) {
    memcpy(dest, string_val->ptr, string_val->len);
    string_val->ptr = dest;
    dest += string_val->len;
  }
}

void Sorter::Run::CopyVarLenDataConvertOffset(char* dest, int64_t offset,
    const vector<StringValue*>& string_values) {
  BOOST_FOREACH(StringValue* string_val, string_values) {
    memcpy(dest, string_val->ptr, string_val->len);
    string_val->ptr = reinterpret_cast<char*>(offset);
    dest += string_val->len;
    offset += string_val->len;
  }
}

// Sorter::TupleSorter methods.
Sorter::TupleSorter::TupleSorter(const TupleRowComparator& comp, int64_t block_size,
    int tuple_size, RuntimeState* state)
  : tuple_size_(tuple_size),
    block_capacity_(block_size / tuple_size),
    last_tuple_block_offset_(tuple_size * ((block_size / tuple_size) - 1)),
    less_than_comp_(comp),
    state_(state) {
  temp_tuple_buffer_ = new uint8_t[tuple_size];
  temp_tuple_row_ = reinterpret_cast<TupleRow*>(&temp_tuple_buffer_);
  swap_buffer_ = new uint8_t[tuple_size];
}

Sorter::TupleSorter::~TupleSorter() {
  delete[] temp_tuple_buffer_;
  delete[] swap_buffer_;
}


inline void* Sorter::TupleSorter::GetDataByRowIdxColIdx(const std::vector<ExprContext*>& context, int64_t row_idx, int col_idx) {
  TupleIterator start_tuple_ = TupleIterator(this, row_idx );
  TupleRow* tuple_row_ = reinterpret_cast<TupleRow*>(&start_tuple_.current_tuple_);
  return context[col_idx]->GetValue(tuple_row_);
}


void Sorter::TupleSorter::SetNull(int64_t& carrier_, int colidx){
  const std::vector<bool>& is_asc_ = less_than_comp_.get_is_asc_();
  const std::vector<int8_t>& nulls_first = less_than_comp_.get_nulls_first_();
  bool asc_sort = is_asc_[colidx];
  int8_t null_first = nulls_first[colidx];
  if( asc_sort ) {
    if (null_first > 0) SetInfPos(carrier_);
    else SetInfNeg(carrier_);
  } else {
    if ( null_first > 0 ) SetInfNeg(carrier_);
    else SetInfPos(carrier_);
  }
}

void Sorter::TupleSorter::SortInt32(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr, const ColumnType& type){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size* sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i) {
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx);
    if (vdata == NULL) {
      SetNull(in[i], colidx);
    } else {
      switch (type.type) {
        case TYPE_BOOLEAN: {
          const bool bvalue_ = *reinterpret_cast<const bool*>(vdata);
          SetKeyInt(in[i], bvalue_);
          continue;
        }
        case TYPE_TINYINT: {
          const int8_t i8value_ = *reinterpret_cast<const int8_t*>(vdata);
          SetKeyInt(in[i], i8value_);
          continue;
        }
        case TYPE_SMALLINT: {
          const int16_t i16value_ = *reinterpret_cast<const int16_t*>(vdata);
          SetKeyInt(in[i], i16value_);
          continue;
        }
        case TYPE_INT: {
          const int32_t i32value_ = *reinterpret_cast<const int32_t*>(vdata);
          SetKeyInt(in[i], i32value_);
          continue;
        }
        default: { DCHECK(false) << "invalid i32 type: " << type; }
      };
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int dx = 0; dx < osd_itv_size; ++dx)
    localptr[osd_itv_start + dx] = GetPtr(out[dx]);
  free(in); free(out);
}

void Sorter::TupleSorter::SortInt64(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata == NULL){
      SetPtr(in[i], localptr[osd_itv_start + i]); SetNull(in[i], colidx);
    } else {
      const int64_t i64value =  *reinterpret_cast<const int64_t*>(vdata);
      set_ptrkey_i64_hi32(in[i], localptr[osd_itv_start + i], i64value);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int dx = 0; dx < osd_itv_size; ++dx)
    localptr[osd_itv_start + dx] = GetPtr(out[dx]);

  int start_ = 0; int end_ = osd_itv_size ;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end_ - 1 ] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) --end_; else break;
  }

  int isd_itv_start = start_;
  int isd_itv_size = 1;
  for (int i = start_ + 1; i < end_; ++i) {
    const int64_t pre_glb_row_no = bulk_start + localptr[osd_itv_start + i - 1 ] ;
    const int64_t cur_glb_row_no = bulk_start + localptr[osd_itv_start + i ] ;
    void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, pre_glb_row_no, colidx);
    void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, cur_glb_row_no, colidx);
    const int64_t prerow_i64 =  *reinterpret_cast<const int64_t*>(predata);
    const int64_t currow_i64 =  *reinterpret_cast<const int64_t*>(curdata);
    if ( compare_i64_hi32bit(prerow_i64, currow_i64) ) {
      ++isd_itv_size;
      if( i != end_ - 1) continue;
    }
    if (isd_itv_size > 1){
      tuple_t* tmp_in; tuple_t* tmp_out;
      posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      memset(tmp_in, 0, isd_itv_size* sizeof(tuple_t));
      for (int dx = 0; dx < isd_itv_size; ++dx)
      {
        const int64_t glb_row_no = bulk_start + localptr[ osd_itv_start + isd_itv_start + dx ];
        void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx);
        const int64_t currow_i64 = *reinterpret_cast<const int64_t*>(vdata_);
        set_ptrkey_i64_lo32(tmp_in[dx], localptr[ osd_itv_start + isd_itv_start + dx ], currow_i64);
      }
      avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);
      for (int dx = 0; dx < isd_itv_size; ++dx)
        localptr[osd_itv_start + isd_itv_start + dx] = GetPtr(tmp_out[dx]);
      free(tmp_in); free(tmp_out);
    }
    isd_itv_start = i;
    isd_itv_size = 1;
  }
}


void Sorter::TupleSorter::SortTime(const int colidx, const int64_t bulk_start,
  const int isd_offset, const int isd_size, int64_t* const localptr){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* tmp_in; tuple_t* tmp_out;
  posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_size* sizeof(tuple_t));
  posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_size* sizeof(tuple_t));
  memset(tmp_in, 0, isd_size* sizeof(tuple_t));
  for (int dx = 0; dx < isd_size; ++dx)
  {
    const int64_t glb_row_no = bulk_start + localptr[isd_offset + dx] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx);
    const TimestampValue* tsvalue_ = reinterpret_cast<const TimestampValue*>(vdata_);
    int64_t i64value = 0;
    if( tsvalue_ -> HasTime()){
      const boost::posix_time::time_duration& time_ = tsvalue_ -> time() ;
      i64value = time_.total_nanoseconds();
    }
    set_ptrkey_i64_hi32(tmp_in[dx], localptr[isd_offset + dx] , i64value);
  }

  avxsort_tuples(&tmp_in, &tmp_out, isd_size);
  for (int dx = 0; dx < isd_size; ++dx) {
    localptr[isd_offset + dx] = GetPtr(tmp_out[dx]);
  }
  free(tmp_in); free(tmp_out);

  int inner_start = 0;
  int inner_size = 1;

  for (int i = 1; i < isd_offset; ++i) {

    const int64_t pre_glb_row_no = bulk_start + localptr[isd_offset + i -1];
    const int64_t cur_glb_row_no = bulk_start + localptr[isd_offset + i ];
    void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, pre_glb_row_no, colidx);
    void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, cur_glb_row_no, colidx);
    const TimestampValue* pre_tsvalue_ = reinterpret_cast<const TimestampValue*>(predata);
    const TimestampValue* cur_tsvalue_ = reinterpret_cast<const TimestampValue*>(curdata);
    int64_t prerow_i64 = 0;
    int64_t currow_i64 = 0;

    if( pre_tsvalue_ -> HasTime()){
      const boost::posix_time::time_duration& time_ = pre_tsvalue_ -> time() ;
      prerow_i64 = time_.total_nanoseconds();
    }

    if( cur_tsvalue_ -> HasTime()){
      const boost::posix_time::time_duration& time_ = cur_tsvalue_ -> time() ;
      currow_i64 = time_.total_nanoseconds();
    }

    if ( compare_i64_hi32bit(prerow_i64, currow_i64) ) {
         ++inner_start;
         if( i != isd_offset - 1) continue;
    }

    if (inner_start > 1){
      tuple_t* inner_in; tuple_t* inner_out;
      posix_memalign((void**)&inner_in, CACHE_LINE_SIZE, inner_size* sizeof(tuple_t));
      posix_memalign((void**)&inner_out, CACHE_LINE_SIZE, inner_size* sizeof(tuple_t));
      memset(inner_in, 0, inner_size* sizeof(tuple_t));
      for (int dx = 0; dx < inner_size; ++dx) {
        const int64_t glb_row_no = bulk_start + localptr[isd_offset + inner_start + dx];
        void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx);
        int64_t currow_i64 = *reinterpret_cast<const int64_t*>(vdata_);
        set_ptrkey_i64_lo32(inner_in[dx], localptr[isd_offset + inner_start + dx], currow_i64);
      }
      avxsort_tuples(&inner_in, &inner_out, inner_size);
      for (int dx = 0; dx < inner_size; ++dx) {
        localptr[isd_offset + inner_start + dx] = GetPtr(inner_out[dx]);
      }
      free(inner_in); free(inner_out);
    }
    inner_start = i;
    inner_size = 1;
  }


}


void Sorter::TupleSorter::SortTimestamp(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx );
    if( vdata == NULL){
       SetNull(in[i], colidx);
    } else {
      const TimestampValue* ts_value = reinterpret_cast<const TimestampValue*>(vdata);
      int i32date = 0; // WARNING: 32bit integer may be not long enough
      if( ts_value -> HasDate() ) {
        const boost::gregorian::date& date_ = ts_value -> date();
        i32date = date_.day_number();
      }else {
        i32date = INT_MAX;
      }
      SetKeyInt(in[i], i32date);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);

  int start_ = 0; int end_ = osd_itv_size;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx );
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end_ -1] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx );
    if( vdata_ == NULL ) --end_; else break;
  }

  int isd_itv_start = start_;
  int isd_itv_size = 1;

  for (int i = start_ + 1; i < end_; ++i) {

    const int64_t pre_glb_row_no = bulk_start + localptr[osd_itv_start + i -1] ;
    const int64_t cur_glb_row_no = bulk_start + localptr[osd_itv_start + i ];
    void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, pre_glb_row_no, colidx );
    void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, cur_glb_row_no, colidx );

    int pre_i32date = INT_MAX;
    int cur_i32date = INT_MAX;
    const TimestampValue* pre_ts_value = reinterpret_cast<const TimestampValue*>(predata);
    const TimestampValue* cur_ts_value = reinterpret_cast<const TimestampValue*>(curdata);
    if( pre_ts_value->HasDate() ) pre_i32date = (pre_ts_value -> date()).day_number();
    if( cur_ts_value->HasDate() ) cur_i32date = (cur_ts_value -> date()).day_number();

    if( pre_i32date == cur_i32date ) {
      ++isd_itv_size;
      if( i != end_ - 1) continue;
    }

    if (isd_itv_size > 1){
      const int isd_offset = osd_itv_start + isd_itv_start;
      SortTime(colidx, bulk_start, isd_offset, isd_itv_size, localptr );
    }
    isd_itv_start = i;
    isd_itv_size = 1;
  }

}


void Sorter::TupleSorter::SortFloat(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ) {

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    SetPtr(in[i], localptr[osd_itv_start + i]);
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, bulk_start + localptr[osd_itv_start + i], colidx );
    if(vdata == NULL){
      SetNull(in[i], colidx);
    } else {
      const float f32value_ = *reinterpret_cast<const float*>(vdata);
      SetKeyFloat(in[i], f32value_);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);

}

void Sorter::TupleSorter::SortDouble(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i) {
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx );
    if( vdata == NULL){
      SetPtr(in[i], localptr[osd_itv_start + i]); SetNull(in[i], colidx);
    } else {
      double d64value =  *reinterpret_cast<const double*>(vdata);
      set_ptrkey_for_d64_hi32bit(in[i], localptr[osd_itv_start + i], d64value);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);

  int start_ = 0; int end_ = osd_itv_size ;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx );
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end_ - 1];
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx );
    if( vdata_ == NULL ) --end_; else break;
  }

  int isd_itv_start = start_;
  int isd_itv_size = 1;
  for (int i = start_ + 1 ; i < end_; ++i) {

    const int64_t pre_glb_row_no = bulk_start + localptr[osd_itv_start + i - 1];
    const int64_t cur_glb_row_no = bulk_start + localptr[osd_itv_start + i];

    void* pre_data = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, pre_glb_row_no, colidx);
    void* cur_data = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, cur_glb_row_no, colidx);
    const double prerow_i64 =  *reinterpret_cast<const double*>(pre_data);
    const double currow_i64 =  *reinterpret_cast<const double*>(cur_data);

    if ( compare_d64_hi32bit( prerow_i64 , currow_i64) ) {
      ++isd_itv_size;
      if( i != end_ - 1 ) continue;
    }

    if (isd_itv_size > 1){
      tuple_t* tmp_in; tuple_t* tmp_out;
      posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      memset(tmp_in, 0, isd_itv_size * sizeof(tuple_t));

      for (int dx = 0; dx < isd_itv_size; ++dx) {
        const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + isd_itv_start + dx ] ;
        void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no , colidx);
        double currow_d64 = *reinterpret_cast<const double*>(vdata_);
        set_ptrkey_for_d64_lo20bit(tmp_in[dx], localptr[osd_itv_start + isd_itv_start + dx ] , currow_d64);
      }
      avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);
      for (int i = 0; i < isd_itv_size; ++i) {
        localptr[osd_itv_start + isd_itv_start + i] = GetPtr(tmp_out[i]);
      }
      free(in); free(out);
    }
    isd_itv_start = i;
    isd_itv_start = 1;
  }
}


void Sorter::TupleSorter::SortStringVarchar(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  int char_offset = 0;
  int char_length_max = 0;
  const int const_char_width = 4;

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata == NULL){
      SetNull(in[i], colidx);
    } else {
      const StringValue* strval = reinterpret_cast<const StringValue*>(vdata);
      if(strval->len > char_length_max) char_length_max = strval->len;
      int length_ = strval -> len - char_offset < const_char_width? strval -> len - char_offset : const_char_width;
      int i32key = trans_4char_to_int( strval->ptr , length_);
      SetKeyInt(in[i], i32key);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);


  int start_ = 0; int end_ = osd_itv_size ;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end_ -1] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) --end_; else break;
  }

  for (int dx = 1; dx < ceil(static_cast<double>(char_length_max)/ const_char_width); ++dx)
  {
    bool should_continue = false;
    int isd_itv_start = start_;
    int isd_itv_size = 1;
    //const int pre_4char_offset = char_offset;
    char_offset = char_offset + const_char_width;
    int dy = start_ + 1;
    for ( ; dy < end_; ++dy) {

      void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, bulk_start + localptr[osd_itv_start + dy -1], colidx);
      void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, bulk_start + localptr[osd_itv_start + dy], colidx);
      const StringValue* prestrval = reinterpret_cast<const StringValue*>(predata);
      const StringValue* curstrval = reinterpret_cast<const StringValue*>(curdata);
      if( prestrval->len < char_offset || curstrval->len < char_offset ) continue;

      if( impala::StringCompare(prestrval->ptr, prestrval->len, curstrval->ptr, curstrval->len, char_offset) == 0 ) {
        ++isd_itv_size;
        if( dy != end_ - 1 ) continue;
      }


      if (isd_itv_size > 1){
        should_continue = true;
        tuple_t* tmp_in; tuple_t* tmp_out;
        posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        memset(tmp_in, 0, isd_itv_size * sizeof(tuple_t));

        for (int dz = 0; dz < isd_itv_size; ++dz)
        {
          const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + isd_itv_start + dz ] ;
          void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
          const StringValue* curstrval = reinterpret_cast<const StringValue*>(vdata);
          int length_ = curstrval->len - char_offset < const_char_width? curstrval->len - char_offset : const_char_width;
          int currow_i32value = 0;
          if( length_ > 0) currow_i32value = trans_4char_to_int( curstrval->ptr + char_offset, length_ );

          SetPtr(tmp_in[dz], localptr[osd_itv_start + isd_itv_start + dz ]);
          SetKeyInt(tmp_in[dz], currow_i32value);
        }
        avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);

        for (int dz = 0; dz < isd_itv_size; ++dz)
          localptr[osd_itv_start + isd_itv_start + dz ] = GetPtr(tmp_out[dz]);
        free(tmp_in); free(tmp_out);
      }

      isd_itv_start = dy;
      isd_itv_size = 1;

    }

    if( !should_continue ) break;

  }

}

void Sorter::TupleSorter::SortChar(const int colidx, const ColumnType& type, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();
  int char_offset = 0;
  int char_length_max = 0;
  const int const_char_width = 4;

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata == NULL){
       SetNull(in[i], colidx);
    } else {
      const char* str_ptr = StringValue::CharSlotToPtr(vdata, type);
      const int64_t len_ = StringValue::UnpaddedCharLength(str_ptr, type.len);
      if(len_ > char_length_max) char_length_max = len_;
      const int length_ = len_ - char_offset < const_char_width? len_ - char_offset : const_char_width;
      const int i32key = trans_4char_to_int( str_ptr , length_);
      SetKeyInt(in[i], i32key);
    }
  }

  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);

  int start_ = 0; int end_ = osd_itv_size ;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_  ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end_ -1] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) --end_; else break;
  }


  for (int dx = 1; dx < ceil((double) char_length_max / const_char_width); ++dx)
  {
    bool should_continue = false;
    int isd_itv_start = start_;
    int isd_itv_size = 1;
    //const int pre_4char_offset = char_offset;
    char_offset = char_offset + const_char_width;
    for (int dy = start_ + 1; dy < end_; ++dy) {

      void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, bulk_start + localptr[osd_itv_start + dy -1], colidx);
      void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, bulk_start + localptr[osd_itv_start + dy], colidx);
      const char* const cur_str_ptr = StringValue::CharSlotToPtr(curdata, type);
      const char* const pre_str_ptr = StringValue::CharSlotToPtr(predata, type);
      const int pre_len_ = StringValue::UnpaddedCharLength(pre_str_ptr, type.len);
      const int cur_len_ = StringValue::UnpaddedCharLength(cur_str_ptr, type.len);
      if( pre_len_ < char_offset || cur_len_< char_offset ) continue;
      if( impala::StringCompare(cur_str_ptr, pre_len_, pre_str_ptr, cur_len_, char_offset) == 0 ) {
        ++isd_itv_size;
        if ( dy != end_ - 1 ) continue;
      }

      if (isd_itv_size > 1){
        should_continue = true;
        tuple_t* tmp_in; tuple_t* tmp_out;
        posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        memset(tmp_in, 0, isd_itv_size * sizeof(tuple_t));

        for (int dz = 0; dz < isd_itv_size; dz++) {

          const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + isd_itv_start + dz ] ;
          void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
          const char* str_ptr = StringValue::CharSlotToPtr(vdata_, type);
          const int64_t len_ = StringValue::UnpaddedCharLength(str_ptr, type.len);
          const int length_ = len_ - char_offset < const_char_width? len_ - char_offset : const_char_width;
          int currow_i32value = 0;
          if( length_ > 0) currow_i32value = trans_4char_to_int( str_ptr + char_offset, length_ );
          SetPtr(tmp_in[dz], localptr[osd_itv_start + isd_itv_start + dz ]);
          SetKeyInt(tmp_in[dz], currow_i32value);
        }
        avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);
        for (int dz = 0; dz < isd_itv_size; ++dz)
          localptr[osd_itv_start + isd_itv_start + dz ] = GetPtr(tmp_out[dz]);
        free(tmp_in); free(tmp_out);
      }
      isd_itv_start = dy;
      isd_itv_size = 1;
    }

    if( !should_continue ) break;
  }
}

void Sorter::TupleSorter::SortDecimal4Value(const int width, const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){
  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if (vdata == NULL) {
      SetNull(in[i], colidx);
    } else {
      const Decimal4Value dec4val =  *reinterpret_cast<const Decimal4Value*>(vdata);
      SetKeyInt(in[i], (int32_t)dec4val.value());
    }
  }
  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i)
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  free(in); free(out);
}


void Sorter::TupleSorter::SortDecimal8Value(const int width, const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){
  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i) {
    SetPtr(in[i], localptr[osd_itv_start + i]);
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    const void* const vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata == NULL){
      SetNull(in[i], colidx);
    } else {
      const Decimal8Value dec8val =  *reinterpret_cast<const Decimal8Value*>(vdata);
      set_ptrkey_i64_hi32(in[i], localptr[osd_itv_start + i], dec8val.value());
    }
  }
  avxsort_tuples(&in, &out, osd_itv_size);
  for (int i = 0; i < osd_itv_size; ++i) {
    localptr[osd_itv_start + i] = GetPtr(out[i]);
  }
  free(in); free(out);

  int start_ = 0; int end_ = osd_itv_size ;
  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[ osd_itv_start + start_] ;
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) ++start_; else break;
  }

  while( start_ < end_ ){
    const int64_t glb_row_no = bulk_start + localptr[ osd_itv_start + end_ - 1];
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) --end_; else break;
  }

  int isd_itv_start = start_;
  int isd_itv_size = 1;

  for (int i = start_ + 1; i < end_; ++i) {

    const int64_t pre_glb_row_no = bulk_start + localptr[osd_itv_start + i];
    const int64_t cur_glb_row_no = bulk_start + localptr[osd_itv_start + i-1];
    void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, pre_glb_row_no, colidx);
    void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, cur_glb_row_no, colidx);

    const Decimal8Value pre_dec8val =  *reinterpret_cast<const Decimal8Value*>(predata);
    const Decimal8Value cur_dec8val =  *reinterpret_cast<const Decimal8Value*>(curdata);
    const int64_t prerow_i64 = static_cast<int64_t>(pre_dec8val.value());
    const int64_t currow_i64 = static_cast<int64_t>(cur_dec8val.value());

    if ( compare_i64_hi32bit(prerow_i64 , currow_i64) ) {
         ++isd_itv_size;
         if( i != end_ - 1) continue;
    }

    if (isd_itv_size > 1){
      tuple_t* tmp_in; tuple_t* tmp_out;
      posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
      memset(tmp_in, 0, isd_itv_size * sizeof(tuple_t));

      for (int dx = 0; dx < isd_itv_size; ++dx)
      {
        const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + isd_itv_start + dx] ;
        void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
        const Decimal8Value dec8val =  *reinterpret_cast<const Decimal8Value*>(vdata_);
        set_ptrkey_i64_lo32(tmp_in[dx], localptr[osd_itv_start + isd_itv_start + dx] , dec8val.value());
      }

      avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);
      for (int i = 0; i < osd_itv_size; ++i) {
        localptr[osd_itv_start + isd_itv_start + i] = GetPtr(tmp_out[i]);
      }
      free(tmp_in); free(tmp_out);
    }
    isd_itv_start = i;
    isd_itv_size = 1;
  }
}


void Sorter::TupleSorter::SortDecimal16Value(const int width, const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){
  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<ExprContext*>& key_expr_ctxs_rhs_ = less_than_comp_.get_key_expr_ctxs_rhs_();

  tuple_t* in; tuple_t* out;
  posix_memalign((void**)&in, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  posix_memalign((void**)&out, CACHE_LINE_SIZE, osd_itv_size* sizeof(tuple_t));
  memset(in, 0, osd_itv_size * sizeof(tuple_t));

  for (int i = 0; i < osd_itv_size; ++i){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + i];
    void* vdata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata == NULL){
      SetPtr(in[i], localptr[osd_itv_start + i]); SetNull(in[i], colidx);
    } else {
      const Decimal16Value dec16val =  *reinterpret_cast<const Decimal16Value*>(vdata);
      const int128_t i128value = dec16val.value();
      set_ptrkey_i128_127_96(in[i], localptr[osd_itv_start + i], i128value);
    }
  }
  avxsort_tuples(&in, &out, osd_itv_size);
  for (int dx = 0; dx < osd_itv_size; ++dx) {
    localptr[osd_itv_start + dx] = GetPtr(out[dx]);
  }
  free(in); free(out);

  int start = 0; int end = osd_itv_size ;
  while( start < end ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + start ];
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) ++start; else break;
  }
  while( start < end ){
    const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + end -1 ];
    void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
    if( vdata_ == NULL ) --end; else break;
  }


  for (int phi = 0; phi < 3; ++phi)
  {
    int isd_itv_start = start;
    int isd_itv_size = 1;
    bool should_repeat = false;

    for (int i = start + 1; i < end; ++i) {
      void* predata = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, bulk_start + localptr[osd_itv_start + i - 1], colidx);
      void* curdata = GetDataByRowIdxColIdx(key_expr_ctxs_rhs_, bulk_start + localptr[osd_itv_start + i], colidx);

      const Decimal16Value pre_dec16val =  *reinterpret_cast<const Decimal16Value*>(predata);
      const Decimal16Value cur_dec16val =  *reinterpret_cast<const Decimal16Value*>(curdata);

      const int128_t prerow_i128 = pre_dec16val.value();
      const int128_t currow_i128 = cur_dec16val.value();

      bool switch_ = true;
      if (phi == 0) {
        switch_ = compare_i128_hi127_96bit(prerow_i128, currow_i128);
      } else if (phi == 1) {
        switch_ = compare_i128_hi127_64bit(prerow_i128, currow_i128);
      } else if (phi == 2) {
        switch_ = compare_i128_hi127_32bit(prerow_i128, currow_i128);
      }

      if ( switch_ ) {
        ++isd_itv_size;
        if( i != end - 1 ) continue;
      }

      if (isd_itv_size > 1){
        should_repeat = true;
        tuple_t* tmp_in; tuple_t* tmp_out;
        posix_memalign((void**)&tmp_in, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        posix_memalign((void**)&tmp_out, CACHE_LINE_SIZE, isd_itv_size* sizeof(tuple_t));
        memset(tmp_in, 0, isd_itv_size * sizeof(tuple_t));

        for (int dx = 0; dx < isd_itv_size; dx++)
        {
          const int64_t glb_row_no = bulk_start + localptr[osd_itv_start + isd_itv_start + dx] ;
          void* vdata_ = GetDataByRowIdxColIdx(key_expr_ctxs_lhs_, glb_row_no, colidx);
          const Decimal16Value dec16_val =  *reinterpret_cast<const Decimal16Value*>(vdata_);
          if (phi == 0) {
            set_ptrkey_i128_95_64(tmp_in[dx], localptr[osd_itv_start + isd_itv_start + dx], dec16_val.value());
          } else if (phi == 1) {
            set_ptrkey_i128_63_32(tmp_in[dx], localptr[osd_itv_start + isd_itv_start + dx], dec16_val.value());
          } else if (phi == 2) {
            set_ptrkey_i128_31_0(tmp_in[dx], localptr[osd_itv_start + isd_itv_start + dx], dec16_val.value());
          }

        }
        avxsort_tuples(&tmp_in, &tmp_out, isd_itv_size);
        for (int dx = 0; dx < isd_itv_size; ++dx) {
          localptr[osd_itv_start + isd_itv_start + dx] = GetPtr(tmp_out[dx]);
        }
        free(tmp_in); free(tmp_out);
      }
      isd_itv_start = i;
      isd_itv_size = 1;
    }

    if( !should_repeat ) break;
  }

}


void Sorter::TupleSorter::SortDecimal(const int width, const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){
  switch( width ){
    case 4: {
      SortDecimal4Value( width, colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return;
    }
    case 8: {
      SortDecimal8Value( width, colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return;
    }
    case 16: {
      SortDecimal16Value( width, colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return;
    }
  }
}

void Sorter::TupleSorter::SorterCore(const int colidx, const int64_t bulk_start,
  const int64_t osd_itv_start, const int64_t osd_itv_size, int64_t* const localptr ){

  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const ColumnType& type = key_expr_ctxs_lhs_[colidx]->root()->type();

  switch ( type.type ) {
    case TYPE_NULL: {
      return;
    }
    case TYPE_BOOLEAN:
    case TYPE_TINYINT:
    case TYPE_SMALLINT:
    case TYPE_INT: {
      SortInt32(colidx, bulk_start, osd_itv_start, osd_itv_size, localptr, type);
      return;
    }
    case TYPE_BIGINT: {
      SortInt64(colidx, bulk_start, osd_itv_start, osd_itv_size, localptr);
      return;
    }
    case TYPE_FLOAT: {
      SortFloat(colidx, bulk_start, osd_itv_start, osd_itv_size, localptr);
      return;
    }
    case TYPE_DOUBLE:
    {
      SortDouble(colidx, bulk_start, osd_itv_start, osd_itv_size, localptr);
      return;
    }
    case TYPE_STRING:
    case TYPE_VARCHAR:
    {
      SortStringVarchar( colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return ;
    }
    case TYPE_CHAR:
    {
      SortChar( colidx, type, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return ;
    }

    case TYPE_TIMESTAMP:
    {
      SortTimestamp(colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
      return;
    }
    case TYPE_DECIMAL:
    {
      switch (type.GetByteSize()) {
        case 4:
          SortDecimal(4, colidx, bulk_start, osd_itv_start, osd_itv_size,localptr );
          return;
        case 8:
          SortDecimal(8, colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
          return;
        case 16:
          SortDecimal(16, colidx, bulk_start, osd_itv_start, osd_itv_size, localptr );
          return;
        default:
          DCHECK(false) << "invalid TYPE_DECIMAL: " << type;
          return;
      }
      return;
    }
    default:
      DCHECK(false) << "invalid type: " << type.DebugString();
      return;
    };
}

bool Sorter::TupleSorter::SortNthColumn(int ColIdx, const int64_t bulk_start,
  const int bulk_size,  const bool asc_sort, int64_t* const localptr) {
  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ =
      less_than_comp_.get_key_expr_ctxs_lhs_();
  if (ColIdx == 0) {
    SorterCore(ColIdx, bulk_start, 0, bulk_size, localptr);
    if (!asc_sort) reverse(localptr, localptr + bulk_size);
    return true;

  } else {
    bool should_sort = false;
    int interval_start = 0;
    int interval_size = 1;

    for (int i = 1; i < bulk_size; ++i) {
      TupleIterator _pre_iter = TupleIterator(this, bulk_start + localptr[i - 1]);
      TupleIterator _cur_iter = TupleIterator(this, bulk_start + localptr[i]);
      TupleRow* _pre_row = reinterpret_cast<TupleRow*>(&_pre_iter.current_tuple_);
      TupleRow* _cur_row = reinterpret_cast<TupleRow*>(&_cur_iter.current_tuple_);
      if (less_than_comp_.prefix_equal_n(_cur_row, _pre_row, ColIdx)) {
        ++interval_size;
        if( i != bulk_size - 1) continue;
      }

      if (interval_size > 1) {
        should_sort = true;
        SorterCore(ColIdx, bulk_start, interval_start, interval_size, localptr);
        if (!asc_sort)  reverse(localptr + interval_start, localptr + interval_start + interval_size);
      }
      interval_start = i;
      interval_size = 1;
    }
    return should_sort;
  }
}


void Sorter::TupleSorter::Sort(Run* run) {
  LOG(INFO) << "[dbg] start sorting@" << __LINE__ ;
  const int bulk_size = 1 << 20;
  struct timeval tvstart, tv_inbulk_sort, tv_bulk_merge, tv_shuffle, tvend;
  run_ = run;
  const std::vector<ExprContext*>& key_expr_ctxs_lhs_ = less_than_comp_.get_key_expr_ctxs_lhs_();
  const std::vector<bool>& is_asc_ = less_than_comp_.get_is_asc_();
  bool UseDefaultSort = false;

  LOG(INFO) << "[dbg] start sorting@" << __LINE__ ;
  if (!CpuInfo::IsSupported(CpuInfo::AVX2)) UseDefaultSort = true;

 // UseDefaultSort = true; // override this setting for experiment;
  gettimeofday(&tvstart, NULL);
  const int64_t nitems = run_->num_tuples_;
  if (!UseDefaultSort) {
    int64_t* const global_ptr = new int64_t[nitems];
    LOG(INFO) << "[dbg] inbulk sorting @" << __LINE__ ;

    const int64_t bulk_num = (int64_t)ceil((double)nitems / bulk_size);
    for (int64_t i = 0; i < nitems; i++) global_ptr[i] = i;
    for (int64_t bulkcur = 0; bulkcur < bulk_num; ++bulkcur) {
      int col_idx = 0;
      const int64_t cur_bulk_start = bulkcur * bulk_size;
      const int cur_bulk_size = nitems - cur_bulk_start < bulk_size ? nitems - cur_bulk_start : bulk_size;
      int64_t* const localptr = new int64_t[cur_bulk_size];
      for (int dx = 0; dx < cur_bulk_size; ++dx) localptr[dx] = dx;
      const int ncolumns = key_expr_ctxs_lhs_.size();
      do {
        bool asc_sort = is_asc_[col_idx];
        bool NeedMoreSort = SortNthColumn(col_idx++, cur_bulk_start, cur_bulk_size, asc_sort, localptr);
        if (NeedMoreSort && (col_idx < ncolumns)) continue;
        else break;
      } while (col_idx < ncolumns);
      for (int dx = 0; dx < cur_bulk_size; ++dx) {
        global_ptr[cur_bulk_start + dx] = cur_bulk_start + (int64_t)localptr[dx];
      }
      delete localptr;
    }
    LOG(INFO) << "[dbg] inbulk sorting @" << __LINE__ ;
    gettimeofday(&tv_inbulk_sort, NULL);
    mycomparison com_obj;
    com_obj.set_comparator( &less_than_comp_ );
    com_obj.set_tuple_sorter( this );
    vector< std::pair<int64_t, int64_t> > idx_vector;
    std::make_heap(idx_vector.begin(), idx_vector.end(), com_obj);

    int64_t* const heap_out = new int64_t[nitems];
    int64_t* const heap_cur = new int64_t[bulk_num];
    memset(heap_cur, 0, sizeof(int64_t) * bulk_num);

    for (int64_t dx = 0; dx < bulk_num; ++dx) {
      const int64_t glb_row_no = global_ptr[dx * bulk_size];
      idx_vector.push_back( make_pair(glb_row_no, dx ) );
      std::push_heap(idx_vector.begin(), idx_vector.end(), com_obj);
     // ++heap_cur[dx];
    }

    for( int64_t dx = 0; dx < nitems; ++dx ){
      pair<int64_t, int64_t> glbrow_bulkno = idx_vector.front();
      heap_out[dx] = glbrow_bulkno.first;

      std::pop_heap(idx_vector.begin(), idx_vector.end(), com_obj);
      idx_vector.pop_back();

      int64_t which_bulk = glbrow_bulkno.second;
      heap_cur[which_bulk]++;

      int64_t bulk_bound;
      if( (which_bulk + 1) * bulk_size < nitems) bulk_bound = bulk_size ;
      else bulk_bound = nitems - which_bulk * bulk_size ;

      if( heap_cur[which_bulk] < bulk_bound ) {
        const int64_t glb_row_no = global_ptr[which_bulk * bulk_size + heap_cur[which_bulk]] ;
        idx_vector.push_back( make_pair(glb_row_no , which_bulk) );
        std::push_heap(idx_vector.begin(), idx_vector.end(), com_obj);
      }

    }
    gettimeofday(&tv_bulk_merge, NULL);
    LOG(INFO) << "[dbg] bulk merge done@" << __LINE__ ;
     // Final phase: shuffle

    uint8_t* temp_data_seg = (uint8_t*)malloc(tuple_size_ * nitems);
    for (int64_t i = 0; i < nitems; ++i) {
      uint8_t* src = TupleIterator(this, heap_out[i]).current_tuple_;
      memcpy(temp_data_seg + i * tuple_size_, src, tuple_size_);
    }
    for (int64_t i = 0; i < nitems; ++i) {
      uint8_t* dst = TupleIterator(this, i).current_tuple_;
      memcpy(dst, temp_data_seg + i * tuple_size_, tuple_size_);
    }

    free(temp_data_seg);
    LOG(INFO) << "[dbg] shuffle done@" << __LINE__ ;

    delete heap_out;
    delete heap_cur;
    delete global_ptr;
    gettimeofday(&tv_shuffle, NULL);
  } else {
    SortHelper(TupleIterator(this, 0), TupleIterator(this, run_->num_tuples_));
    LOG(INFO) << "[dbg] impala sort done@" << __LINE__ ;

  }
  LOG(INFO) << "[dbg] all done@" << __LINE__ ;

  run->is_sorted_ = true;
  gettimeofday(&tvend, NULL);
  if( UseDefaultSort ){
    long t_total = (tvend.tv_sec - tvstart.tv_sec) * 1000000 +
                   (tvend.tv_usec - tvstart.tv_usec);
    LOG(INFO) << "[dbg] time consumed in impala sort  " << t_total << " microsecond";
    LOG(INFO) << "[dbg] impala sort throughput  " <<  (double) nitems / (double)t_total * 1000000 << " tuples/second";

  }else {
    LOG(INFO) << "[dbg] sorting row number: " << run_->num_tuples_;
    long t_bulkstort = (tv_inbulk_sort.tv_sec - tvstart.tv_sec) * 1000000 +
                       (tv_inbulk_sort.tv_usec - tvstart.tv_usec);
    LOG(INFO) << "[dbg] time consumed in bulks sorting  " << t_bulkstort
              << " microsecond";

    long t_bulkmerge = (tv_bulk_merge.tv_sec - tv_inbulk_sort.tv_sec) * 1000000 +
                       (tv_bulk_merge.tv_usec - tv_inbulk_sort.tv_usec);
    LOG(INFO) << "[dbg] time consumed in bulks merging  " << t_bulkmerge
              << " microsecond";

    long t_shuffle = (tv_shuffle.tv_sec - tv_bulk_merge.tv_sec) * 1000000 +
                     (tv_shuffle.tv_usec - tv_bulk_merge.tv_usec);
    LOG(INFO) << "[dbg] time consumed in bulks shuffle  " << t_shuffle
              << " microsecond";

    long t_total = (tvend.tv_sec - tvstart.tv_sec) * 1000000 +
                   (tvend.tv_usec - tvstart.tv_usec);

    LOG(INFO) << "[dbg] time consumed in avx sort  " << t_total << " microsecond";
    LOG(INFO) << "[dbg] avx sort throughput  " <<  (double) nitems / (double)t_total * 1000000 << " tuples/second";



  }




 }

// Sort the sequence of tuples from [first, last).
// Begin with a sorted sequence of size 1 [first, first+1).
// During each pass of the outermost loop, add the next tuple (at position 'i') to
// the sorted sequence by comparing it to each element of the sorted sequence
// (reverse order) to find its correct place in the sorted sequence, copying tuples
// along the way.
void Sorter::TupleSorter::InsertionSort(const TupleIterator& first,
    const TupleIterator& last) {
  TupleIterator insert_iter = first;
  insert_iter.Next();
  for (; insert_iter.index_ < last.index_; insert_iter.Next()) {
    // insert_iter points to the tuple after the currently sorted sequence that must
    // be inserted into the sorted sequence. Copy to temp_tuple_row_ since it may be
    // overwritten by the one at position 'insert_iter - 1'
    memcpy(temp_tuple_buffer_, insert_iter.current_tuple_, tuple_size_);

    // 'iter' points to the tuple that temp_tuple_row_ will be compared to.
    // 'copy_to' is the where iter should be copied to if it is >= temp_tuple_row_.
    // copy_to always to the next row after 'iter'
    TupleIterator iter = insert_iter;
    iter.Prev();
    uint8_t* copy_to = insert_iter.current_tuple_;
    while (less_than_comp_(temp_tuple_row_,
        reinterpret_cast<TupleRow*>(&iter.current_tuple_))) {
      memcpy(copy_to, iter.current_tuple_, tuple_size_);
      copy_to = iter.current_tuple_;
      // Break if 'iter' has reached the first row, meaning that temp_tuple_row_
      // will be inserted in position 'first'
      if (iter.index_ <= first.index_) break;
      iter.Prev();
    }

    memcpy(copy_to, temp_tuple_buffer_, tuple_size_);
  }
}

Sorter::TupleSorter::TupleIterator Sorter::TupleSorter::Partition(TupleIterator first,
    TupleIterator last, Tuple* pivot) {
  // Copy pivot into temp_tuple since it points to a tuple within [first, last).
  memcpy(temp_tuple_buffer_, pivot, tuple_size_);

  last.Prev();
  while (true) {
    // Search for the first and last out-of-place elements, and swap them.
    while (less_than_comp_(reinterpret_cast<TupleRow*>(&first.current_tuple_),
        temp_tuple_row_)) {
      first.Next();
    }
    while (less_than_comp_(temp_tuple_row_,
        reinterpret_cast<TupleRow*>(&last.current_tuple_))) {
      last.Prev();
    }

    if (first.index_ >= last.index_) break;
    // Swap first and last tuples.
    Swap(first.current_tuple_, last.current_tuple_);

    first.Next();
    last.Prev();
  }

  return first;
}

void Sorter::TupleSorter::SortHelper(TupleIterator first, TupleIterator last) {
  if (UNLIKELY(state_->is_cancelled())) return;
  // Use insertion sort for smaller sequences.
  while (last.index_ - first.index_ > INSERTION_THRESHOLD) {
    TupleIterator iter(this, first.index_ + (last.index_ - first.index_) / 2);
    DCHECK(iter.current_tuple_ != NULL);
    // Partition() splits the tuples in [first, last) into two groups (<= pivot
    // and >= pivot) in-place. 'cut' is the index of the first tuple in the second group.
    TupleIterator cut = Partition(first, last,
        reinterpret_cast<Tuple*>(iter.current_tuple_));
    SortHelper(cut, last);
    last = cut;
    if (UNLIKELY(state_->is_cancelled())) return;
  }

  InsertionSort(first, last);
}

inline void Sorter::TupleSorter::Swap(uint8_t* left, uint8_t* right) {
  memcpy(swap_buffer_, left, tuple_size_);
  memcpy(left, right, tuple_size_);
  memcpy(right, swap_buffer_, tuple_size_);
}

// Sorter methods
Sorter::Sorter(const TupleRowComparator& compare_less_than,
    const vector<ExprContext*>& slot_materialize_expr_ctxs,
    RowDescriptor* output_row_desc, MemTracker* mem_tracker,
    RuntimeProfile* profile, RuntimeState* state)
  : state_(state),
    compare_less_than_(compare_less_than),
    in_mem_tuple_sorter_(NULL),
    block_mgr_(state->block_mgr()),
    block_mgr_client_(NULL),
    has_var_len_slots_(false),
    sort_tuple_slot_expr_ctxs_(slot_materialize_expr_ctxs),
    mem_tracker_(mem_tracker),
    output_row_desc_(output_row_desc),
    unsorted_run_(NULL),
    profile_(profile),
    initial_runs_counter_(NULL),
    num_merges_counter_(NULL),
    in_mem_sort_timer_(NULL),
    sorted_data_size_(NULL) {
}

Sorter::~Sorter() {
  // Delete blocks from the block mgr.
  for (deque<Run*>::iterator it = sorted_runs_.begin();
      it != sorted_runs_.end(); ++it) {
    (*it)->DeleteAllBlocks();
  }
  for (deque<Run*>::iterator it = merging_runs_.begin();
      it != merging_runs_.end(); ++it) {
    (*it)->DeleteAllBlocks();
  }
  if (unsorted_run_ != NULL) unsorted_run_->DeleteAllBlocks();
  block_mgr_->ClearReservations(block_mgr_client_);
}

Status Sorter::Init() {
  DCHECK(unsorted_run_ == NULL) << "Already initialized";
  TupleDescriptor* sort_tuple_desc = output_row_desc_->tuple_descriptors()[0];
  has_var_len_slots_ = sort_tuple_desc->HasVarlenSlots();
  in_mem_tuple_sorter_.reset(new TupleSorter(compare_less_than_,
      block_mgr_->max_block_size(), sort_tuple_desc->byte_size(), state_));
  unsorted_run_ = obj_pool_.Add(new Run(this, sort_tuple_desc, true));

  initial_runs_counter_ = ADD_COUNTER(profile_, "InitialRunsCreated", TUnit::UNIT);
  num_merges_counter_ = ADD_COUNTER(profile_, "TotalMergesPerformed", TUnit::UNIT);
  in_mem_sort_timer_ = ADD_TIMER(profile_, "InMemorySortTime");
  sorted_data_size_ = ADD_COUNTER(profile_, "SortDataSize", TUnit::BYTES);

  int min_blocks_required = BLOCKS_REQUIRED_FOR_MERGE;
  // Fixed and var-length blocks are separate, so we need BLOCKS_REQUIRED_FOR_MERGE
  // blocks for both if there is var-length data.
  if (output_row_desc_->tuple_descriptors()[0]->HasVarlenSlots()) {
    min_blocks_required *= 2;
  }
  RETURN_IF_ERROR(block_mgr_->RegisterClient(min_blocks_required, mem_tracker_, state_,
      &block_mgr_client_));

  DCHECK(unsorted_run_ != NULL);
  RETURN_IF_ERROR(unsorted_run_->Init());
  return Status::OK();
}

Status Sorter::AddBatch(RowBatch* batch) {
  DCHECK(unsorted_run_ != NULL);
  DCHECK(batch != NULL);
  int num_processed = 0;
  int cur_batch_index = 0;
  while (cur_batch_index < batch->num_rows()) {
    if (has_var_len_slots_) {
      RETURN_IF_ERROR(unsorted_run_->AddBatch<true>(
          batch, cur_batch_index, &num_processed));
    } else {
      RETURN_IF_ERROR(unsorted_run_->AddBatch<false>(
          batch, cur_batch_index, &num_processed));
    }
    cur_batch_index += num_processed;
    if (cur_batch_index < batch->num_rows()) {
      // The current run is full. Sort it and begin the next one.
      RETURN_IF_ERROR(SortRun());
      RETURN_IF_ERROR(sorted_runs_.back()->UnpinAllBlocks());
      unsorted_run_ = obj_pool_.Add(
          new Run(this, output_row_desc_->tuple_descriptors()[0], true));
      RETURN_IF_ERROR(unsorted_run_->Init());
    }
  }
  return Status::OK();
}

Status Sorter::InputDone() {
  // Sort the tuples accumulated so far in the current run.
  RETURN_IF_ERROR(SortRun());

  if (sorted_runs_.size() == 1) {
    // The entire input fit in one run. Read sorted rows in GetNext() directly
    // from the sorted run.
    RETURN_IF_ERROR(sorted_runs_.back()->PrepareRead());
  } else {
    // At least one merge is necessary.
    int blocks_per_run = has_var_len_slots_ ? 2 : 1;
    int min_buffers_for_merge = sorted_runs_.size() * blocks_per_run;
    // Check if the final run needs to be unpinned.
    bool unpinned_final = false;
    if (block_mgr_->num_free_buffers() < min_buffers_for_merge - blocks_per_run) {
      // Number of available buffers is less than the size of the final run and
      // the buffers needed to read the remainder of the runs in memory.
      // Unpin the final run.
      RETURN_IF_ERROR(sorted_runs_.back()->UnpinAllBlocks());
      unpinned_final = true;
    } else {
      // No need to unpin the current run. There is enough memory to stream the
      // other runs.
      // TODO: revisit. It might be better to unpin some from this run if it means
      // we can get double buffering in the other runs.
    }

    // For an intermediate merge, intermediate_merge_batch contains deep-copied rows from
    // the input runs. If (unmerged_sorted_runs_.size() > max_runs_per_final_merge),
    // one or more intermediate merges are required.
    // TODO: Attempt to allocate more memory before doing intermediate merges. This may
    // be possible if other operators have relinquished memory after the sort has built
    // its runs.
    if (min_buffers_for_merge > block_mgr_->available_allocated_buffers()) {
      DCHECK(unpinned_final);
      RETURN_IF_ERROR(MergeIntermediateRuns());
    }

    // Create the final merger.
    CreateMerger(sorted_runs_.size());
  }
  return Status::OK();
}

Status Sorter::GetNext(RowBatch* output_batch, bool* eos) {
  if (sorted_runs_.size() == 1) {
    DCHECK(sorted_runs_.back()->is_pinned_);
    // In this case, only TupleRows are copied into output_batch. Sorted tuples are left
    // in the pinned blocks in the single sorted run.
    RETURN_IF_ERROR(sorted_runs_.back()->GetNext<false>(output_batch, eos));
    if (*eos) sorted_runs_.back()->TransferResources(output_batch);
  } else {
    // In this case, rows are deep copied into output_batch.
    RETURN_IF_ERROR(merger_->GetNext(output_batch, eos));
  }
  return Status::OK();
}

Status Sorter::Reset() {
  merger_.reset();
  merging_runs_.clear();
  sorted_runs_.clear();
  obj_pool_.Clear();
  DCHECK(unsorted_run_ == NULL);
  unsorted_run_ = obj_pool_.Add(
      new Run(this, output_row_desc_->tuple_descriptors()[0], true));
  RETURN_IF_ERROR(unsorted_run_->Init());
  return Status::OK();
}

Status Sorter::SortRun() {
  BufferedBlockMgr::Block* last_block = unsorted_run_->fixed_len_blocks_.back();
  if (last_block->valid_data_len() > 0) {
    sorted_data_size_->Add(last_block->valid_data_len());
  } else {
    RETURN_IF_ERROR(last_block->Delete());
    unsorted_run_->fixed_len_blocks_.pop_back();
  }
  if (has_var_len_slots_) {
    DCHECK(unsorted_run_->var_len_copy_block_ != NULL);
    last_block = unsorted_run_->var_len_blocks_.back();
    if (last_block->valid_data_len() > 0) {
      sorted_data_size_->Add(last_block->valid_data_len());
    } else {
      RETURN_IF_ERROR(last_block->Delete());
      unsorted_run_->var_len_blocks_.pop_back();
      if (unsorted_run_->var_len_blocks_.size() == 0) {
        RETURN_IF_ERROR(unsorted_run_->var_len_copy_block_->Delete());
        unsorted_run_->var_len_copy_block_ = NULL;
      }
    }
  }
  {
    SCOPED_TIMER(in_mem_sort_timer_);
    in_mem_tuple_sorter_->Sort(unsorted_run_);
    RETURN_IF_CANCELLED(state_);
  }
  sorted_runs_.push_back(unsorted_run_);
  unsorted_run_ = NULL;
  return Status::OK();
}

uint64_t Sorter::EstimateMergeMem(uint64_t available_blocks,
    RowDescriptor* row_desc, int merge_batch_size) {
  bool has_var_len_slots = row_desc->tuple_descriptors()[0]->HasVarlenSlots();
  int blocks_per_run = has_var_len_slots ? 2 : 1;
  int max_input_runs_per_merge = (available_blocks / blocks_per_run) - 1;
  // During a merge, the batches corresponding to the input runs contain only TupleRows.
  // (The data itself is in pinned blocks held by the run)
  uint64_t input_batch_mem =
      merge_batch_size * sizeof(Tuple*) * max_input_runs_per_merge;
  // Since rows are deep copied into the output batch for the merger, use a pessimistic
  // estimate of the memory required.
  uint64_t output_batch_mem = RowBatch::AT_CAPACITY_MEM_USAGE;

  return input_batch_mem + output_batch_mem;
}

Status Sorter::MergeIntermediateRuns() {
  int blocks_per_run = has_var_len_slots_ ? 2 : 1;
  int max_runs_per_final_merge =
      block_mgr_->available_allocated_buffers() / blocks_per_run;

  // During an intermediate merge, blocks from the output sorted run will have to be
  // pinned.
  int max_runs_per_intermediate_merge = max_runs_per_final_merge - 1;
  DCHECK_GT(max_runs_per_intermediate_merge, 1);
  // For an intermediate merge, intermediate_merge_batch contains deep-copied rows from
  // the input runs. If (sorted_runs_.size() > max_runs_per_final_merge),
  // one or more intermediate merges are required.
  scoped_ptr<RowBatch> intermediate_merge_batch;
  while (sorted_runs_.size() > max_runs_per_final_merge) {
    // An intermediate merge adds one merge to unmerged_sorted_runs_.
    // Merging 'runs - (max_runs_final_ - 1)' number of runs is sifficient to guarantee
    // that the final merge can be performed.
    int num_runs_to_merge = min<int>(max_runs_per_intermediate_merge,
        sorted_runs_.size() - max_runs_per_intermediate_merge);
    CreateMerger(num_runs_to_merge);
    RowBatch intermediate_merge_batch(*output_row_desc_, state_->batch_size(),
        mem_tracker_);
    // merged_run is the new sorted run that is produced by the intermediate merge.
    Run* merged_run = obj_pool_.Add(
        new Run(this, output_row_desc_->tuple_descriptors()[0], false));
    RETURN_IF_ERROR(merged_run->Init());
    bool eos = false;
    while (!eos) {
      // Copy rows into the new run until done.
      int num_copied;
      RETURN_IF_CANCELLED(state_);
      RETURN_IF_ERROR(merger_->GetNext(&intermediate_merge_batch, &eos));
      Status ret_status;
      if (has_var_len_slots_) {
        ret_status = merged_run->AddBatch<true>(&intermediate_merge_batch,
            0, &num_copied);
      } else {
        ret_status = merged_run->AddBatch<false>(&intermediate_merge_batch,
            0, &num_copied);
      }
      if (!ret_status.ok()) return ret_status;

      DCHECK_EQ(num_copied, intermediate_merge_batch.num_rows());
      intermediate_merge_batch.Reset();
    }

    BufferedBlockMgr::Block* last_block = merged_run->fixed_len_blocks_.back();
    if (last_block->valid_data_len() > 0) {
      RETURN_IF_ERROR(last_block->Unpin());
    } else {
      RETURN_IF_ERROR(last_block->Delete());
      merged_run->fixed_len_blocks_.pop_back();
    }
    if (has_var_len_slots_) {
      last_block = merged_run->var_len_blocks_.back();
      if (last_block->valid_data_len() > 0) {
        RETURN_IF_ERROR(last_block->Unpin());
      } else {
        RETURN_IF_ERROR(last_block->Delete());
        merged_run->var_len_blocks_.pop_back();
      }
    }
    merged_run->is_pinned_ = false;
    sorted_runs_.push_back(merged_run);
  }

  return Status::OK();
}

Status Sorter::CreateMerger(int num_runs) {
  DCHECK_GT(num_runs, 1);

  // Clean up the runs from the previous merge.
  for (deque<Run*>::iterator it = merging_runs_.begin();
      it != merging_runs_.end(); ++it) {
    (*it)->DeleteAllBlocks();
  }
  merging_runs_.clear();
  merger_.reset(
      new SortedRunMerger(compare_less_than_, output_row_desc_, profile_, true));

  vector<function<Status (RowBatch**)> > merge_runs;
  merge_runs.reserve(num_runs);
  for (int i = 0; i < num_runs; ++i) {
    Run* run = sorted_runs_.front();
    RETURN_IF_ERROR(run->PrepareRead());
    // Run::GetNextBatch() is used by the merger to retrieve a batch of rows to merge
    // from this run.
    merge_runs.push_back(bind<Status>(mem_fn(&Run::GetNextBatch), run, _1));
    sorted_runs_.pop_front();
    merging_runs_.push_back(run);
  }
  RETURN_IF_ERROR(merger_->Prepare(merge_runs));

  num_merges_counter_->Add(1);
  return Status::OK();
}

} // namespace impala
