"""A block manager that manages token blocks."""
import enum
import time
from typing import Dict, List, Optional, Set, Tuple

from vllm.block import BlockTable, PhysicalTokenBlock, HashTable
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.cache_policy import CachePool


class BlockAllocator:
    """Manages free physical token blocks for a device.

    The allocator maintains a list of free blocks and allocates a block when
    requested. When a block is freed, its reference count is decremented. If
    the reference count becomes zero, the block is added back to the free list.
    """

    def __init__(
        self,
        device: Device,
        block_size: int,
        num_blocks: int,
    ) -> None:
        self.device = device
        self.block_size = block_size
        self.num_blocks = num_blocks

        ## Initialize a Pool for the prefixing cache.
        self.cache_pool = CachePool()

        # Initialize the free blocks.
        self.free_blocks: BlockTable = []
        for i in range(num_blocks):
            block = PhysicalTokenBlock(device=device,
                                       block_number=i,
                                       block_size=block_size)
            self.free_blocks.append(block)

    def allocate(self) -> PhysicalTokenBlock:
        if not self.free_blocks:
            raise ValueError("Out of memory! No free blocks are available.")
        block = self.free_blocks.pop()
        block.ref_count = 1
        return block

    def free(self, block: PhysicalTokenBlock) -> Optional[int]:
        if block.ref_count == 0:
            raise ValueError(f"Double free! {block} is already freed.")
        block.ref_count -= 1
        if block.ref_count == 0:
            #print(self.cache_pool.curr_pool_size)
            ## Not update immediately. But wait until the pool is full.
            ret_block = self.cache_pool.add_block(block)
            if (ret_block != None):
                self.free_blocks.append(ret_block)
                return ret_block.hash_val
        
        return None


    def get_num_free_blocks(self) -> int:
        return len(self.free_blocks)


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class BlockSpaceManager:
    """Manages the mapping between logical and physical token blocks."""

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        ## For performance.
        self.total_seq = 0
        self.hit = 0

        self.block_sliding_window = None
        if sliding_window is not None:
            assert sliding_window % block_size == 0, (sliding_window,
                                                      block_size)
            self.block_sliding_window = sliding_window // block_size

        self.watermark = watermark
        assert watermark >= 0.0

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        self.gpu_allocator = BlockAllocator(Device.GPU, block_size,
                                            num_gpu_blocks)
        self.cpu_allocator = BlockAllocator(Device.CPU, block_size,
                                            num_cpu_blocks)

        # Mapping: seq_id -> BlockTable.
        self.block_tables: Dict[int, BlockTable] = {}

        ## Define the hash table, the hash table is global for entire sequence group.
        self.hash_table : HashTable = {}

        ## Per sequence hash list used for freeing. Mapping: seq_id -> Hash value list.
        self.hash_lists: Dict[int, List[int]] = {}


    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        # FIXME(woosuk): Here we assume that all sequences in the group share
        # the same prompt. This may not be true for preempted sequences.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_required_blocks = len(seq.logical_token_blocks)

        if seq_group.prefix is not None and seq_group.prefix.allocated:
            num_required_blocks -= seq_group.prefix.get_num_blocks()

        if self.block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.block_sliding_window)
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()

        # Use watermark to avoid frequent cache eviction.
        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        # NOTE: Here we assume that all sequences in the group have the same
        # prompt.
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]

        # Allocate new physical token blocks that will store the prompt tokens.
        num_prompt_blocks = len(seq.logical_token_blocks)

        block_table: BlockTable = []
        prefix_block_table: BlockTable = []
        num_prefix_blocks = 0

        ## Calculate the prefix and hash value for each sequence.
        ## Assume that all sequences in the group have the same prompt.
        logical_token_blocks = seq.logical_token_blocks
        hash_per_block = []
        prev_block = None
        for block in logical_token_blocks:
            if (prev_block != None):
                prev_token_ids = prev_block.get_token_ids()
                block.prefix = prev_block.prefix + prev_token_ids
            curr_token_ids = block.get_token_ids()
            curr_block_fix = block.prefix + curr_token_ids
            hash_val = hash(tuple(curr_block_fix))
            block.hash_val = hash_val
            hash_per_block.append(hash_val)
            prev_block = block

        prefix = seq_group.prefix
        if prefix is not None and prefix.allocated:
            # Prefix has already been allocated. Use the existing block table.
            num_prompt_blocks -= prefix.get_num_blocks()

            ## Include the index.
            for i, block in enumerate(prefix.block_table):
                block.ref_count += seq_group.num_seqs()
                if (block.prefix_length == -1):
                    block.prefix_length = i

                ## Update the hash table.
                hash_val_i = hash_per_block[i]
                if (hash_val_i not in self.hash_table):
                    block.hash_val = hash_val
                    self.hash_table[hash_val_i] = block
                else:
                    self.gpu_allocator.cache_pool.remove_block(block)
                block_table.append(block)

        ## Initialize the offset.
        i = 0
        if (prefix == None):
            base_len = 0
        else:
            base_len = prefix.get_num_blocks()

        for logical_idx in range(num_prompt_blocks):
            
            if (self.block_sliding_window is not None
                    and logical_idx >= self.block_sliding_window):
                block = block_table[logical_idx % self.block_sliding_window]
            else:
                ## Update the hash table.
                self.total_seq += 1
                hash_val_i = hash_per_block[i + base_len]
                if (hash_val_i not in self.hash_table):
                    block = self.gpu_allocator.allocate()
                    block.hash_val = hash_val_i
                    block.prefix_length = i + base_len
                    self.hash_table[hash_val_i] = block
                    block.ref_count = seq_group.num_seqs()
                else:
                    self.hit += 1
                    block = self.hash_table[hash_val_i]
                    self.gpu_allocator.cache_pool.remove_block(block)
                    block.ref_count += seq_group.num_seqs()
                    block.last_accessed_time = time.time()
        
            # Set the reference counts of the token blocks.
            block_table.append(block)

            i += 1
                
                

        if prefix is not None and not prefix.allocated:
            # Allocate blocks for the prefix, we will compute the prefix's
            # KV cache in this run.
            num_prefix_blocks = prefix.get_num_blocks()
            prefix_block_table = block_table[:num_prefix_blocks]
            for block in prefix_block_table:
                block.ref_count += 1
            prefix.set_block_table(prefix_block_table)

        # Assign the block table for each sequence.
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            self.hash_lists[seq.seq_id] = hash_per_block.copy()
            self.block_tables[seq.seq_id] = block_table.copy()

    def can_append_slot(self, seq_group: SequenceGroup) -> bool:
        # Simple heuristic: If there is at least one free block
        # for each sequence, we can append.
        num_free_gpu_blocks = self.gpu_allocator.get_num_free_blocks()
        num_seqs = seq_group.num_seqs(status=SequenceStatus.RUNNING)
        return num_seqs <= num_free_gpu_blocks

    def append_slot(self, seq: Sequence) -> Optional[Tuple[int, int]]:
        """Allocate a physical slot for a new token."""
        logical_blocks = seq.logical_token_blocks
        block_table = self.block_tables[seq.seq_id]

        # block_table = self.block_tables[seq.seq_id]
        ## Use the hash_list to replace block_table.
        hash_list = self.hash_lists[seq.seq_id]
        self.total_seq += 1

        if len(block_table) < len(logical_blocks):
            ## Needs research in sliding window.
            if (self.block_sliding_window
                    and len(block_table) >= self.block_sliding_window):
                # reuse a block
                block_table.append(block_table[len(block_table) %
                                               self.block_sliding_window])
            else:
                # The sequence has a new logical block.
                # Allocate a new physical block.

                ## Update the hash table and block information.
                last_block = seq.logical_token_blocks[-1]
                last_2_block = seq.logical_token_blocks[-2]
                token_ids = last_2_block.get_token_ids()
                last_block.prefix = last_2_block.prefix + token_ids
                hash_val = hash(tuple(last_block.prefix + last_block.get_token_ids()))
                last_block.hash_val = hash_val
                if (hash_val not in self.hash_table):
                    block = self.gpu_allocator.allocate()
                    block.hash_val = hash_val
                    block.prefix_length = len(hash_list)
                    self.hash_table[hash_val] = block

                else:
                    self.hit += 1
                    block = self.hash_table[hash_val]
                    self.gpu_allocator.cache_pool.remove_block(block)
                    block.ref_count += 1
                    block.last_accessed_time = time.time()

                self.hash_lists[seq.seq_id].append(hash_val)
                block_table.append(block)
                return None

        # We want to append the token to the last physical block.
        ## Now the way to obtain the last block is to use hash table.
        last_hash_val = hash_list[-1]
        last_block = self.hash_table[last_hash_val]
        last_virtual_block = seq.logical_token_blocks[-1]

        ## Calculate the hash value since the virtual block is updated.
        hash_val = hash(tuple(last_virtual_block.prefix + last_virtual_block.get_token_ids()))
        last_virtual_block.hash_val = hash_val

        ## Update the hash lists.
        hash_list[-1] = hash_val
        
        assert last_block.device == Device.GPU
        if last_block.ref_count == 1:
            # Not shared with other sequences. Appendable.

            ## Change key of the hash table.
            if (hash_val not in self.hash_table):
                self.hash_table[hash_val] = self.hash_table.pop(last_hash_val)
            else:
                self.hit += 1
                self.hash_table.pop(last_hash_val)
                new_block = self.hash_table[hash_val]
                new_block.ref_count += 1
                new_block.last_accessed_time = time.time()
                block_table[-1] = new_block
            last_block.hash_val = hash_val
            return None
        else:
            # The last block is shared with other sequences.
            # Copy on Write: Allocate a new block and copy the tokens.
            if (hash_val not in self.hash_table):
                new_block = self.gpu_allocator.allocate()
                new_block.prefix_length = last_block.prefix_length
                new_block.hash_val = hash_val
                self.hash_table[hash_val] = new_block
                block_table[-1] = new_block
                self.gpu_allocator.free(last_block)
                return last_block.block_number, new_block.block_number
            
            else:
                new_block = self.hash_table[hash_val]
                new_block.ref_count += 1
                new_block.last_accessed_time = time.time()
                self.gpu_allocator.cache_pool.remove_block(new_block)
                block_table[-1] = new_block
                self.gpu_allocator.free(last_block)
                return None

    ## Needs further implementation.
    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        # NOTE: fork does not allocate a new physical block.
        # Thus, it is always safe from OOM.
        src_block_table = self.block_tables[parent_seq.seq_id]
        self.block_tables[child_seq.seq_id] = src_block_table.copy()
        for block in src_block_table:
            block.ref_count += 1

    def _get_physical_blocks(
            self, seq_group: SequenceGroup) -> List[PhysicalTokenBlock]:
        # NOTE: Here, we assume that the physical blocks are only shared by
        # the sequences in the same group.
        blocks: Set[PhysicalTokenBlock] = set()
        for seq in seq_group.get_seqs():
            if seq.is_finished():
                continue
            blocks.update(self.block_tables[seq.seq_id])
        return list(blocks)

    def can_swap_in(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        num_swapped_seqs = seq_group.num_seqs(status=SequenceStatus.SWAPPED)
        num_free_blocks = self.gpu_allocator.get_num_free_blocks()
        # NOTE: Conservatively, we assume that every sequence will allocate
        # at least one free block right after the swap-in.
        # NOTE: This should match the logic in can_append_slot().
        num_required_blocks = len(blocks) + num_swapped_seqs
        return num_free_blocks - num_required_blocks >= self.watermark_blocks

    def swap_in(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # CPU block -> GPU block.
        if seq_group.prefix is not None:
            # make sure to swap in the prefix first
            assert seq_group.prefix.allocated and seq_group.prefix.computed

        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]
            if seq_group.prefix is not None:
                for block in seq_group.prefix.block_table:
                    new_block_table.append(block)
                    block.ref_count += 1

            ## Change the original cpu physical block to gpu on hash table.
            hash_list = self.hash_lists[seq.seq_id]
            for i, cpu_block in enumerate(block_table):
                hash_val = hash_list[i]
                if cpu_block in mapping:
                    gpu_block = mapping[cpu_block]
                    gpu_block.ref_count += 1
                    
                else:
                    if (hash_val not in self.hash_table):
                        gpu_block = self.gpu_allocator.allocate()
                        gpu_block.prefix_length = cpu_block.prefix_length
                        gpu_block.hash_val = cpu_block.hash_val
                    else:
                        gpu_block = self.hash_table[hash_val]
                        gpu_block.ref_count += 1
                        self.gpu_allocator.cache_pool.remove_block(gpu_block)

                    mapping[cpu_block] = gpu_block
                new_block_table.append(gpu_block)
                # Free the CPU block swapped in to GPU.
                self.cpu_allocator.free(cpu_block)

                hash_val = hash_list[i]
                self.hash_table[hash_val] = gpu_block

            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            cpu_block.block_number: gpu_block.block_number
            for cpu_block, gpu_block in mapping.items()
        }
        return block_number_mapping

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        blocks = self._get_physical_blocks(seq_group)
        return len(blocks) <= self.cpu_allocator.get_num_free_blocks()

    def swap_out(self, seq_group: SequenceGroup) -> Dict[int, int]:
        # GPU block -> CPU block.
        mapping: Dict[PhysicalTokenBlock, PhysicalTokenBlock] = {}
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            new_block_table: BlockTable = []
            block_table = self.block_tables[seq.seq_id]

            hash_list = self.hash_lists[seq.seq_id]

            for i, gpu_block in enumerate(block_table):
                hash_val = hash_list[i]
                if (seq_group.prefix is not None
                        and gpu_block in seq_group.prefix.block_table):
                    # NOTE: We do not swap out the prefix blocks for now.
                    self.gpu_allocator.free(gpu_block)
                    continue
                
                ## Change the original gpu physical block to cpu on hash table.
                if gpu_block in mapping:
                    cpu_block = mapping[gpu_block]
                    cpu_block.ref_count += 1
                else:
                    if (hash_val not in self.hash_table):
                        cpu_block = self.cpu_allocator.allocate()
                        cpu_block.prefix_length = gpu_block.prefix_length
                        cpu_block.hash_val = gpu_block.hash_val
                    else:
                        cpu_block = self.hash_table[hash_val]
                        cpu_block.ref_count += 1
                        self.cpu_allocator.cache_pool.remove_block(cpu_block)
                    mapping[gpu_block] = cpu_block
                new_block_table.append(cpu_block)
                # Free the GPU block swapped out to CPU.
                self.gpu_allocator.free(gpu_block)

                hash_val = hash_list[i]
                self.hash_table[hash_val] = cpu_block

            self.block_tables[seq.seq_id] = new_block_table

        block_number_mapping = {
            gpu_block.block_number: cpu_block.block_number
            for gpu_block, cpu_block in mapping.items()
        }
        return block_number_mapping

    ## Totally Change this function.
    def _free_block_table(self, block_table: BlockTable) -> Optional[int]:
        hash_val_list = []
        for block in set(block_table):
            if block.device == Device.GPU:
                hash_val = self.gpu_allocator.free(block)
            else:
                hash_val = self.cpu_allocator.free(block)
            
            if (hash_val != None):
                hash_val_list.append(hash_val)
        return hash_val_list



    def _free_hash_table(self, hash_list) -> None:
        for hash_val in hash_list:
            self.hash_table.pop(hash_val)

    def free(self, seq: Sequence) -> None:
        if seq.seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return
        block_table = self.block_tables[seq.seq_id]
        ## Free the hash_lists and hash_table.
        real_hash_list = self._free_block_table(block_table)
        if (len(real_hash_list) > 0):
            self._free_hash_table(real_hash_list)
        del self.block_tables[seq.seq_id]
        del self.hash_lists[seq.seq_id]

    def reset(self) -> None:
        for block_table in self.block_tables.values():
            real_hash_list = self._free_block_table(block_table)
            if (len(real_hash_list) > 0):
                self._free_hash_table(real_hash_list)
        ## Clean the hash table and hash lists.
        self.block_tables.clear()
        self.hash_lists.clear()

    def get_block_table(self, seq: Sequence) -> List[int]:
        block_table = self.block_tables[seq.seq_id]
        return [block.block_number for block in block_table]

    def get_num_free_gpu_blocks(self) -> int:
        return self.gpu_allocator.get_num_free_blocks()

    def get_num_free_cpu_blocks(self) -> int:
        return self.cpu_allocator.get_num_free_blocks()
