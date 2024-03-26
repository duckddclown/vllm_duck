from typing import Optional
import heapq
from vllm.block import BlockTable, PhysicalTokenBlock

POOLSIZE = 100

class prefix_dict_list:
    def __init__(
        self,
    ) -> None:
        self.prefix_length_heap = []
        self.prefix_length_dict = {}
    
    def add_block(self, block: PhysicalTokenBlock) -> None:
        prefix_length = block.prefix_length
        if (prefix_length in self.prefix_length_dict):
            self.prefix_length_dict[prefix_length].append(block)
        else:
            self.prefix_length_dict[prefix_length] = [block]
            heapq.heappush(self.prefix_length_heap, -prefix_length)
        return
    
    def pop_block(self) -> PhysicalTokenBlock:
        prefix_length = - self.prefix_length_heap[0]
        block_list = self.prefix_length_dict[prefix_length]
        block = block_list[0]
        del block_list[0]

        if (len(block_list) == 0):
            del self.prefix_length_dict[prefix_length]
            heapq.heappop(self.prefix_length_heap)

        return block
    
    def remove_block(self, block: PhysicalTokenBlock) -> bool:
        prefix_length = block.prefix_length
        if (prefix_length not in self.prefix_length_dict):
            return False
        block_list = self.prefix_length_dict[prefix_length]
        if (block not in block_list):
            return False
        block_list.remove(block)

        if (len(block_list) == 0):
            del self.prefix_length_dict[prefix_length]
            heapq.heappop(self.prefix_length_heap)

        return True
    
    def get_length(self) -> int:
        return len(self.prefix_length_heap)
    
class accessed_time_dict_list:
    def __init__(
        self,
    ) -> None:
        self.accessed_time_heap = []
        self.accessed_time_dict = {}

    def add_block(self, block: PhysicalTokenBlock) -> None:
        accessed_time = block.last_accessed_time
        if (accessed_time in self.accessed_time_dict):
            self.accessed_time_dict[accessed_time].add_block(block)
        else:
            self.accessed_time_dict[accessed_time] = prefix_dict_list()
            self.accessed_time_dict[accessed_time].add_block(block)
            heapq.heappush(self.accessed_time_heap, accessed_time)
        return
    
    def pop_block(self) -> PhysicalTokenBlock:
        accessed_time = self.accessed_time_heap[0]
        prefix_object: prefix_dict_list = self.accessed_time_dict[accessed_time]
        block = prefix_object.pop_block()

        if (prefix_object.get_length() == 0):
            del self.accessed_time_dict[accessed_time]
            heapq.heappop(self.accessed_time_heap)

        return block

    def remove_block(self, block:PhysicalTokenBlock) -> bool:
        accessed_time = block.last_accessed_time
        if (accessed_time not in self.accessed_time_dict):
            return False
        prefix_object: prefix_dict_list = self.accessed_time_dict[accessed_time]
        if (prefix_object.remove_block(block) == False):
            return False
        if (prefix_object.get_length() == 0):
            del self.accessed_time_dict[accessed_time]
            heapq.heappop(self.accessed_time_heap)
        
        return True

    
class CachePool:
    def __init__(
        self,
        pool_size = POOLSIZE
    ) -> None:
        self.pool_size = pool_size
        self.curr_pool_size = 0
        self.access_time_object = accessed_time_dict_list()
    
    def add_block(self, new_block: PhysicalTokenBlock) -> Optional[PhysicalTokenBlock]:
        if (self.curr_pool_size < self.pool_size):
            self.access_time_object.add_block(new_block)
            self.curr_pool_size += 1
            return None
        
        else:
            block = self.access_time_object.pop_block()
            self.access_time_object.add_block(new_block)
            return block
        
    def remove_block(self, block: PhysicalTokenBlock) -> None:
        if (self.access_time_object.remove_block(block)):
            self.curr_pool_size -= 1
        return

