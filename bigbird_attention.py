import sys
import numpy as np
from attention import get_callback

from blocksparse.transformer import BlocksparseTransformer
#from blocksparse import BlocksparseTransformer

batch_size = 16
num_attention_heads = 1
size_per_head = 512
from_seq_length = 4096
to_seq_length = 4096

num_rand_blocks = 3
from_block_size = 64
to_block_size = 64

blocksparse_bs = 32

def bigbird_layout(from_seq_length, to_seq_length, from_block_size, 
                    to_block_size, num_rand_blocks, num_attention_heads,
                    blocksparse_bs):
    
    raise NotImplementedError

def bigbird_callback():
    def mask_in_each_blocksparse_block(blk_shape, head_idx, q_idx, k_idx, blk_idx, 
                                        from_block_size, to_block_size):
        """
        given the bigbird attention patterns with from/to block size,
        compute mask for each blocksparse block 
        """
        raise NotImplementedError
    return mask_in_each_blocksparse_block

if __name__ == '__main__':
    is_fp16 = len(sys.argv) > 1 and sys.argv[1] == 'fp16'
    dtype = tf.float16 if is_fp16 else tf.float32
    q = tf.random_normal(shape=[batch_size, num_attention_heads, from_seq_length, size_per_head], dtype = dtype)
    k = tf.random_normal(shape=[batch_size, num_attention_heads, to_seq_length, size_per_head], dtype = dtype)
    v = tf.random_normal(shape=[batch_size, num_attention_heads, to_seq_length, size_per_head], dtype = dtype)

    layout = bigbird_layout(from_seq_length, to_seq_length, from_block_size, to_block_size,
                            num_rand_blocks, num_attention_heads, blocksparse_bs)
    bst = BlocksparseTransformer(layout, block_size = blocksparse_bs, 
                                            mask_callback=bigbird_callback(),
                                            heads = num_attention_heads)