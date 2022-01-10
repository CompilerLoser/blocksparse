# sparse_attention(https://github.com/openai/sparse_attention)
# uses blocksparse to implement the three basic sparse patterns, 
# we follow it to implement blocked attentions in bigbird.
# For now we just consider the basic condition: the attention 
# are blocked in a same size with blocksparse compute blocksize. 

import sys
import numpy as np
import tensorflow as tf
import time

from blocksparse.transformer import BlocksparseTransformer

def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
  """Create adjacency list of random attention.

  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.

  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
  rand_attn = np.zeros(
      (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32)
  middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
  last = to_seq_length // to_block_size - 1
  if last_idx > (2 * to_block_size):
    last = (last_idx // to_block_size) - 1

  r = num_rand_blocks  # shorthand
  for i in range(1, from_seq_length // from_block_size-1):
    start = i-2
    end = i
    if i == 1:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[2:last])[:r]
    elif i == 2:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[3:last])[:r]
    elif i == from_seq_length // from_block_size - 3:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -3: should have been sliced till last-3
    elif i == from_seq_length // from_block_size - 2:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -4: should have been sliced till last-4
    else:
      if start > last:
        start = last
        rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
      elif (end+1) == last:
        rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r]
      else:
        rand_attn[i-1, :] = np.random.permutation(
            np.concatenate((middle_seq[:start], middle_seq[end+1:last])))[:r]
  return rand_attn


def generate_rand_attn_list(from_seq_length, from_block_size,
         to_block_size, num_rand_blocks, num_attention_heads):
    # old plans used in paper
    if from_seq_length in [1024, 2048, 3072, 4096]:
        rand_attn = [
            bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
              from_seq_length, from_seq_length,# ? 
              from_block_size, to_block_size, num_rand_blocks,
              last_idx=from_seq_length
            )[:(from_seq_length // from_block_size - 2)]
            for _ in range(num_attention_heads)
        ]
    else:
        raise NotImplementedError
    rand_attn = np.stack(rand_attn, axis=0)
    return rand_attn

"""
def bigbird_layout(from_seq_length, to_seq_length, from_block_size, 
                    to_block_size, num_rand_blocks, num_attention_heads,
                    blocksparse_bs, rand_attn):
    assert from_seq_length == to_seq_length
    assert from_block_size == to_block_size
    assert from_seq_length % blocksparse_bs == 0
    blk_num = from_seq_length // blocksparse_bs

    layouts = []
    for head in range(num_attention_heads):
        layout = np.zeros([blk_num, blk_num], dtype=np.bool)
        for i in range(blk_num):
            if (i) * blocksparse_bs < from_block_size - 1 \
                or (i+1)*blocksparse_bs > (from_seq_length - from_block_size -1):
                layout[i,:] = 1
            else:
                layout[i,:] = 1
        layouts.append(layout)
    layouts = np.array(layouts)
    
    raise NotImplementedError

def bigbird_callback():
    def mask_in_each_blocksparse_block(blk_shape, head_idx, q_idx, k_idx, blk_idx, 
                                        from_block_size, to_block_size):
        #given the bigbird attention patterns with from/to block size,
        #compute mask for each blocksparse block 
        raise NotImplementedError
    return mask_in_each_blocksparse_block
"""


def bigbird_layout_simple(rand_attn):
    layouts = []
    blk_num = rand_attn.shape[1] + 2
    for h in range(rand_attn.shape[0]):
        layout = np.zeros([blk_num, blk_num], dtype = bool)
        for i in range(blk_num):
            if i == 0 or i == blk_num - 1:
                layout[i,:] = 1
            else:
                layout[i,0] = 1
                layout[i,-1] = 1
                layout[i, i-1:i+1] = 1 
                #for r in rand_attn[h, i+1, :]:
                for idx in range(rand_attn.shape[2]):
                  layout[i, rand_attn[h, i-1, idx]] = 1
        layouts.append(layout)
    layouts = np.array(layouts)
    return layouts


def bigbird_callback_simple():
    def mask_in_each_blocksparse_block(blk_shape, head_idx, qry_idx, key_idx, blk_idx):
        mask = np.ones(blk_shape, dtype=np.bool)
        return mask
    return mask_in_each_blocksparse_block

batch_size = 128
num_attention_heads = 1
size_per_head = 512
from_seq_length = 4096
to_seq_length = 4096

num_rand_blocks = 3
from_block_size = 32 

to_block_size = 32

blocksparse_bs = 32 

if __name__ == '__main__':
    # simple version
    assert from_seq_length == to_seq_length
    assert from_block_size == to_block_size
    assert blocksparse_bs == from_block_size

    is_fp16 = len(sys.argv) > 1 and sys.argv[1] == 'fp16'
    dtype = tf.float16 if is_fp16 else tf.float32

    q = tf.random_normal(shape=[batch_size, num_attention_heads, from_seq_length, size_per_head], dtype = dtype)
    k = tf.random_normal(shape=[batch_size, num_attention_heads, to_seq_length, size_per_head], dtype = dtype)
    v = tf.random_normal(shape=[batch_size, num_attention_heads, to_seq_length, size_per_head], dtype = dtype)

    start = time.perf_counter()

  # basically, we just need to prepare the blk_layout and the mask callback for each blk, 
  # then init a BlocksparseTransformer object which provide compute ops that related to 
  # the sparse pattern specficed by the layout and mask. 

  # layout = bigbird_layout(from_seq_length, to_seq_length, from_block_size, to_block_size, num_rand_blocks, num_attention_heads, blocksparse_bs)
  # for simple version, attention blocksize == compute blocksize(blocksparse size), 
  # we can get block layout from rand_attn directly.


    # TODO: pack these into a function and run in tf.Session
    rand_attn = generate_rand_attn_list(from_seq_length, from_block_size,
       to_block_size, num_rand_blocks, num_attention_heads)
    
    layout = bigbird_layout_simple(rand_attn)

    bst = BlocksparseTransformer(layout, block_size = blocksparse_bs, 
                                          mask_callback=bigbird_callback_simple(),# a callback that allways return ones(blk_shape, int32)
                                          heads = num_attention_heads)
