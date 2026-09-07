#include <metal_stdlib>
using namespace metal;

// Fused single-token decode paged attention kernel in Metal Shading Language (MSL)
// Directly reads non-contiguous physical KV blocks from TensorBlockStore using logical block tables.

kernel void paged_attention_decode_kernel(
    device const float* query              [[buffer(0)]], // [num_attn_heads, head_dim]
    device const float* k_store            [[buffer(1)]], // [num_layers, total_blocks, block_size, num_kv_heads, head_dim]
    device const float* v_store            [[buffer(2)]], // [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
    device const int*   block_table        [[buffer(3)]], // [num_blocks_in_table]
    device float*       output             [[buffer(4)]], // [num_attn_heads, head_dim]
    constant int&       num_attn_heads     [[buffer(5)]],
    constant int&       num_kv_heads       [[buffer(6)]],
    constant int&       head_dim           [[buffer(7)]],
    constant int&       block_size         [[buffer(8)]],
    constant int&       total_blocks       [[buffer(9)]],
    constant int&       num_blocks_in_table[[buffer(10)]],
    constant int&       seq_length         [[buffer(11)]],
    constant int&       layer_idx          [[buffer(12)]],
    constant float&     scale              [[buffer(13)]],
    uint thread_id                         [[thread_position_in_grid]]
) {
    if ((int)thread_id >= num_attn_heads) {
        return;
    }

    int head_idx = (int)thread_id;
    int queries_per_kv = num_attn_heads / num_kv_heads;
    int kv_head_idx = head_idx / queries_per_kv;

    // Running online softmax accumulators per query head
    float m_prev = -INFINITY;
    float l_prev = 0.0f;

    // Local accumulator for output vector [head_dim]
    // Max head_dim supported per thread registers: 128
    float acc[128];
    for (int d = 0; d < head_dim; ++d) {
        acc[d] = 0.0f;
    }

    // Pointer offsets in 5D physical tensor store:
    // [layer_idx, phys_block_id, t, kv_head_idx, d]
    // Strides for k_store & v_store:
    // layer_stride = total_blocks * block_size * num_kv_heads * head_dim
    // block_stride = block_size * num_kv_heads * head_dim
    // token_stride = num_kv_heads * head_dim
    // kv_head_stride = head_dim
    long layer_stride = (long)total_blocks * block_size * num_kv_heads * head_dim;
    long block_stride = (long)block_size * num_kv_heads * head_dim;
    long token_stride = (long)num_kv_heads * head_dim;
    long kv_head_stride = (long)head_dim;

    long layer_offset = (long)layer_idx * layer_stride;
    long q_offset = (long)head_idx * head_dim;

    // Iterate over physical blocks in logical order
    for (int b = 0; b < num_blocks_in_table; ++b) {
        int phys_block_id = block_table[b];
        int start_token = b * block_size;
        int end_token = min((b + 1) * block_size, seq_length);
        int valid_tokens = end_token - start_token;

        if (valid_tokens <= 0) {
            break;
        }

        long phys_block_offset = layer_offset + (long)phys_block_id * block_stride;

        for (int t = 0; t < valid_tokens; ++t) {
            long token_offset = phys_block_offset + (long)t * token_stride + (long)kv_head_idx * kv_head_stride;

            // Dot product q . k_t
            float score = 0.0f;
            for (int d = 0; d < head_dim; ++d) {
                score += query[q_offset + d] * k_store[token_offset + d];
            }
            score *= scale;

            // Online softmax update
            float m_new = max(m_prev, score);
            float alpha = exp(m_prev - m_new);
            float p_t = exp(score - m_new);

            l_prev = alpha * l_prev + p_t;

            for (int d = 0; d < head_dim; ++d) {
                acc[d] = alpha * acc[d] + p_t * v_store[token_offset + d];
            }

            m_prev = m_new;
        }
    }

    // Write normalized output vector
    long out_offset = (long)head_idx * head_dim;
    float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int d = 0; d < head_dim; ++d) {
        output[out_offset + d] = acc[d] * inv_l;
    }
}
