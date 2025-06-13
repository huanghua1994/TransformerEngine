#! /bin/bash
set -x

AR_THRESHOLD=${AR_THRESHOLD:=1073741824}
AG_THRESHOLD=${AG_THRESHOLD:=1073741824}
RS_THRESHOLD=${RS_THRESHOLD:=134217728}
XLA_BASE_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true
                --xla_gpu_enable_triton_gemm=false
                --xla_gpu_enable_highest_priority_async_stream=true
                --xla_gpu_all_gather_combine_threshold_bytes=${AG_THRESHOLD}
                --xla_gpu_reduce_scatter_combine_threshold_bytes=${RS_THRESHOLD}
                --xla_gpu_enable_pipelined_all_gather=true
                --xla_gpu_enable_pipelined_reduce_scatter=true
                --xla_gpu_collective_permute_combine_threshold_bytes=536870912
                --xla_ignore_channel_id=true
                --xla_gpu_enable_nccl_comm_splitting=false"

XLA_HLO_DUMP_FLAGS="--xla_dump_hlo_as_text
                    --xla_dump_hlo_pass_re=.*
                    --xla_dump_to=/tmp/hlo-dump"

XLA_ENABLE_P2P_FLAGS="--xla_gpu_threshold_for_windowed_einsum_mib=0
                      --xla_gpu_multi_streamed_windowed_einsum=true
                      --xla_gpu_use_memcpy_local_p2p=true"

XLA_DISABLE_CUDA_GRAPH_FLAG="--xla_gpu_enable_command_buffer=\"\""

export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRAC:=0.9}

export XLA_FLAGS="${XLA_BASE_FLAGS}
                  ${XLA_HLO_DUMP_FLAGS}
                  ${XLA_ENABLE_P2P_FLAGS}"

export JAX_ENABLE_PGLE=true
export TF_CPP_VMODULE=profile_guided_latency_estimator=10,latency_hiding_scheduler=10,gpu_hlo_module=10
export TF_CPP_MIN_LOG_LEVEL=0
export TF_CPP_MAX_LOG_LEVEL=10

nsys profile --cuda-graph-trace=node python bench_fused_attn.py
#python bench_fused_attn.py
