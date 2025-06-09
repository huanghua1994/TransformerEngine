import os
import time
import jax
import jax.numpy as jnp
from transformer_engine.jax.sharding import MeshResource
from test_fused_attn import FusedAttnRunner, SeqDescFormat
from transformer_engine.jax.attention import (
    is_fused_attn_kernel_available,
    AttnBiasType,
    AttnMaskType,
    QKVLayout,
    CPStrategy,
)

def impl_test_context_parallel_attn(
    device_count,
    mesh_shape,
    mesh_axes,
    mesh_resource,
    data_shape,
    kv_groups,
    attn_mask_type,
    dtype,
    qkv_layout,
    load_balanced,
    cp_strategy,
    use_shardy,
    use_scan_ring=False,
    window_size=None,
    num_tests=0,
):
    if qkv_layout.is_thd():
        if cp_strategy == CPStrategy.ALL_GATHER:
            raise ValueError("THD doesn't support all gather context parallelism.")
        if not load_balanced and cp_strategy == CPStrategy.RING:
            raise ValueError("THD + ring doesn't support unbalanced context parallelism.")

    assert not use_scan_ring or cp_strategy == CPStrategy.RING

    if use_scan_ring:
        os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "1"
    else:
        os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"] = "0"

    jax.config.update("jax_use_shardy_partitioner", use_shardy)
    attn_bias_type = AttnBiasType.NO_BIAS
    bias_shape = None
    dropout_prob = 0.0
    is_training = True
    dp_size, cp_size, tp_size = mesh_shape

    batch, seqlen, num_head, hidden = data_shape

    # Scale the sequence length by 2*CP so its never too small as we scale up test.
    # 2*CP is used since we split into two CP groups for load balancing.
    seqlen = seqlen * cp_size * 2
    data_shape = batch, seqlen, num_head, hidden

    num_kv_heads = num_head // kv_groups

    runner = FusedAttnRunner(
        batch,
        seqlen,
        seqlen,
        num_head,
        num_kv_heads,
        hidden,
        attn_bias_type,
        attn_mask_type,
        dropout_prob,
        dtype,
        is_training,
        qkv_layout,
        bias_shape,
        window_size,
        SeqDescFormat.SegmentIDs,
        number_of_devices=device_count,
        mesh_shape=mesh_shape,
        mesh_axes=mesh_axes,
        mesh_resource=mesh_resource,
        cp_strategy=cp_strategy,
        cp_load_balanced=load_balanced,
    )

    def check_has_backend_for_mask(mask_type):
        return is_fused_attn_kernel_available(
            dtype,
            dtype,
            qkv_layout,
            attn_bias_type,
            mask_type,
            dropout_prob,
            num_head,
            num_kv_heads,
            seqlen,
            seqlen,
            hidden,
            None,
        )  # no SWA for CP

    # For causal masking we depend on having bottom right support also.
    # The API does not check this and instead we rely on lower level checks to raise
    # and exception if the step backend is not supported. This was a deliberate API
    # decision to keep the CP size or flag out of the function.
    has_backend = check_has_backend_for_mask(attn_mask_type)
    if cp_size > 1 and attn_mask_type == AttnMaskType.CAUSAL_MASK:
        has_backend &= check_has_backend_for_mask(AttnMaskType.CAUSAL_BOTTOM_RIGHT_MASK)

    if not has_backend:
        raise ValueError(f"No FusedAttn backend found {cp_size=} {attn_mask_type=}.")

    if dp_size > 1 and batch % dp_size != 0:
        raise ValueError(f"Skipping {batch=} not a multiple of {dp_size=}")

    # make sure the mesh even divides cp and tp axis
    if num_head % kv_groups != 0 or (num_head // kv_groups) % tp_size != 0:
        raise ValueError(f"Skipping {kv_groups=} not multiple of {data_shape=} or {tp_size=}")

    runner.bench_forward(num_tests=num_tests)
    del os.environ["NVTE_FUSED_RING_ATTENTION_USE_SCAN"]

if __name__ == "__main__":
    device_count = 8
    mesh_shape = (1, 8, 1)
    mesh_axes = ("dp", "cp", "tp")
    mesh_resource = MeshResource(dp_resource="dp", cp_resource="cp", tp_resource="tp")
    # (batch, seqlen, num_head, hidden)
    # The actual sequence length will be scaled by 2 * CP size
    data_shape = [16, 8192 // (2 * 8), 64, 128]
    kv_groups = 1
    attn_mask_type = AttnMaskType.PADDING_CAUSAL_MASK
    dtype = jnp.bfloat16
    qkv_layout = QKVLayout.THD_THD_THD
    load_balanced = True
    cp_strategy = CPStrategy.RING
    use_shardy = False
    use_scan_ring = False
    window_size = (4096, 0)

    impl_test_context_parallel_attn(
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        kv_groups,
        attn_mask_type,
        dtype,
        qkv_layout,
        load_balanced,
        cp_strategy,
        use_shardy,
        use_scan_ring,
        window_size,
        num_tests=5,
    )