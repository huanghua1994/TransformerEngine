/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "pybind.h"

#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <optional>
#include <vector>

#include "../common.h"
#include "../extensions.h"
#include "common.h"

namespace transformer_engine::pytorch {

PyTypeObject *Float8TensorPythonClass = nullptr;  /// TODO Remove
PyTypeObject *Float8TensorBasePythonClass = nullptr;
PyTypeObject *Float8QuantizerClass = nullptr;
PyTypeObject *Float8CurrentScalingQuantizerClass = nullptr;
PyTypeObject *MXFP8TensorPythonClass = nullptr;  /// TODO Remove
PyTypeObject *MXFP8TensorBasePythonClass = nullptr;
PyTypeObject *MXFP8QuantizerClass = nullptr;
PyTypeObject *Float8BlockwiseQTensorPythonClass = nullptr;
PyTypeObject *Float8BlockwiseQTensorBasePythonClass = nullptr;
PyTypeObject *Float8BlockwiseQuantizerClass = nullptr;

void init_float8_extension() {
  if (Float8TensorPythonClass) return;
  auto fp8_module = py::module_::import("transformer_engine.pytorch.tensor.float8_tensor");
  Float8QuantizerClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "Float8Quantizer"));
  Float8CurrentScalingQuantizerClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8CurrentScalingQuantizer"));
  Float8TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "Float8Tensor"));
  auto fp8_base_module =
      py::module_::import("transformer_engine.pytorch.tensor._internal.float8_tensor_base");
  Float8TensorBasePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "Float8TensorBase"));
  NVTE_CHECK(Float8TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch Float8 extension.");
}

void init_mxfp8_extension() {
  if (MXFP8TensorPythonClass) return;
  auto fp8_module = py::module_::import("transformer_engine.pytorch.tensor.mxfp8_tensor");
  MXFP8QuantizerClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Quantizer"));
  MXFP8TensorPythonClass =
      reinterpret_cast<PyTypeObject *>(PyObject_GetAttrString(fp8_module.ptr(), "MXFP8Tensor"));
  auto fp8_base_module =
      py::module_::import("transformer_engine.pytorch.tensor._internal.mxfp8_tensor_base");
  MXFP8TensorBasePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "MXFP8TensorBase"));
  NVTE_CHECK(MXFP8TensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch MXFP8 extension.");
}

void init_float8blockwise_extension() {
  if (Float8BlockwiseQTensorBasePythonClass) return;
  auto fp8_module =
      py::module_::import("transformer_engine.pytorch.tensor.float8_blockwise_tensor");
  auto fp8_base_module = py::module_::import(
      "transformer_engine.pytorch.tensor._internal.float8_blockwise_tensor_base");
  Float8BlockwiseQuantizerClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8BlockQuantizer"));
  Float8BlockwiseQTensorBasePythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_base_module.ptr(), "Float8BlockwiseQTensorBase"));
  Float8BlockwiseQTensorPythonClass = reinterpret_cast<PyTypeObject *>(
      PyObject_GetAttrString(fp8_module.ptr(), "Float8BlockwiseQTensor"));

  NVTE_CHECK(Float8BlockwiseQuantizerClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
  NVTE_CHECK(Float8BlockwiseQTensorBasePythonClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
  NVTE_CHECK(Float8BlockwiseQTensorPythonClass != nullptr,
             "Internal error: could not initialize pyTorch float8blockwise extension.");
}

void init_extension() {
  init_float8_extension();
  init_mxfp8_extension();
  init_float8blockwise_extension();
}

}  // namespace transformer_engine::pytorch

#include "common/util/pybind_helper.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  NVTE_DECLARE_COMMON_PYBIND11_HANDLES(m)
  m.def("quantize", transformer_engine::pytorch::quantize, py::arg("tensor"), py::arg("quantizer"),
        py::arg("output") = py::none(), py::arg("noop") = py::none());
  m.def("dequantize", &transformer_engine::pytorch::dequantize, "Dequantize", py::arg("input"),
        py::arg("otype"));

  m.def("bgrad_quantize", transformer_engine::pytorch::bgrad_quantize,
        "Compute bias gradient and quantize", py::arg("input"), py::arg("quantizer"));
  m.def("generic_gemm", transformer_engine::pytorch::gemm, "Compute GEMM (matrix-matrix multiply)",
        py::arg("A"), py::arg("transA"), py::arg("B"), py::arg("transB"), py::arg("D"),
        py::arg("quantizer"), py::arg("output_dtype"), py::arg("bias"), py::arg("bias_type"),
        py::arg("gelu"), py::arg("gelu_in"), py::arg("grad"), py::arg("workspace"),
        py::arg("workspace_size"), py::arg("accumulate"), py::arg("use_split_accumulator"),
        py::arg("comm_overlap") = nullptr, py::arg("comm_type") = std::nullopt,
        py::arg("extra_output") = std::nullopt, py::arg("bulk_overlap") = false);
  m.def("gelu", transformer_engine::pytorch::gelu, "GeLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("relu", transformer_engine::pytorch::relu, "ReLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("geglu", transformer_engine::pytorch::geglu, "GeGLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("qgeglu", transformer_engine::pytorch::qgeglu, "QuickGeGLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("reglu", transformer_engine::pytorch::reglu, "ReGLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("swiglu", transformer_engine::pytorch::swiglu, "SwiGLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("qgelu", transformer_engine::pytorch::qgelu, "QuickGELU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("srelu", transformer_engine::pytorch::srelu, "Squared ReLU activation", py::arg("input"),
        py::arg("quantizer"));
  m.def("dgelu", transformer_engine::pytorch::dgelu, "Backward of GeLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("drelu", transformer_engine::pytorch::drelu, "Backward of ReLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dgeglu", transformer_engine::pytorch::dgeglu, "Backward of GeGLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dqgeglu", transformer_engine::pytorch::dqgeglu, "Backward of QuickGeGLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dreglu", transformer_engine::pytorch::dreglu, "Backward of ReGLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dswiglu", transformer_engine::pytorch::dswiglu, "Backward of SwiGLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dqgelu", transformer_engine::pytorch::dqgelu, "Backward of QuickGELU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dsrelu", transformer_engine::pytorch::dsrelu, "Backward of Squared ReLU", py::arg("grad"),
        py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dbias_dgelu", transformer_engine::pytorch::dbias_dgelu, "DGeLU + DBias + Quantize",
        py::arg("grad"), py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dbias_dsilu", transformer_engine::pytorch::dbias_dsilu, "DSiLU + DBias + Quantize",
        py::arg("grad"), py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dbias_drelu", transformer_engine::pytorch::dbias_drelu, "DReLU + DBias + Quantize",
        py::arg("grad"), py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dbias_dqgelu", transformer_engine::pytorch::dbias_dqgelu, "DQGeLU + DBias + Quantize",
        py::arg("grad"), py::arg("fwd_input"), py::arg("quantizer"));
  m.def("dbias_dsrelu", transformer_engine::pytorch::dbias_dsrelu,
        "DSquaredReLU + DBias + Quantize", py::arg("grad"), py::arg("fwd_input"),
        py::arg("quantizer"));

  // Permutation functions
  m.def("moe_permute_fwd", transformer_engine::pytorch::moe_permute_fwd, "MOE permute FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("moe_permute_bwd", transformer_engine::pytorch::moe_permute_bwd, "MOE permute BWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("moe_unpermute_fwd", transformer_engine::pytorch::moe_unpermute_fwd, "MOE unpermute FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("moe_unpermute_bwd", transformer_engine::pytorch::moe_unpermute_bwd, "MOE unpermute BWD",
        py::call_guard<py::gil_scoped_release>());

  // Softmax functions
  m.def("scaled_softmax_forward", &transformer_engine::pytorch::scaled_softmax_forward,
        "Scaled Softmax FWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_softmax_backward", &transformer_engine::pytorch::scaled_softmax_backward,
        "Scaled Softmax BWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_masked_softmax_forward, "Scaled Masked Softmax FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_masked_softmax_backward, "Scaled Masked Softmax BWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_upper_triang_masked_softmax_forward,
        "Scaled Upper-Triangular Masked Softmax FWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_upper_triang_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_upper_triang_masked_softmax_backward,
        "Scaled Upper-Triangular Masked Softmax BWD", py::call_guard<py::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_forward",
        &transformer_engine::pytorch::scaled_aligned_causal_masked_softmax_forward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax FWD",
        py::call_guard<py::gil_scoped_release>());
  m.def("scaled_aligned_causal_masked_softmax_backward",
        &transformer_engine::pytorch::scaled_aligned_causal_masked_softmax_backward,
        "Scaled Bottom-Right Corner Aligned Masked Softmax BWD",
        py::call_guard<py::gil_scoped_release>());

  // Other granular functions
  m.def("layernorm_fwd", &transformer_engine::pytorch::layernorm_fwd, "LayerNorm", py::arg("input"),
        py::arg("weight"), py::arg("bias"), py::arg("eps"), py::arg("ln_out"), py::arg("quantizer"),
        py::arg("otype"), py::arg("sm_margin"), py::arg("zero_centered_gamma"));
  m.def("layernorm_bwd", &transformer_engine::pytorch::layernorm_bwd, "Backward of LayerNorm");
  m.def("rmsnorm_fwd", &transformer_engine::pytorch::rmsnorm_fwd, "RMSNorm", py::arg("input"),
        py::arg("weight"), py::arg("eps"), py::arg("ln_out"), py::arg("quantizer"),
        py::arg("otype"), py::arg("sm_margin"), py::arg("zero_centered_gamma"));
  m.def("rmsnorm_bwd", &transformer_engine::pytorch::rmsnorm_bwd, "Backward of RMSNorm");
  m.def("multi_tensor_quantize", &transformer_engine::pytorch::multi_tensor_quantize,
        "Multi-tensor quantize", py::arg("tensor_list"), py::arg("quantizer_list"));
  m.def("split_quantize", &transformer_engine::pytorch::split_quantize,
        "Split and multi-tensor quantize", py::arg("tensor"), py::arg("split_sections"),
        py::arg("quantizer_list"));
  m.def("te_general_grouped_gemm", &transformer_engine::pytorch::te_general_grouped_gemm,
        "Grouped GEMM");
  m.def("fp8_transpose", &transformer_engine::pytorch::fp8_transpose, "Transpose with FP8 I/O",
        py::arg("input"), py::arg("dtype"), py::kw_only(), py::arg("out"),
        py::call_guard<py::gil_scoped_release>());
  m.def("get_fused_attn_backend", &transformer_engine::pytorch::get_fused_attn_backend,
        "Get Fused Attention backend", py::call_guard<py::gil_scoped_release>());
  m.def("compute_amax", &transformer_engine::pytorch::compute_amax,
        "Compute absolute max value in tensor", py::arg("input"), py::arg("amax"),
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_amax_and_scale_update_after_reduction",
        &transformer_engine::pytorch::fused_amax_and_scale_update_after_reduction,
        "Update amax history and FP8 scale/scale_inv after reduction",
        py::call_guard<py::gil_scoped_release>());
  m.def("fp8_block_scaling_compute_partial_amax",
        &transformer_engine::pytorch::fp8_block_scaling_compute_partial_amax,
        "Compute partial amax from master weights for fp8 block scaling", py::arg("tensor"),
        py::arg("amax"), py::arg("h"), py::arg("w"), py::arg("start_offset"), py::arg("block_len"),
        py::call_guard<py::gil_scoped_release>());
  m.def("fp8_block_scaling_partial_cast",
        &transformer_engine::pytorch::fp8_block_scaling_partial_cast,
        "Partial cast from master weights for fp8 block scaling", py::arg("inp"), py::arg("out"),
        py::arg("scale"), py::arg("h"), py::arg("w"), py::arg("start_offset"), py::arg("block_len"),
        py::arg("out_dtype"), py::call_guard<py::gil_scoped_release>());
  m.def("fused_multi_row_padding", &transformer_engine::pytorch::fused_multi_row_padding,
        "Fused Multi-tensor padding", py::call_guard<py::gil_scoped_release>());
  m.def("fused_multi_row_unpadding", &transformer_engine::pytorch::fused_multi_row_unpadding,
        "Fused Multi-tensor unpadding", py::call_guard<py::gil_scoped_release>());

  // attention kernels
  m.def("fa_prepare_fwd", &transformer_engine::pytorch::fa_prepare_fwd,
        "Prepare QKV for Flash Attention", py::call_guard<py::gil_scoped_release>());
  m.def("fa_prepare_bwd", &transformer_engine::pytorch::fa_prepare_bwd,
        "Backward of QKV preparation for Flash Attention",
        py::call_guard<py::gil_scoped_release>());
  m.def("fused_attn_fwd", &transformer_engine::pytorch::fused_attn_fwd,
        "Fused Attention FP8/BF16/FP16 FWD with separate Q, K and V");
  m.def("fused_attn_bwd", &transformer_engine::pytorch::fused_attn_bwd,
        "Fused Attention FP8/BF16/FP16 BWD with separate Q, K and V");
  m.def("copy_to_kv_cache", &transformer_engine::pytorch::copy_to_kv_cache,
        "Copy new KV tokens to KV cache", py::call_guard<py::gil_scoped_release>());
  m.def("convert_thd_to_bshd", &transformer_engine::pytorch::convert_thd_to_bshd,
        "Convert a tensor from THD to BSHD", py::call_guard<py::gil_scoped_release>());
  m.def("convert_bshd_to_thd", &transformer_engine::pytorch::convert_bshd_to_thd,
        "Convert a tesnor from BSHD to THD", py::call_guard<py::gil_scoped_release>());

  // fused apply rope
  m.def("fused_rope_forward", &transformer_engine::pytorch::fused_rope_forward,
        "Fused Apply RoPE FWD", py::call_guard<py::gil_scoped_release>());
  m.def("fused_rope_backward", &transformer_engine::pytorch::fused_rope_backward,
        "Fused Apply RoPE BWD", py::call_guard<py::gil_scoped_release>());

  // fused router
  m.def("fused_topk_with_score_function_fwd",
        &transformer_engine::pytorch::fused_topk_with_score_function_fwd, py::arg("logits"),
        py::arg("topk"), py::arg("use_pre_softmax"), py::arg("num_groups"), py::arg("group_topk"),
        py::arg("scaling_factor"), py::arg("score_function"), py::arg("expert_bias"),
        "Fused topk softmax fwd");
  m.def("fused_topk_with_score_function_bwd",
        &transformer_engine::pytorch::fused_topk_with_score_function_bwd, py::arg("num_tokens"),
        py::arg("num_experts"), py::arg("routing_map"), py::arg("intermediate_output"),
        py::arg("grad_probs"), py::arg("topk"), py::arg("use_pre_softmax"),
        py::arg("scaling_factor"), py::arg("score_function"), "Fused topk softmax bwd");
  m.def("fused_score_for_moe_aux_loss_fwd",
        &transformer_engine::pytorch::fused_score_for_moe_aux_loss_fwd, py::arg("logits"),
        py::arg("topk"), py::arg("score_function"), "Fused topk softmax fwd");
  m.def("fused_score_for_moe_aux_loss_bwd",
        &transformer_engine::pytorch::fused_score_for_moe_aux_loss_bwd, py::arg("num_tokens"),
        py::arg("num_experts"), py::arg("intermediate_output"), py::arg("grad_scores"),
        py::arg("topk"), py::arg("score_function"), "Fused topk softmax bwd");
  m.def("fused_moe_aux_loss_fwd", &transformer_engine::pytorch::fused_moe_aux_loss_fwd,
        py::arg("probs"), py::arg("tokens_per_expert"), py::arg("total_num_tokens"),
        py::arg("num_experts"), py::arg("num_rows"), py::arg("num_cols"), py::arg("topk"),
        py::arg("coeff"), "Fused aux loss fwd");
  m.def("fused_moe_aux_loss_bwd", &transformer_engine::pytorch::fused_moe_aux_loss_bwd,
        py::arg("Const_buf"), py::arg("tokens_per_expert"), py::arg("num_rows"),
        py::arg("num_cols"), py::arg("grad_aux_loss"), "Fused aux loss bwd");

  // Misc
  m.def("get_cublasLt_version", &transformer_engine::pytorch::get_cublasLt_version,
        "Get cublasLt version", py::call_guard<py::gil_scoped_release>());
  m.def("get_cudnn_version", &transformer_engine::pytorch::get_cudnn_version, "Get cuDNN version",
        py::call_guard<py::gil_scoped_release>());
  m.def("get_num_cublas_streams", &nvte_get_num_compute_streams, "Get number of compute streams",
        py::call_guard<py::gil_scoped_release>());

  // Support THD format for Context Parallel
  m.def("thd_read_half_tensor", &transformer_engine::pytorch::thd_read_half_tensor,
        "Read the first half(half_idx=0) or the second half(half_idx=1) of each sequence in a THD "
        "tensor",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_second_half_lse_correction",
        &transformer_engine::pytorch::thd_second_half_lse_correction,
        "Correct the second half of the softmax_lse", py::call_guard<py::gil_scoped_release>());
  m.def("thd_read_second_half_lse", &transformer_engine::pytorch::thd_read_second_half_lse,
        "Read the second half of the softmax_lse", py::call_guard<py::gil_scoped_release>());
  m.def("thd_out_correction", &transformer_engine::pytorch::thd_out_correction,
        "Correct the THD format output of context parallelism in forward pass",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_grad_correction", &transformer_engine::pytorch::thd_grad_correction,
        "Correct the THD format gradients of context parallelism in backward pass",
        py::call_guard<py::gil_scoped_release>());
  m.def("thd_get_partitioned_indices", &transformer_engine::pytorch::thd_get_partitioned_indices,
        "Generate partitioned indices for inputs in THD format",
        py::call_guard<py::gil_scoped_release>());

  // nvshmem functions
  m.def("init_nvshmem_backend", &transformer_engine::pytorch::init_nvshmem_backend,
        "Initialize nvshmem backend with Pytorch distributed process groups",
        py::call_guard<py::gil_scoped_release>());
  m.def("create_nvshmem_tensor", &transformer_engine::pytorch::create_nvshmem_tensor,
        "Create a tensor in NVSHMEM shared memory", py::call_guard<py::gil_scoped_release>());
  m.def("nvshmem_send_on_current_stream",
        &transformer_engine::pytorch::nvshmem_send_on_current_stream,
        "Asynchronously send tensor data to a remote PE using NVSHMEM on the current CUDA stream",
        py::call_guard<py::gil_scoped_release>());
  m.def("nvshmem_wait_on_current_stream",
        &transformer_engine::pytorch::nvshmem_wait_on_current_stream,
        "Wait for a signal value to be updated by a remote PE using NVSHMEM on the current CUDA "
        "stream",
        py::call_guard<py::gil_scoped_release>());
  m.def("nvshmem_finalize", &transformer_engine::pytorch::nvshmem_finalize,
        "Clean up and finalize the NVSHMEM communication backend and free associated resources",
        py::call_guard<py::gil_scoped_release>());

  // multi-tensor functions
  m.def("multi_tensor_scale", &transformer_engine::pytorch::multi_tensor_scale_cuda,
        "Fused overflow check + scale for a list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_l2norm", &transformer_engine::pytorch::multi_tensor_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_unscale_l2norm",
        &transformer_engine::pytorch::multi_tensor_unscale_l2norm_cuda,
        "Computes L2 norm for a list of contiguous tensors after unscaling (unscaling is only "
        "performed for L2 norm computation, and tensors are not updated)",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam", &transformer_engine::pytorch::multi_tensor_adam_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_param_remainder",
        &transformer_engine::pytorch::multi_tensor_adam_param_remainder_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer"
        "where the master parameters only store the remainder bits",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_fp8", &transformer_engine::pytorch::multi_tensor_adam_fp8_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable",
        &transformer_engine::pytorch::multi_tensor_adam_capturable_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support and LR scheduling",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_adam_capturable_master",
        &transformer_engine::pytorch::multi_tensor_adam_capturable_master_cuda,
        "Compute and apply gradient update to parameters for Adam optimizer with CUDA graph "
        "support, LR scheduling and FP32 master weights",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_sgd", &transformer_engine::pytorch::multi_tensor_sgd_cuda,
        "Fused SGD optimizer for list of contiguous tensors",
        py::call_guard<py::gil_scoped_release>());
  m.def("multi_tensor_compute_scale_and_scale_inv",
        &transformer_engine::pytorch::multi_tensor_compute_scale_and_scale_inv_cuda,
        "Fused compute scale and scale_inv from amax", py::call_guard<py::gil_scoped_release>());

  // Data structures
  py::class_<transformer_engine::pytorch::FP8TensorMeta>(m, "FP8TensorMeta")
      .def(py::init<>())
      .def_readwrite("scale", &transformer_engine::pytorch::FP8TensorMeta::scale)
      .def_readwrite("scale_inv", &transformer_engine::pytorch::FP8TensorMeta::scale_inv)
      .def_readwrite("amax_history", &transformer_engine::pytorch::FP8TensorMeta::amax_history);

  py::enum_<transformer_engine::pytorch::FP8FwdTensors>(m, "FP8FwdTensors")
      .value("GEMM1_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_INPUT)
      .value("GEMM1_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_WEIGHT)
      .value("GEMM1_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM1_OUTPUT)
      .value("GEMM2_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_INPUT)
      .value("GEMM2_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_WEIGHT)
      .value("GEMM2_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM2_OUTPUT)
      .value("GEMM3_INPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_INPUT)
      .value("GEMM3_WEIGHT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_WEIGHT)
      .value("GEMM3_OUTPUT", transformer_engine::pytorch::FP8FwdTensors::GEMM3_OUTPUT);

  py::enum_<transformer_engine::pytorch::FP8BwdTensors>(m, "FP8BwdTensors")
      .value("GRAD_OUTPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT1)
      .value("GRAD_INPUT1", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT1)
      .value("GRAD_OUTPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT2)
      .value("GRAD_INPUT2", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT2)
      .value("GRAD_OUTPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_OUTPUT3)
      .value("GRAD_INPUT3", transformer_engine::pytorch::FP8BwdTensors::GRAD_INPUT3);

  py::class_<CommOverlapHelper>(m, "CommOverlapHelper")
      .def(py::init<>(), py::call_guard<py::gil_scoped_release>())
      .def(py::init<c10d::ProcessGroup *, std::optional<c10d::ProcessGroup *>>(),
           py::call_guard<py::gil_scoped_release>(), py::arg("world_group"),
           py::arg("intra_node_group") = py::none());

  py::class_<CommOverlap, std::shared_ptr<CommOverlap>, transformer_engine::CommOverlapBase,
             transformer_engine::CommOverlapCore>(m, "CommOverlap")
      .def(py::init<const std::vector<size_t> &, at::ScalarType, CommOverlapHelper *, int, int, int,
                    int, int, int, int, bool, bool, bool>(),
           py::call_guard<py::gil_scoped_release>(), py::arg("buffer_shape"),
           py::arg("buffer_dtype"), py::arg("helper"), py::arg("tp_size"),
           py::arg("num_splits") = 3, py::arg("num_max_streams") = NVTE_COMM_OVERLAP_MAX_STREAMS,
           py::arg("comm_cga_size") = 2, py::arg("gemm_priority") = 0, py::arg("comm_priority") = 0,
           py::arg("num_comm_sm") = 16, py::arg("set_sm_margin") = true,
           py::arg("atomic_gemm") = false, py::arg("rs_overlap_first_gemm") = false)
      .def("copy_into_buffer", &CommOverlap::copy_into_buffer, py::arg("input"),
           py::arg("local_chunk") = false)
      .def("get_buffer", &CommOverlap::get_buffer, py::arg("local_chunk") = false,
           py::arg("shape") = std::nullopt)
      .def("get_communication_stream", &CommOverlap::get_communication_stream);

  py::class_<CommOverlapP2P, std::shared_ptr<CommOverlapP2P>,
             transformer_engine::CommOverlapP2PBase, transformer_engine::CommOverlapCore>(
      m, "CommOverlapP2P")
      .def(py::init<const std::vector<size_t> &, at::ScalarType, CommOverlapHelper *, int,
                    transformer_engine::CommOverlapType, int, int, int, int, int, bool, bool, bool,
                    bool>(),
           py::call_guard<py::gil_scoped_release>(), py::arg("buffer_shape"),
           py::arg("buffer_dtype"), py::arg("helper"), py::arg("tp_size"), py::arg("comm_type"),
           py::arg("num_max_streams") = NVTE_COMM_OVERLAP_MAX_STREAMS, py::arg("comm_cga_size") = 1,
           py::arg("gemm_priority") = 0, py::arg("comm_priority") = 0, py::arg("num_comm_sm") = 1,
           py::arg("set_sm_margin") = false, py::arg("atomic_gemm") = false,
           py::arg("use_ce") = true, py::arg("aggregate") = false)
      .def("copy_into_buffer", &CommOverlapP2P::copy_into_buffer, py::arg("input"),
           py::arg("local_chunk") = false)
      .def("get_buffer", &CommOverlapP2P::get_buffer, py::arg("local_chunk") = false,
           py::arg("shape") = std::nullopt)
      .def("get_communication_stream", &CommOverlapP2P::get_communication_stream);
}
