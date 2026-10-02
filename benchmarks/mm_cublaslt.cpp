// Shape-specific cuBLASLt heuristic search; raw pointers avoid tensor-dispatch overhead.
#include <pybind11/pybind11.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <cstdint>

void ck(cublasStatus_t s) { if(s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLASLt status " + std::to_string(s)); }
struct Plan {
  cublasLtHandle_t h;
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t a,b,d;
  void* work = nullptr;
  size_t work_bytes = 64 * 1024 * 1024;
  bool integer;
  std::vector<cublasLtMatmulHeuristicResult_t> algos;
  Plan(int m,int k,int n,uintptr_t bias,bool int8=false):integer(int8) {
    ck(cublasLtCreate(&h));
    ck(cublasLtMatmulDescCreate(&op,int8?CUBLAS_COMPUTE_32I:CUBLAS_COMPUTE_32F,int8?CUDA_R_32I:CUDA_R_32F));
    cublasOperation_t trans=CUBLAS_OP_T;
    ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&trans,sizeof(trans)));
    if(bias && !int8) {
      cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_BIAS;
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));
      void* ptr=reinterpret_cast<void*>(bias);
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&ptr,sizeof(ptr)));
    }
    ck(cublasLtMatrixLayoutCreate(&a,int8?CUDA_R_8I:CUDA_R_16BF,k,n,k));
    ck(cublasLtMatrixLayoutCreate(&b,int8?CUDA_R_8I:CUDA_R_16BF,k,m,k));
    ck(cublasLtMatrixLayoutCreate(&d,int8?CUDA_R_32I:CUDA_R_16BF,n,m,n));
    cublasLtMatmulPreference_t pref; ck(cublasLtMatmulPreferenceCreate(&pref));
    ck(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&work_bytes,sizeof(work_bytes)));
    algos.resize(32); int count=0;
    ck(cublasLtMatmulAlgoGetHeuristic(h,op,a,b,d,d,pref,32,algos.data(),&count));
    algos.resize(count); cublasLtMatmulPreferenceDestroy(pref);
    if(cudaMalloc(&work,work_bytes)!=cudaSuccess) throw std::runtime_error("workspace allocation failed");
  }
  int count() const { return algos.size(); }
  void run(int index,uintptr_t w,uintptr_t x,uintptr_t y,uintptr_t stream) {
    float alpha=1.0f,beta=0.0f;
    int ai=1,bi=0;
    ck(cublasLtMatmul(h,op,integer?(void*)&ai:(void*)&alpha,reinterpret_cast<void*>(w),a,reinterpret_cast<void*>(x),b,
      integer?(void*)&bi:(void*)&beta,reinterpret_cast<void*>(y),d,reinterpret_cast<void*>(y),d,&algos.at(index).algo,
      work,work_bytes,reinterpret_cast<cudaStream_t>(stream)));
  }
  void run_bias(int index,uintptr_t w,uintptr_t x,uintptr_t bias,uintptr_t y,uintptr_t stream) {
    // Benchmark-only, single CUDA stream: share one plan per shape, not per
    // weight tensor. Updating the pointer avoids hundreds of 64 MiB workspaces.
    if(bias) {
      void* ptr=reinterpret_cast<void*>(bias);
      ck(cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&ptr,sizeof(ptr)));
    }
    run(index,w,x,y,stream);
  }
  ~Plan() {cudaFree(work); cublasLtMatrixLayoutDestroy(a);cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(d);cublasLtMatmulDescDestroy(op);cublasLtDestroy(h);}
};
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  pybind11::class_<Plan>(m,"Plan").def(pybind11::init<int,int,int,uintptr_t,bool>())
    .def("count",&Plan::count).def("run",&Plan::run).def("run_bias",&Plan::run_bias);
}
