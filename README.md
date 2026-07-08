Adapted From https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference

Use the official implementation of DeepSeek as the baseline.

**Optimization Goal: Reduce single-request latency to the SOTA level.**

TODO: 
1. https://arxiv.org/pdf/2605.02568
2. [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/pdf/2605.19269)
3. https://github.com/NVIDIA/Megatron-LM/issues/4957
4. https://zhuanlan.zhihu.com/p/2035845795082133865
5. [Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/system/files/osdi23-shi.pdf)
