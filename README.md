Adapted From https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/tree/main/inference

Use the official implementation of DeepSeek as the baseline.

**Optimization Goal: Reduce single-request latency to the SOTA level.**

TODO: 
1. https://arxiv.org/pdf/2605.02568
2. [CODA: Rewriting Transformer Blocks as GEMM-Epilogue Programs](https://arxiv.org/pdf/2605.19269)
3. https://github.com/NVIDIA/Megatron-LM/issues/4957
4. https://zhuanlan.zhihu.com/p/2035845795082133865
5. [Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/system/files/osdi23-shi.pdf)
6. [Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X GPUs](https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)
7. [Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://arxiv.org/pdf/2604.15379)
8. [Fleet: Hierarchical Task-based Abstraction for Megakernels on Multi-Die GPUs](https://arxiv.org/pdf/2604.15379)
9. [1T 模型通往 1000 Tokens/s 的两次跃迁：关于推理系统、执行边界与 Co-Design](https://www.tilert.ai/blog/breaking-1000-tps-zh.html)
10. [速度：大模型推理的下一个 Scaling Law](https://www.tilert.ai/blog/speed-as-the-next-scaling-law-zh.html)
