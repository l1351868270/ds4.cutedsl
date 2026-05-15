```
# prefill
NCCL_DEBUG=WARN nohup torchrun --nproc-per-node  8 ./benchmark/bench.py \
  --ckpt-path /wh-test/lishuliang/models/deepseek-ai/DeepSeek-V4-Flash-fp8/   \
  --config ./inference/config_fp8.json \
  --input-file ./benchmark/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max  65536 \
  --step-incr 2048  \
  --gen-tokens 128 \
  --csv-path ./benchmark/record_$(date +%Y%m%d)_$(date +%H%M%S).csv 
```