参考[dsv4](https://github.com/antirez/ds4/tree/main/speed-bench) 实现

参见inference/README.md 保证模型能够正常推理

```
NCCL_DEBUG=WARN torchrun --nproc-per-node  8 ./benchmark/bench.py \
  --ckpt-path deepseek-ai/DeepSeek-V4-Flash-fp8/  \
  --config ./inference/config_fp8.json \
  --input-file ./benchmark/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max  65536 \
  --step-incr 2048  \
  --gen-tokens 128 \
  --csv-path ./benchmark/record_$(date +%Y%m%d)_$(date +%H%M%S).csv
```

使用CSV文件生成SVG图片:

```
python3 benchmark/plot_speed.py benchmark/recod*.csv --title "$(date +%Y%m%d)_$(date +%H%M%S) t/s"
```