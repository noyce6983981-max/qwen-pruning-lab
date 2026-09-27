# 受控部署微基准

两轮独立进程，四模型正序/反序；batch=1，固定32输出，预热2次、计时7次。每格为该轮中位数 [最小, 最大]。

| 模型 | 输入 | 轮次 | 首token 秒 | 后31token tok/s | 分配峰值 GiB | 保留峰值 GiB |
| --- | --- | --- | --- | --- | --- | --- |
| original_step0 | 128 | 1 | 0.0439 [0.0369, 0.0518] | 28.2957 [26.8215, 29.9587] | 3.2636 | 3.8105 |
| original_step0 | 128 | 2 | 0.0491 [0.0463, 0.0614] | 23.0937 [20.3213, 24.2219] | 3.2636 | 3.8105 |
| original_step0 | 512 | 1 | 0.1052 [0.0980, 0.1078] | 28.6861 [27.1167, 29.8683] | 3.4148 | 3.8457 |
| original_step0 | 512 | 2 | 0.1023 [0.0976, 0.1030] | 21.9154 [19.1788, 23.7214] | 3.4148 | 3.8457 |
| calibrated_5888_step0 | 128 | 1 | 0.0462 [0.0414, 0.0569] | 27.2931 [25.9735, 28.6634] | 3.1825 | 3.2051 |
| calibrated_5888_step0 | 128 | 2 | 0.0591 [0.0481, 0.0615] | 21.5147 [20.7533, 23.2204] | 3.1825 | 3.2051 |
| calibrated_5888_step0 | 512 | 1 | 0.1054 [0.1023, 0.1223] | 23.3369 [18.2256, 25.0620] | 3.3328 | 3.4160 |
| calibrated_5888_step0 | 512 | 2 | 0.1030 [0.0999, 0.1084] | 22.2807 [19.4248, 23.8347] | 3.3328 | 3.4160 |
| original_step64_merged | 128 | 1 | 0.0579 [0.0450, 0.0653] | 22.0467 [17.3701, 22.9183] | 3.2645 | 3.2871 |
| original_step64_merged | 128 | 2 | 0.0826 [0.0743, 0.1020] | 14.1870 [12.2709, 15.1932] | 3.2645 | 3.2871 |
| original_step64_merged | 512 | 1 | 0.1049 [0.1034, 0.1445] | 22.6384 [17.3418, 24.2101] | 3.4148 | 3.4980 |
| original_step64_merged | 512 | 2 | 0.1052 [0.1011, 0.1394] | 20.9908 [16.8013, 23.2745] | 3.4148 | 3.4980 |
| calibrated_5888_step64_merged | 128 | 1 | 0.1006 [0.0677, 0.1383] | 13.8967 [13.2059, 16.1852] | 3.1825 | 3.2051 |
| calibrated_5888_step64_merged | 128 | 2 | 0.0922 [0.0730, 0.1087] | 12.2697 [11.1387, 16.6210] | 3.1825 | 3.2051 |
| calibrated_5888_step64_merged | 512 | 1 | 0.1126 [0.1070, 0.1279] | 14.0658 [12.5138, 16.2124] | 3.3328 | 3.4160 |
| calibrated_5888_step64_merged | 512 | 2 | 0.1063 [0.0975, 0.1098] | 22.8827 [19.1393, 26.5829] | 3.3328 | 3.4160 |

显存列为该输入长度两轮的最大值；加载峰值和每次计时原始记录在JSON。

| 对比 | 输入 | 首token判断 | 后31token判断 |
| --- | --- | --- | --- |
| original_step0 → calibrated_5888_step0 | 128 | 未观察到稳定加速 | 未观察到稳定加速 |
| original_step0 → calibrated_5888_step0 | 512 | 未观察到稳定加速 | 未观察到稳定加速 |
| original_step0 → original_step64_merged | 128 | 未观察到稳定加速 | 此固定微基准中观察到两轮方向一致且范围分离的变慢 |
| original_step0 → original_step64_merged | 512 | 未观察到稳定加速 | 未观察到稳定加速 |
| original_step0 → calibrated_5888_step64_merged | 128 | 此固定微基准中观察到两轮方向一致且范围分离的变慢 | 此固定微基准中观察到两轮方向一致且范围分离的变慢 |
| original_step0 → calibrated_5888_step64_merged | 512 | 未观察到稳定加速 | 未观察到稳定加速 |
| original_step64_merged → calibrated_5888_step64_merged | 128 | 未观察到稳定加速 | 未观察到稳定加速 |
| original_step64_merged → calibrated_5888_step64_merged | 512 | 未观察到稳定加速 | 未观察到稳定加速 |
| calibrated_5888_step0 → calibrated_5888_step64_merged | 128 | 此固定微基准中观察到两轮方向一致且范围分离的变慢 | 此固定微基准中观察到两轮方向一致且范围分离的变慢 |
| calibrated_5888_step0 → calibrated_5888_step64_merged | 512 | 未观察到稳定加速 | 未观察到稳定加速 |

只要两轮方向不一致，或任一轮范围重叠，就不把中位数差称为稳定加速。这里的判断不是显著性检验。

| 模型 | 去重参数 | 去重权重张量字节 | 权重文件字节 |
| --- | --- | --- | --- |
| original_step0 | 1720574976 | 3441149952 | 4063515592 |
| calibrated_5888_step0 | 1676534784 | 3353069568 | 3353104728 |
| original_step64_merged | 1720574976 | 3441149952 | 3441185608 |
| calibrated_5888_step64_merged | 1676534784 | 3353069568 | 3353105216 |

温度、功耗、SM/显存频率、整卡占用、驱动与Windows电源状态的可得性和原始值保留在JSON；未修改电源设置。

- Single machine and GPU, two independent process rounds; synthetic repeated Chinese token fixture.
- No live LoRA; merged BF16 checkpoints only. 512 input plus 32 output is a timing workload, not the 512-token QA protocol.
- TTFT excludes tokenization, disk loading and CPU output transfer; later-token rate includes Python/control overhead.
- Allocator allocated/reserved are not whole-device memory. Disk weight bytes include safetensors headers.
- Original snapshot and save_pretrained exports may serialize tied weights differently; disk-byte savings must not be attributed entirely to pruning.
- Telemetry is sampled outside timing, not continuous; unavailable fields remain unavailable. No automatic power-setting changes.
- Nonoverlapping ranges in both sessions are a conservative descriptive condition, not statistical significance or guaranteed deployment speed.
- Quality acceptance is a separate prerequisite enforced by the experiment orchestrator, not inferred from speed.
