# AIRE：smoke test、正式实验与断点续跑

## 当前实验断点

本地拉取的 iterations_20261014_011847_3157341 包含162个一小时任务，0–86已成功，87–161尚无有效结果。此信息只代表本地快照，AIRE上可能已有新结果。对应原作业日志编号为8255467，不是最早9任务smoke的8255465。

配置更新只调整资源：主机内存16GB，每任务1张GPU；续跑默认并发上限3。算法仍为原9配置、3600秒、相同实例与顺序。内部时间种子每次启动重新生成并记录。续跑是任务级断点，不是从未完成任务的内部布局或随机状态继续；不完整任务会从头计算。

## 现在怎样续跑

将 resume_iterations.py、resume_iterations.sh、更新后的 run_iterations.slurm 上传到AIRE的bin_packing/hpc。不要覆盖或修改已提交实验的算法源码、metaheuristic_iterations.py、实例或几何；续跑会核对原manifest的哈希，只允许LF/CRLF差异。

在AIRE hpc目录执行：

```bash
export RESULT_DIR=/mnt/scratch/pmywe/GPU/3D_irregular_packing_stream/bin_packing/hpc_results/iterations_20261014_011847_3157341

# 先扫描：不提交、不归档、不改成功结果
DRY_RUN=1 bash resume_iterations.sh "$RESULT_DIR"

# 确认原实验是否还在运行
squeue -j 8255467

# 如需中止原资源申请以重新提交，只取消这组实验
scancel 8255467

# 等squeue确认旧任务全部退出，再按实际AIRE断点提交
MAX_PARALLEL=3 bash resume_iterations.sh "$RESULT_DIR"
```

如果原实验仍在运行，续跑脚本会拒绝提交。没有必要取消已完成作业；scancel行仅适用于仍然活动的原实验。取消后必须等待退出，因为正在保存的结果可能在短时间内继续写入。

脚本会自动：重新扫描每个任务；跳过身份、时间预算和验证状态符合要求的成功结果；保留原task_id；归档失败JSON及残留events至resume_archive；生成新的resume_manifest；只对缺失/无效任务提交稀疏array；使用16GB内存。它不创建全新的162任务实验，也不覆盖成功结果。

sbatch提交成功前若归档已发生，归档文件仍可恢复；重新执行脚本会再次依据当前状态生成待运行集合。新旧Slurm日志使用不同job_id，不会混在同一文件。

## 下次实验提交前的检查流程

1. 在准备环境中运行 `python -m unittest discover -s hpc -p 'test_*iterations.py'` 和算法控制流程测试；检查计数、kick边界、复算指标和恢复规则。
2. 用shapesnew/cube/seed3运行9配置启动smoke：TIME_LIMIT=30、MAX_PARALLEL=1、16GB。使用新结果目录，避免污染正式实验。30秒只能检查环境和启动，不保证100次失败或kick分支实际执行。
3. 检查9个task JSON的status/validation；运行collect；若不完整则先查slurm/*.err和错误字段。必要时在单独调试manifest中降低kick阈值以验证真实kick，不能将这些任务并入正式结果。
4. 对小、中、大代表实例执行资源试跑，覆盖构造、LS和kick。用 `sacct -j JOBID --format=JobID,State,Elapsed,ReqMem,MaxRSS` 检查峰值和OOM；用nvidia-smi观察GPU显存/忙碌程度。大实例的30秒预算仍可能被一次长构造调用超出，不能据短任务推断所有实例峰值。
5. 确认最大实际RAM需求后留余量；当前16GB依据管理员约5GB实测作为起点。对特别大的任务按实际峰值单独提高内存，避免所有任务申请240GB。
6. 正式prepare并固定源码/输入；用MAX_PARALLEL=3开始提交，按AIRE资源和账户限制调整；SLURM时间包含初始化、预热和不可中断的装箱调用，算法预算仍为3600秒。

启动smoke示例：

```bash
DATASETS=shapesnew SHAPES=cube SEEDS=3 TIME_LIMIT=30 MAX_PARALLEL=1 \
  bash submit_iterations.sh --mem=16G
```

## 监控、汇总与选择参数

运行时查看squeue、slurm日志和task JSON状态。结束后：

```bash
python metaheuristic_iterations.py collect \
  --manifest "$RESULT_DIR/manifest.json" --output "$RESULT_DIR"
```

仍使用原manifest汇总整个162任务实验。iteration_runs保留失败/缺失，iteration_summary给出均值/标准差和完整性，iteration_ranking只纳入9配置全完成的dataset/shape。最终先各算法内部比较三组平均N-U*与波动，再做跨实例平均排名；不能用未完成的局部结果宣称最佳参数。

重新提交采用新时间种子，因此失败任务不是原轨迹的复现。实际种子保存到结果中；如需复现某次轨迹，需要另外明确传入保存的内部种子。
