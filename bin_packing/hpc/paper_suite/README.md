# EJOR 实验流程

对应 2026-10-16 的《EJOR paper update summary.docx》。所有新实验使用独立的冻结代码包，不复用旧 iteration 实验的结果。入口为 `suite.py`，准备、汇总和提交只依赖 Python 标准库；GPU worker 使用已有的 AIRE `packing` conda 环境。

## 实验及依赖

|阶段|运行内容|正式任务数|下一阶段使用的结果|
|---|---|---:|---|
|calibration|每个 dataset/shape 五个顺序 seed 的 geometry-only CA|150|三个 weight 和三个 radiation capacity|
|rcl|10%、15%、20%；一次随机 CA + 50 次实际 LS evaluation|270|每个 dataset/shape 的最佳 tolerance|
|tuning|三个代表数据集、两种箱形、三个 seed、十个配置|180|三个算法各自的 iteration 配置|
|full|15 数据集 × 2 箱形 × 3 weight × 3 radiation × 3 算法 × 5 seed|4050|算法比较结果|

共4650条计划，其中150条calibration可由历史CA导入，导入后只需4500个GPU任务。tuning和full各任务整体预算3600秒，名义合计4230 GPU-hours，另加RCL固定工作量耗时。文件加载另计；算法入口内初始化、CA、LS、kick和重构均计入预算，不执行被丢弃的CA预热。已开始的packing允许完成，因此是软上限。

CA的1800次参数实验已完成。按dataset/shape平均N−U*最低、平局时间最短，固定PES和reference range。原始JSON中的placements结合实例体积可恢复n*：`prepare --ca-json-dir PATH` 会校验并导入150条参考结果，避免GPU重跑。`run_all.sh` 自动使用存在的 `bin_packing/hpc_results/CA_jsons_results`，可用CA_JSON_DIR覆盖；没有历史目录时才重新运行reference CA。GRASP的随机位置RCL始终使用top/bottom最多16个去重点，LS使用选定range。

全部阶段的参数、数值和两个测试入口见 [EXPERIMENT_PARAMETERS.md](EXPERIMENT_PARAMETERS.md)，逐任务计划和30组容量见 `EXPERIMENT_PARAMETERS.json`。小实例全流程用 `run_smoke.sh`；最大实例32G主存压力测试用 `run_memory_smoke.sh`，二者用途独立。

调参默认代表数据集：`shapesnew`、`Merged4_normal`、`st04_Example5_normal`。它们是本流程采用的小/中/大实例方案，可用 `--tuning-datasets` 显式替换；不是按文件字节数自动推断规模。

## 一次性提交

在 AIRE 的 repo 根目录，先加载已有 Python 环境，确保最新代码、instances、data 和 CA 汇总 CSV 已上传：

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate packing

MAX_PARALLEL=3 bash bin_packing/hpc/paper_suite/run_all.sh
```

这条命令创建 `bin_packing/hpc_results/ejor_日期时间_PID/`，冻结所需源码和数据，并一次提交所有阶段的 Slurm 依赖图。默认按 16G/32G 分组，每个任务一张 GPU、4 CPU；不同内存组和数组分片串行，因此同时使用 GPU 不超过 3 张。默认 Slurm walltime 为 12 小时，用于容纳无 3600 秒限制的 reference CA 和 CA+50 LS；算法的 3600 秒预算独立设置。固定工作量任务如果超出 walltime 会失败并阻止下游，不会把不完整结果当作有效的 CA+50。

`CA param test result` 未被 Git 跟踪时，需要另行上传 CSV。自定义路径：

```bash
CA_SUMMARY=/mnt/scratch/pmywe/ca_results/summary_by_dataset.csv \
EJOR_OUTPUT=/mnt/scratch/pmywe/GPU/ejor_v1 \
MAX_PARALLEL=3 bash bin_packing/hpc/paper_suite/run_all.sh
```

仅查看将提交的命令，可以在 `run_all.sh` 后加 `--dry-run`；它会准备冻结包，但不会提交。若集群限制同时待排队任务数，不必一次提交 4650 个，可用 `--through tuning` 先提交前三阶段，后面用同一冻结包 `submit-all --pending` 接上完整实验。数组按最多 900 个本地下标分片，避免把 4050 直接用作数组索引。

阶段之间设置 CPU 汇总任务：GPU 数组全部结束（afterany）后检查每一个预期结果，仅当全部有效且通过验证才写出 selection 并正常退出。下一阶段依赖该汇总任务 afterok。失败时下游可能显示 DependencyNeverSatisfied；检查、取消这些阻塞的后续任务，再恢复。不会在参数尚未选出时提前使用默认参数运行 full。

## 分步准备与提交

```bash
python bin_packing/hpc/paper_suite/suite.py prepare \
  --ca-summary "CA param test result/summary_by_dataset.csv" \
  --output /mnt/scratch/pmywe/GPU/ejor_v1

python /mnt/scratch/pmywe/GPU/ejor_v1/driver/suite.py submit-all \
  --bundle /mnt/scratch/pmywe/GPU/ejor_v1 --max-parallel 3 --dry-run

python /mnt/scratch/pmywe/GPU/ejor_v1/driver/suite.py submit-all \
  --bundle /mnt/scratch/pmywe/GPU/ejor_v1 --max-parallel 3
```

提交选项：`--small-memory 16 --large-memory 32 --walltime 12:00:00 --partition gpu`。`chess`/`shapesnew` 默认归小内存组，其余归 32G；这是保守起始策略，不能证明每种后续 packing 状态都不会超过 32G。CPU 汇总使用集群默认分区；如需要可传 `--control-partition <cpu分区名>`。环境变量 `HPC_MODULE`、`HPC_ENV` 默认分别为 `miniforge`、`packing`，可覆盖；设为 `none` 可跳过相应操作。

冻结包可以在 Windows 本地生成，然后整个目录上传到 AIRE；包内不依赖原机器的绝对路径。冻结后不要修改 `driver`、`runtime` 或 inputs，worker 会校验所需文件 SHA256。需要改算法时创建新包，不能将新代码结果混入旧协议。

## 一次小规模流程验证

推荐直接执行 `bash bin_packing/hpc/paper_suite/run_smoke.sh`。它提交真实 GPU 的 82 个小任务，并在最终汇总后自动生成 `smoke_verification.json`。具体命令及 PASS 标准见 [SMOKE.md](SMOKE.md)。以下是等价的分步准备方式。

这用于验证提交依赖和结果格式，不用于正式参数选择：

```bash
python bin_packing/hpc/paper_suite/suite.py prepare \
  --ca-summary "CA param test result/summary_by_dataset.csv" \
  --output /mnt/scratch/pmywe/GPU/ejor_smoke \
  --datasets shapesnew --tuning-datasets shapesnew --time-limit 30 --smoke
python /mnt/scratch/pmywe/GPU/ejor_smoke/driver/suite.py submit-all \
  --bundle /mnt/scratch/pmywe/GPU/ejor_smoke --max-parallel 1
```

smoke 包只用一个 dataset、顺序 seed=3，两个箱形；reference 每形一个 seed，RCL 改为 CA+3 LS，仍覆盖三种 tolerance、十个 tuning 配置和九种容量组合。包内 `smoke=true` 明确标识，不应进入正式实验汇总。

## 监控与断点恢复

```bash
squeue -u "$USER"
sstat -j JOBID --allsteps --format=JobID,AveRSS,MaxRSS
sacct -j ARRAY_JOBID --format=JobID,State,Elapsed,ReqMem,MaxRSS,AllocTRES
```

单任务结果位于 `results/<stage>/task_<id>.json`，逐次构造/LS/kick 日志为对应 `task_<id>_events.jsonl`；日志在 `logs/`。每个结果记录输入顺序 seed、独立内部随机种子、容量、参数、N/U/U*、N−U*、构造/LS/kick 次数、RCL 大小分布、缓存命中计数、运行时间和 Linux MaxRSS。最终 packing 保存各物体的 orientation/translation，结合冻结输入可恢复，不保存庞大的完整体素布局。

自动验证物体完整性、边界、碰撞、occupancy、N/U/U*，另外验证 weight/radiation；可达性由 packing engine 检查，未独立重新验证。

若失败或中断，先用 `scancel` 取消同包所有尚在运行/等待的后续任务。`submit-all` 会拒绝与同包活动任务重叠提交。只有确认队列中同包没有任务后，才可清理中断留下的 lock：

```bash
python BUNDLE/driver/suite.py unlock --bundle BUNDLE
python BUNDLE/driver/suite.py submit-all --bundle BUNDLE --pending --max-parallel 3
```

成功且 bundle/case 身份匹配的任务跳过；失败报告与中断事件会归档后重跑。依赖选择重新汇总；没有合法结果的阶段不会输出有效 selection。若需要更多内存，可在恢复时提高 `--large-memory`。同一包禁止重叠提交。

手工汇总（允许在任务完成前查看 runs.csv，但不完整阶段返回非零）：

```bash
python BUNDLE/driver/suite.py collect --bundle BUNDLE --stage full
```

各阶段生成 `runs.csv`、`completion.json`，全部成功后生成 `selection.json`；full 另有 `summary.csv`，按 dataset/shape/weight/radiation/algorithm 聚合五个 seed 的均值和标准差。RCL 以同一 dataset/shape 下平均 N−U* 最小选择，平局记录全部候选并取较小 tolerance。Iteration 配置先算各 dataset/shape 的 seed 平均目标值，再算配置排名，最后用平均排名选择，避免大 N 数据集支配跨数据集平均；排名平局按 config_id 固定取值。完整实验保留每个实例的原始 N−U*，不将不同数据集的 N−U* 原值直接平均作为唯一总排名。

## 已检查的参数语义

* G_NI100：连续 100 次 LS 无改善才重新构造，改善清零，无总 LS 次数上限。
* G_B100/G_B125/G_B150：每轮最多 100/125/150 次实际 LS evaluation，改善不重置该总预算。
* I_K100/I_K125/I_K150：连续失败达到阈值触发 kick，无总 iteration limit；受整体 3600 秒控制。
* GI_B300_K100/K125/K150：每轮 300 次 LS evaluation，连续失败触发 kick；kick 单独计数，不占 300。持续无改善时最多 3/2/2 次 kick，最后一次 LS 也可触发 kick。
* LS 按大小和原 packing 顺序概率选物体，阶段由连续失败数决定；orientation 采用 first-improvement：找到改善即接受并重新选物体，并非每次都评估完全部 orientation。
* 两级 RCL：每个 orientation 内按同一 tau 随机抽位置，再对各 orientation 抽出的 PES 按同一 tau 过滤、随机抽 orientation；不是把全部 orientation-position 放进一个全局 RCL。两层容差可能叠加。
* geometry-only 使用 first-fit；有约束 full 使用放入物体后的 filling × radiation-left ratio 排序，并先做 weight/radiation 硬过滤。两者均用 run-scoped 状态缓存。
* 输入实例已打乱，worker 不重新 shuffle。顺序 seed 与内部搜索随机种子分离；内部种子按 time_ns、host、pid 和任务身份生成并保存。
