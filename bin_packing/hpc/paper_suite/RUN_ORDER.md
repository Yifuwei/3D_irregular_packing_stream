# 实验顺序与运行命令

本说明对应 2026-10-16 的实现。所有命令在 AIRE repo 根目录执行。使用正式数据前，需要将本地更新的代码与 CA 汇总 CSV 上传到 AIRE；本地文件修改不会自动出现在远端。

## 顺序和各阶段的自动输出

|顺序|阶段|正式任务数|如何决定参数|成功后写出|
|---|---|---:|---|---|
|0|读取已完成的 CA 参数结果|不重跑 1800 次|每 dataset/shape 取最低平均 N−U*，平局取更短时间，冻结 PES/reference range|`manifest.json` 的 `ca_parameters`|
|1|容量标定 calibration|150|15 dataset × 2 shape × 5 顺序 seed 的 reference CA，按 U*/n* 计算三档容量|`results/calibration/selection.json`|
|2|RCL tolerance|270|15 × 2 × 3 seed × 3 tolerance；每次随机 CA + 50 LS；每 dataset/shape 选最低平均 N−U*|`results/rcl/selection.json`|
|3|Iteration tuning|180|3 代表 dataset × 2 shape × 3 seed × 10 配置，各 3600 秒；按 dataset/shape 内的 seed 均值排名，再选平均排名最低的配置|`results/tuning/selection.json`|
|4|完整实验 full|4050|读取已选 tolerance、iteration 和容量；15 × 2 × 5 seed × 3 算法 × 3 weight × 3 radiation，各 3600 秒|`results/full/runs.csv`、`summary.csv`|

四种 GRASP 配置：NI100、B100、B125、B150；三种 ILS 配置：K100、K125、K150；三种 hybrid 配置：B300_K100、B300_K125、B300_K150。全部使用 first-improvement LS。

容量标定与 RCL 在科学上互不依赖；当前脚本串行执行它们，以控制 GPU 并行总数。Iteration 阶段中的 ILS 不需要 tolerance，但统一等待 RCL 阶段完成，便于整套实验管理。无约束阶段使用 first-fit；full 启用 weight/radiation 硬约束和 bin priority。

## 为什么可以提前提交全部任务

提前提交的是 Slurm 作业和依赖关系，并没有提前替未来实验填入一个默认 tolerance。

```text
calibration GPU 数组
    ↓ afterany：数组结束后运行汇总检查
calibration CPU 汇总：完整性验证 → 保存容量 → 成功退出
    ↓ afterok：只有上一步成功才释放
RCL GPU 数组
    ↓ afterany
RCL CPU 汇总：验证全部 270 结果 → 自动选择 tolerance → 写 selection.json
    ↓ afterok
iteration GPU 数组：启动时读取 RCL selection.json
    ↓ afterany
iteration CPU 汇总：自动选择三个算法的 iteration 配置 → 写 selection.json
    ↓ afterok
full GPU 数组：启动时读取 tolerance、iteration、容量三个 selection.json
    ↓ afterany
full CPU 汇总：逐实例及分组结果 CSV
```

`suite.py::submit()` 用 `sbatch --parsable` 获取作业 ID，拼接 `--dependency=afterany:...` 和 `--dependency=afterok:...`。阶段结束后 `suite.py::collect()` 检查每个预期 task 的身份、成功状态和验证标记，全部有效才选参数。缺失/失败会非零退出，下游不会执行。

参数选择使用 `protocol.py::select_tolerances()` 和 `select_tuning()`。真正启动 GPU worker 后，`worker.py::read_selections()` 从磁盘读取刚产生的 JSON，`settings_for()` 再把选值填入算法调用。例如 RCL 选出某 dataset/cube 的 0.15，后续 GRASP 与 GRASP_ILS 在该 dataset/cube 就使用 `rcl_tolerance=0.15`。ILS 不使用 RCL。

RCL 平局保留所有并列 tolerance，确定性取较小值；iteration 平均排名平局按 config_id 取值。所有候选原始结果均保留，因此可以检查自动选择依据。

## 环境与冻结包

```bash
cd /mnt/scratch/pmywe/GPU/3D_irregular_packing_stream
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate packing
```

确认 `CA param test result/summary_by_dataset.csv` 在远端存在。默认 CA 汇总目录不一定随 Git 上传，可以自己指定 CSV 路径。

创建正式包（目录必须是全新的；不要指向旧结果）：

```bash
export BUNDLE=/mnt/scratch/pmywe/GPU/ejor_paper_20261016_v1
python bin_packing/hpc/paper_suite/suite.py prepare \
  --ca-summary "CA param test result/summary_by_dataset.csv" \
  --output "$BUNDLE"
```

准备只会复制所需源码和输入、生成清单，不使用 GPU。可以先查看正式提交图：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --max-parallel 3 --dry-run
```

## 方式 A 一次提交全部

已经 prepare 好的包：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --max-parallel 3
```

如果想用一条命令同时 prepare + submit，则不用前面的 prepare，而执行：

```bash
EJOR_OUTPUT=/mnt/scratch/pmywe/GPU/ejor_paper_20261016_v1 \
MAX_PARALLEL=3 bash bin_packing/hpc/paper_suite/run_all.sh
```

这两条是替代方案，不要对同一个包重复提交。不同内存组和数组分片串行，最多同时 3 个 GPU worker；小组 16G、其他 32G，每个 worker 1 GPU/4 CPU。超过账户待提交任务数限制时，可以用下面的分步方式。

## 方式 B 分步提交

使用同一个正式 `BUNDLE`。每一步等待该阶段 GPU 任务和 CPU 汇总任务完成，再执行下一步；不要在同包任务仍排队时重复提交。

第一步，只提交容量标定：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --through calibration --pending --max-parallel 3
```

确认 `results/calibration/completion.json` 的 incomplete 为空，并生成 selection.json。第二步：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --through rcl --pending --max-parallel 3
```

第三步：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --through tuning --pending --max-parallel 3
```

第四步：

```bash
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --through full --pending --max-parallel 3
```

`--through` 包括之前的阶段；`--pending` 跳过已成功的 GPU 任务，只重新做轻量汇总检查并补交后续任务。通常不需要手工 collect，Slurm CPU 汇总任务会自动运行。

查看选出的 tolerance 和 iteration：

```bash
python -m json.tool "$BUNDLE/results/rcl/selection.json"
python -m json.tool "$BUNDLE/results/tuning/selection.json"
```

## 每阶段已经做过哪些测试

2026-10-16 本地合并运行 63 项 CPU 测试，全部通过。没有连接 AIRE，不能将这些测试表述为完成了真实 GPU 实验。

|阶段|已实际运行的本地测试|结果|
|---|---|---|
|准备|冻结文件、输入复制、SHA256、完整计划 150/270/180/4050、数组分片和资源依赖|通过|
|calibration|三档容量公式、最大单体下界、完整阶段汇总写出 selection|通过|
|rcl|两级 RCL、负/零 PES、cube/cylinder 新空箱、真实 controller CA+50；模拟 cube=15%/cylinder=20% 的结果能写出不同最优值|通过|
|tuning|GRASP 四组、ILS 无额外 iteration cap、hybrid 300 LS 与 kick 分离；实际读入上一步选择文件|通过|
|full|模拟完整参数交接：GRASP B125、ILS K150、hybrid B300_K125、对应 tolerance 与容量；汇总全部 54 个 smoke 配置生成 CSV|通过|
|失败／恢复|RCL 缺少一个结果不生成 tolerance、下游读取失败；跳过成功任务、依赖 afterok/afterany 检查|通过|

表中的 15%/20% 和 B125/K150/B300_K125 是测试人为设定的期望值，不是目前真实实验得出的最优参数。四阶段交接测试执行真实的汇总、JSON 写读和算法参数组装，仅 GPU 搜索结果使用模拟数据。

复现流程测试：

```bash
python -m unittest discover -s bin_packing/hpc/paper_suite -p test_suite.py -v
python -m unittest discover -s bin_packing/test -p test_paper_controllers.py -v
```

## AIRE 上逐阶段验证的小包

这个小包运行真实 GPU，覆盖四阶段的自动参数选择与交接。与正式包分开，禁止混入正式结果：

```bash
export SMOKE=/mnt/scratch/pmywe/GPU/ejor_flow_smoke_20261016_v1
python bin_packing/hpc/paper_suite/suite.py prepare \
  --ca-summary "CA param test result/summary_by_dataset.csv" \
  --output "$SMOKE" --datasets shapesnew --tuning-datasets shapesnew \
  --time-limit 30 --smoke
python "$SMOKE/driver/suite.py" submit-all \
  --bundle "$SMOKE" --max-parallel 1
```

共 82 个小任务：calibration 2、RCL 6、tuning 20、full 54。一个 dataset、seed=3、两个 shape；RCL 为 CA+3 LS，其余有时间预算的任务为 30 秒。验证文件、CUDA、Slurm 和参数交接；短预算不保证真实搜索触发 100/125/150 次 kick，阈值控制流已由 CPU 测试覆盖。

完成后检查四个 `results/<stage>/completion.json`：success 应分别是 2、6、20、54，incomplete 均为空；full 应产生 summary.csv。确认小包成功后，再提交正式包。

## 失败时

检查 `logs/`、失败的 task JSON 以及 completion.json；先取消同包所有活动/等待任务。确认没有同包任务后清理中断 lock，再补交：

```bash
python "$BUNDLE/driver/suite.py" unlock --bundle "$BUNDLE"
python "$BUNDLE/driver/suite.py" submit-all \
  --bundle "$BUNDLE" --pending --max-parallel 3
```

如果是内存不足，可恢复时增加 `--large-memory`。如果要改算法源码或实验协议，准备一个新包；不能修改冻结 runtime 后接着混跑旧结果。
