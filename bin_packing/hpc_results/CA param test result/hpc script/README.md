# HPC：nesting strategy × selection range

本目录接入当前仓库的 `bin_packing/src`，默认对所有实例去除 radio/weight 约束并合并重复，运行 `fixed_CA`，以比较选点策略和候选范围，箱体尺寸直接读取实例。`open_dimension` 的条带装箱接口不同，未接入此入口。

## 参考仓库的运行逻辑

参考 `C:/work/NFV_POOL_TESTING/test/job_submit.sh`、`code_launcher_hpc.py` 和 `code_launcher_hpc_constructive_only.py`：

1. 参数生成器枚举实例、算法、随机种子和策略等组合，生成文本参数表，每行一次实验。
2. `sbatch` 提交 Slurm job array；数组编号从 0 开始，参数行号从 1 开始。
3. 每个任务申请 1 GPU、4 CPU、240G 内存、3 小时；参考脚本限制同时运行 8 个任务。
4. 节点加载 `miniforge`，激活 `packing` 环境，限制 OMP/MKL/OpenBLAS 线程数。
5. 参考脚本通过 `sed` 取出对应参数行，`srun` + `numactl` 启动 Python `run_one`，运行一次装箱。
6. 每个任务写独立 JSON，Slurm 为每个任务写独立 stdout/stderr。

当前实现保留这个结构：

| 文件 | 用途 |
| --- | --- |
| `prepare.py` | 生成实验矩阵、`all_instances.txt` 和实验清单 `all_instances.manifest.json` |
| `submit.sh` | 计算数组长度，提前创建日志目录，冻结参数副本，调用 sbatch |
| `job_submit.sh` | Slurm 资源配置、Conda 激活、线程限制、srun 启动 |
| `task.py` | 根据 0-based task ID 取一行，使用 shlex 解析参数 |
| `run.py` | 加载实例及其箱体尺寸、调用当前算法、验证并写结果 |
| `collect.py` | 按实验清单汇总 CSV，包括未完成、失败和非法结果 |

与参考脚本相比，数组长度不再写死为 `0-33`，不依赖从 test 目录启动；参数使用 Python 解析，支持带空格的路径。资源、邮件通知及 NUMA 配置与参考脚本保持一致。

## 第一次运行

将整个仓库上传至 HPC，保留 `bin_packing/data`、`instances`、`src` 目录。在仓库根目录执行：

```bash
module load miniforge
conda activate packing
python -m pip install -r lib_requirements.txt
# 仓库已附全量 all_instances.txt；仅在不存在时用 python hpc/prepare.py 生成。
python hpc/task.py hpc/all_instances.txt 0 --dry-run
bash hpc/submit.sh hpc/all_instances.txt 8
```

`packing` 环境需预先创建，且其 Numba/CUDA 组合需能访问节点驱动。已有可用环境时无需重复安装依赖。Python 需为 3.10 或以上。`--dry-run` 只需要 Python 标准库：检查实例文本、文件路径、配置，不导入科学计算库，不检查 CUDA 或读取完整体素。

默认扫描 `bin_packing/instances/*.txt` 的全部文件，合并几何等价实例。当前 391 个文件合并为 150 组，生成 **1,800 个任务**：

- 150 组实例：15 个数据集 × 5 种物体顺序 × 2 种容器类型。

- 4 种策略：`minimum_aabb_volume`、`minimum_aabb_edges_len`、`maximal_residual_box`、`overlap_distance`。
- 3 种范围：`bottom`、`bottom_top`、`all`。
箱体尺寸不是额外实验维度，每个实例只有文件表头给出的一个尺寸。

例如 chess_seq13_cube 直接使用 `56×56×56`，不做缩放或尺寸覆盖。保持物体顺序、物体体素和体积不变；所有物体 radio 和容器 max_radio 置为 0，rho 固定为 1。这样 radio 检查恒通过，weight 不再产生额外填充率限制；正常几何体积和碰撞检查仍然保留。实例名中的 `seq13` 决定既存物体顺序；算法 RNG 单独固定为 42。

旧的 `params.txt` / `params.manifest.json` 仅是单实例的 12 任务示例，当前默认入口使用 `all_instances.txt`。

生成器不会覆盖已有参数表。新增实验请给出新的 `--params` 和 `--output-dir`，避免和正在排队的实验混淆。

## instances 文件格式与去重规则

当前目录共 391 个 `.txt`：150 个旧格式文件（两字段表头），241 个扩展格式文件（四字段表头）。

```text
cube    (56, 56, 56)
cube    (56, 56, 56)    999999    1
../Chess/classic_pawn_extracoarse_nh.binvox    2245    3.788159...    780
```

实际字段用 Tab 分隔。表头为容器类型、尺寸，以及可选的 max_radio、weight/rho；物体行为文件路径、体积、radio、piece_type。文件名中的 `_seq13` 表示已存储的物体顺序，`_cube`/`_cylinder` 表示容器，`_2000_0.2` 等后缀表示约束变体，不是新几何。

忽略约束后，按“容器类型、原始尺寸、按顺序排列的物体路径/体积/类型编号”分组，绝不合并不同顺序或不同容器。Merged4_normal 有 110 个文件，其余多数数据集各 20 个；shapesnew 有额外的 COMPLETE 副本。共得到 150 组不同几何实例。

`instance_inventory.json` 保存本次完整盘点及分组。`all_instances.manifest.json` 的每个任务含 `source_instances`，保留它代表的全部原文件名；汇总 CSV 同样保留这些映射。重复项只计算一次，不删除或更改任何原始实例文件。

`prepare.py` 默认重新扫描并去重；`--instances ...` 可限制到指定实例，`--keep-duplicates` 可显式逐文件运行。分组以路径及元数据为依据，并不识别不同路径但二进制内容相同的物体文件。

## 自定义实验矩阵

```bash
python hpc/prepare.py \
  --instances chess_seq13_cube_999999_1 chess_seq19_cube_999999_1 \
  --strategies minimum_aabb_volume overlap_distance \
  --ranges bottom bottom_top all \
  --params hpc/chess_sweep.txt --output-dir hpc/results/chess_sweep

bash hpc/submit.sh hpc/chess_sweep.txt 8
```

当前旋转实现依赖等长数组轴，因此入口明确要求 `X=Y=Z`。圆柱实例也使用等长体素数组，容器类型由实例第一行决定；例如可选 `chess_seq13_cylinder_999999_1`。入口不会把过大的物体裁成较小物体；原始方向超出数组尺寸时记录失败。尚未加入“先换一个初始方向再装箱”的逻辑。

矩阵生成器将 `--oes` 设置为与 `--pes` 相同。在当前 `packing_iter_ls.py` 的 SCH 分支中，跨方向比较使用 SC 返回的评分；`--oes` 并不是这一路径上完全独立的实验变量。

## 先试跑一个任务

在已获得 GPU 的交互式作业中运行：

```bash
python hpc/task.py hpc/all_instances.txt 0
```

或直接运行单次实验：

```bash
python hpc/run.py --instance-id chess_seq13_cube_999999_1 \
  --pes overlap_distance --oes overlap_distance --range bottom_top \
  --output hpc/results/manual.json
```

`run.py` 还支持 `ILS`、`GRASP` 等现有算法，但矩阵生成器固定 `fixed_CA`。算法时间参数是内部迭代限制，不是强制进程超时；尤其初始构造可能超过它，最终硬时限由 Slurm 控制。

## 集群配置

`job_submit.sh` 沿用参考脚本的资源默认值，请按实际集群调整。提交时可以覆盖：

```bash
bash hpc/submit.sh hpc/all_instances.txt 4 --partition=gpu --mem=64G --time=02:00:00
```

可通过环境变量修改环境名称，无需编辑脚本：

```bash
HPC_MODULE=miniforge HPC_CONDA_ENV=packing bash hpc/submit.sh hpc/all_instances.txt 8
```

使用已配置环境时，设置 `HPC_MODULE=none HPC_CONDA_ENV=none`。默认与参考脚本一致，加载 miniforge、读取 `~/.bashrc`、激活 packing，并启用 `numactl --interleave=all`；若节点没有 numactl，可设 `HPC_NUMACTL=0`。

邮件配置沿用参考文件：`BEGIN,END,FAIL` 发至 `pmywe@leeds.ac.uk`。这里只配置通知，尚未提交任何作业。可在提交时追加 `--mail-type=NONE` 禁用，或 `--mail-user=你的邮箱` 覆盖地址。

从 Windows 上传脚本时保留 LF 换行，用 `bash hpc/submit.sh` 启动。提交器将参数表复制到 `hpc/submissions/`，队列中的任务始终读取该副本。

默认 `HPC_ARRAY_SIZE=1000`，1,800 个任务拆为 2 个数组（1000 + 800）。各数组通过 `afterany` 依次启动，因此全局同时运行数量仍由第二个参数控制；上一批有失败任务也会继续下一批。脚本将全局任务偏移传给作业入口，并在 `*.jobs.tsv` 保存 Job ID、偏移及长度。若集群的 MaxArraySize 更小，例如设 `HPC_ARRAY_SIZE=500 bash hpc/submit.sh hpc/all_instances.txt 8`。队列的总提交数量限制仍需以实际集群为准。

## 结果、检查与重跑

```bash
python hpc/collect.py hpc/all_instances.manifest.json --output hpc/summary.csv
```

CSV 按 task ID 给出代表实例、对应全部源文件名、geometry_only 模式、策略、范围、箱体尺寸、`best_N`、`best_U`、`best_U_star`、用时和状态，便于比较装箱数量、利用率及运行成本。

同一命令同时生成 `hpc/summary_by_dataset.csv`，按 dataset、bin shape、原始箱体尺寸、nesting strategy、selection range 和约束模式分组。当前为 15 × 2 × 4 × 3 = 360 行，每组对应实例顺序种子 3、7、13、19、53；不是算法 RNG（后者固定为 42）。

对装箱数、U、U_star、初始解指标和两种耗时分别输出 `_mean`、`_std` 和 `_n`。标准差为样本标准差（ddof=1）；不足两条有效数据时标准差留空。仅统计 status=success 且 validation_ok=true 的实验，缺失/失败不按零填充，指标为 null 时单独排除。`complete` 表示预期的五个种子全部成功，`success_count` 和 `missing_or_unsuccessful_seeds` 标出完成情况。不完整组的均值仅代表已成功的种子；应筛选 complete=true 且对应指标 _n=5 后进行完整五种子比较。可通过 `--stats-output` 指定统计表路径。

每个 JSON 保存完整参数、Slurm ID、Git commit、实际尺寸、指标、每件物体的方向和平移及检查结果。Git commit 不包含工作区未提交修改；实验归档时请同时保存源码快照。

验证使用原始物体和最终方向/平移重建未取模坐标，检查矩形边界、圆柱体素掩码、重叠、体素阵列一致性及物体数量。它不验证整个可达运动路径，也不是连续网格几何验证。

- `success`：算法完成且布局验证通过。
- `invalid`：算法返回但布局验证失败，任务退出码为 1。
- `failed`：Python 运行异常，保存 traceback，任务退出码为 1。
- `missing`：尚无 JSON，可能尚未运行、被 Slurm 超时/OOM 终止或在初始化时失败；检查对应日志。

生成的任务包含 `--resume`，仅跳过参数匹配的 `success`（配置包含 geometry_only 标记，不会复用旧的受约束结果）；参数匹配的 `failed`/`invalid` 可重跑。不要同时提交两份相同参数表到相同输出目录。默认不会将结果写入原有 `bin_packing/result`。

## 本地验证

```bash
python -m unittest discover -s hpc/tests -v
```

测试覆盖矩阵数量、尺寸、任务行映射、算法参数对接、结果汇总及越界识别；不会运行真正的 CUDA 装箱或提交 Slurm。
