# 简化版 metaheuristic 迭代参数实验

每次任务使用一张 GPU，直接调用独立的 improved_GRASP、improved_ILS、improved_GRASP_ILS，无旧算法的预算适配或单箱强制继续搜索。

默认：shapesnew、Merged4_normal、st04_Example5_normal；cube/cylinder；seed=3/7/13；每次3600秒，共162个任务。三个数据集按已有CA平均耗时的最小、中位、最大选择，不代表已验证的metaheuristic难度分级。

| 算法 | 配置 |
|---|---|
| improved_GRASP | LS每轮100/125/150，停滞重启阈值统一150 |
| improved_ILS | kick阈值100/125/150，总iteration_limit=None |
| improved_GRASP_ILS | 每轮300次LS，kick阈值100/125/150 |

混合算法kick计入额外iteration，不消耗300次LS额度。第300次LS后允许满足阈值的kick，随后重启。持续失败时三组分别最多3/2/2次kick。

公共设置：SCH、minimum_aabb_volume、bottom_top、accessibility=True、medium kick、RCL tolerance=0.10。输入使用既有 *_999999_1.txt 实例；实际radio/rho由该实例读取。输入顺序使用对应seq seed。内部搜索种子在每个任务启动时由time.time_ns()、进程ID、主机名和任务ID混合生成，与顺序seed无关；piece/position/kick/Python全局随机源使用各自生成的种子，实际值和时间戳保存到task_N.json。单独ILS的NumPy全局kick随机源使用生成的kick种子。每轮重启不重新设种子。

因此重复运行同一个任务也会产生新的内部搜索轨迹。配置间仅输入顺序配对，内部搜索随机流不再配对；三个种子对应三次“不同输入顺序+独立内部随机搜索”重复。复现某次搜索需要使用结果JSON中的实际内部种子，不能只使用顺序seed。

## 准备与提交

将当前worktree的相关src、test、hpc文件及原始几何/instances同步到HPC项目目录，保持仓库相对目录。PROJECT_DIR指向bin_packing，不是仓库根目录。准备过程检查所有实例及几何存在、记录源码和输入哈希；提交后不能修改这些文件，否则任务会拒绝运行。此脚本不自动创建源码快照，请保留上传版本。

```bash
cd /path/to/repository/bin_packing/hpc
export PYTHON_BIN=/path/to/cuda/environment/bin/python
MAX_PARALLEL=20 bash submit_iterations.sh
```

沿用既有集群默认值：gpu分区、miniforge模块、packing环境、4 CPU、240G内存、1 GPU、SLURM walltime=6小时。3600秒是算法预算；六小时allocation给加载、统一构造预热及不可中断调用留余量，可通过sbatch选项调整。

```bash
# 只生成manifest，不调用sbatch
DRY_RUN=1 bash submit_iterations.sh

# cube-only，81任务
SHAPES=cube MAX_PARALLEL=20 bash submit_iterations.sh

# 功能smoke：一个实例、一个seed、9配置、30秒搜索
DATASETS=shapesnew SHAPES=cube SEEDS=3 TIME_LIMIT=30 bash submit_iterations.sh

# 自定义分区/内存/调度walltime
bash submit_iterations.sh --partition=gpu --mem=240G --time=06:00:00

# 无module/conda/numactl的环境
HPC_MODULE=none HPC_CONDA_ENV=none HPC_NUMACTL=0 bash submit_iterations.sh
```

仅准备和汇总不需要CUDA/NumPy等依赖，计算任务需要原项目完整GPU环境。统一执行一次构造预热，不计入算法时间；后续搜索调用可能仍有首次编译费用，它们计入搜索时间。所有候选完成后的观察/日志开销也计入算法时间。

## 汇总

```bash
python metaheuristic_iterations.py collect \
  --manifest /path/to/results/manifest.json --output /path/to/results
```

- task_N.json：配置、种子、最终N/U/U*、N-U*、初始指标、LS/kick/构造次数与耗时、最优解出现时间及布局验证。
- task_N_events.jsonl：每次更好的已完成候选的时间和目标值，可用于anytime曲线。
- iteration_runs.csv：所有任务，保留missing/failed/unreadable状态。
- iteration_summary.csv：dataset/shape/config的均值及标准差，附success_count/complete。未完成组的均值仅来自成功任务，不能当完整配对实验用于排名。
- iteration_ranking.csv：只纳入9配置全部完成的dataset/shape组合，按平均N-U*排名（并列平均名次），分别汇总cube、cylinder及Overall；没有完整组合时明确标记unavailable。

U*按去除最空bin后的平均filling rate独立复算；单箱回退为U。验证物体完整性、朝向/平移与布局一致、箱体边界、无重叠、占用网格以及N/U/U*。accessibility由装箱引擎执行，未在runner中独立重新验证。

时间到期时正在执行的调用允许完成；algorithm_wall_s可能超过3600秒。事件也可能在3600秒后出现，绘制3600秒anytime结果时不能使用这些未来事件。

结果文件已存在时禁止覆盖。基础设施失败可将原任务结果及日志移到归档目录后，仅重跑相应array index；保留旧记录。collect会保留失败与缺失，不能静默删除。

默认最重实例的CA构造本身平均耗时约1059秒。短smoke仍可能等待构造很久；一小时内也可能很少kick。要结合实际计数解释参数效果，不能因为阈值写成100就认为已发生kick。

本地可运行无GPU测试：`python -m unittest discover -s hpc -p test_metaheuristic_iterations.py`。实际GPU性能与集群环境需在HPC smoke中验证；脚本不会自动提交后续验证实验。
