# 实验参数总表

2026-10-16 当前配置。用途分开：小实例 smoke 验证流程；大实例 memory probe 检查 32GiB 主存；正式实验用于参数选择和算法比较。全部数据集、30 组容量和逐任务计划见 `EXPERIMENT_PARAMETERS.json`。运行后才确定的参数用选择文件表达，不提前填写假定最优值。

## 共用算法设置

|参数|值|
|---|---|
|packing_alg / selection_type|SCH / bounding_box|
|可达性检查|True|
|orientation|10 个标签：x_0；x/y/z 各 90、180、270 度|
|LS 接受|first-improvement，严格改善就接受|
|物体选择|AABB 边长大小权重 + 原 packing 顺序权重；阶段由连续失败数决定|
|位置与方向评价|统一 PES，没有独立 OES|
|RCL|位置层随机抽取，再在 orientation 层按相同 tolerance 随机抽取|
|位置候选|top/bottom 至多 16 个去重极值；新空 cube 使用极端角点|
|容差公式|PES ≤ best PES + tolerance × abs(best PES)|
|LS 的 selection range|下表按 dataset/shape 从历史 CA 结果选择；随机 CA 的 RCL 始终按 top/bottom 极值|
|alpha / random_CA_threshold|1 / 5；RCL 模式按 tolerance 过滤，旧 threshold 不替代 tolerance|
|kick level / 比例|medium / ceil(0.5 × 物体数) 个物体改变 orientation|
|rho / weight_factor|1 / 1；full 的 weight 容量另外硬检查|
|缓存|run-scoped 成功/失败 packing 缓存；first-fit 也使用缓存；随机 CA key 包含 RNG 状态|
|flag_NFV_POOL|False；不等于关闭 NFV GPU 计算，也不等于关闭 packing 状态缓存|
|density / axis / encourage_dbl|5 / z / True|
|visualisation / TRACE|False / False|
|无约束阶段|first-fit；max_weight=箱体积；max_radio=输入辐射总量+1，实际不限制|
|full 阶段|weight/radiation 硬过滤；按放入后的 filling × remaining radiation ratio 排箱|
|内部随机种子|worker 启动时由 time_ns/host/pid/task 派生四个种子并保存；不是顺序 seed|
|输入 shuffle|False，instances 已按顺序 seed 排好|
|预热|不执行被丢弃的完整 CA；算法内初始化/JIT/构造/LS/kick/重构计入预算|
|时间边界|软上限；已开始的 packing 完成并评价，再停止|

## 各阶段正式实验参数

|阶段|数据集 / 箱形 / 顺序 seeds|工作量或时间|参数候选|任务数|
|---|---|---|---|---:|
|CA 参数结果|15 / cube+cylinder / 3,7,13,19,53|已完成|4 PES × all,bottom,bottom_top|1800 历史结果|
|calibration|15 / cube+cylinder / 3,7,13,19,53|导入 reference CA，不再执行 GPU|每 dataset/shape 选历史平均 N−U* 最低配置，平局选更短时间|150 条历史 reference、30 组容量|
|RCL|15 / cube+cylinder / 3,7,13|一次随机 CA + 50 次 LS；restart_limit=1；无算法秒数限制、Slurm 默认12h|tau=0.10,0.15,0.20，位置与 orientation 相同|270|
|Iteration|shapesnew,Merged4_normal,st04_Example5_normal / 两种箱形 / 3,7,13|每任务整体3600秒|下表十组配置|180|
|full|15 / cube+cylinder / 3,7,13,19,53|每任务整体3600秒|三个已调参算法 × 三 weight × 三 radiation；继承已选 tau 和 iteration|4050|

三种算法的所有 iteration 配置：

|算法 / config_id|连续无改善阈值|每轮 LS 总预算|kick|restart_limit|
|---|---:|---:|---|---|
|GRASP G_NI100|100，改善清零|无|无|无，由整体时间限制|
|GRASP G_B100|100|100，改善不重置预算|无|无|
|GRASP G_B125|125|125|无|无|
|GRASP G_B150|150|150|无|无|
|ILS I_K100|100|无额外 iteration_limit|达到阈值，medium|不做随机重构|
|ILS I_K125|125|无额外 iteration_limit|达到阈值，medium|同上|
|ILS I_K150|150|无额外 iteration_limit|达到阈值，medium|同上|
|GRASP_ILS GI_B300_K100|100|300 LS/轮|达到阈值，medium|无|
|GRASP_ILS GI_B300_K125|125|300 LS/轮|达到阈值，medium|无|
|GRASP_ILS GI_B300_K150|150|300 LS/轮|达到阈值，medium|无|

kick 单独计数，不占 hybrid 的 300 次 LS；始终失败时每轮最多 3/2/2 次 kick。单箱 LS 可提前结束。RCL 阶段只有一次构造、不 kick；no_improvement_limit=ls_evaluation_limit=50，以实际邻居评价次数计数。

RCL 按 dataset/shape 的 seed 平均 N−U* 最低选择；平局保留并列值并取较小 tau。Iteration 按 dataset/shape 内的 seed 平均目标值排名，再按各配置平均排名最低选择；排名相同按 config_id 取值。选值写入 `results/rcl/selection.json` 和 `results/tuning/selection.json`，full 启动时读取。

正式 GPU worker：1 GPU、4 CPU；chess/shapesnew 为16G，其他32G；全流程默认最多并行3；Slurm walltime默认12h/任务。CPU 汇总1 CPU、2G、30分钟。历史结果导入后实际GPU任务为4500个。

## 小实例 smoke 全部参数

|参数|cube|cylinder|
|---|---|---|
|dataset / 顺序 seed|shapesnew / 3|shapesnew / 3|
|物体数|43|43|
|网格尺寸|16×16×16|19×19×19|
|箱体积|4096|π×9.5²×19|
|物体总体积|9028|9028|
|PES|minimum_aabb_edges_len|maximal_residual_box|
|reference CA / LS range|bottom_top|bottom|
|无约束 max_radio|6409.351782315826|8156.058298875396|
|无约束 max_weight|4096|π×9.5²×19|

|阶段|所有设置|任务数|
|---|---|---:|
|calibration|每箱形1次新CA，ALG=fixed_CA，iteration_limit=0，time_limit=None；使用seed=3；得到smoke专用三档容量|2|
|RCL|tau=.10/.15/.20；restart_limit=1；总LS=3、无改善阈值=3；不kick；无算法秒数限制|6|
|Iteration|上述十种配置全部保留；整体time_limit=30秒；读取smoke选出的tau|20|
|full|三算法、三weight×三radiation；整体30秒；读取smoke容量及选定配置|54|

总82任务。每任务1GPU、4CPU、16G；最多并行1；Slurm walltime=01:00:00。CPU汇总1CPU/2G/30分钟。最终自动输出 `smoke_verification.json` 的 PASS/FAIL。短预算可能未达到kick阈值，不能由此判断正式实验的最优参数。

默认小包重新执行两个很小的CA，以验证完整运行链；容量因此基于一个seed，**不同于正式五seed容量**。需要测试历史导入路径可给run_smoke额外设置CA_JSON_DIR；这时calibration导入10条历史reference，不用GPU，剩余80个GPU任务不变。

## 大实例 32GiB 主存测试全部参数

|参数|值|
|---|---|
|dataset / instance|st04_Example5_normal / st04_Example5_normal_seq3_cylinder_999999_1.txt|
|顺序 seed / 物体数|3 / 50|
|箱形 / 网格|cylinder / 180×180×180|
|箱体积|4580442.088933918|
|物体总体积|1740615|
|输入稠密数组单元数|291600000，当前15数据集中最大|
|PES / LS range|minimum_aabb_edges_len / bottom|
|约束|geometry-only，first-fit；max_weight=4580442.088933918，max_radio=11205.908318087153|
|GRASP|NI100；tau=.15固定；无总LS cap；无restart cap|
|ILS|kick=100；iteration_limit=None；medium kick，改变25个物体|
|GRASP_ILS|每轮300 LS；kick=100；tau=.15固定；medium kick，改变25个物体；无restart cap|
|三任务时间|各整体3600秒，包含CA，不做额外完整预热|
|资源|每任务1GPU、4CPU、**32G主存硬限制**；array=0-2%1，串行|
|Slurm walltime|06:00:00/任务，允许已开始packing完成|
|tolerance/iteration来源|为资源测试固定的代表配置，不声称性能最优|
|输出|逐算法JSON中的host_max_rss_kib、counts、耗时；GNU time -v；Slurm sacct|

这个入口只有3个GPU任务，不提交calibration/RCL调参/full容量网格。用于测试一小时搜索和缓存增长时的内存；不能证明所有带weight/radiation配置都不超过32G，也不能只根据某个任务成功就宣布三算法全部通过。

原选定CA在该dataset/cylinder的五seed平均耗时约1117.89秒，因此用30秒预算不适合检查完整LS运行的内存。

## 已恢复的该大实例正式容量数据

来自五seed历史CA，PES=minimum_aabb_edges_len，range=bottom，非memory测试实际启用的约束：

|标定量|值|
|---|---:|
|平均U*|0.2207354181908939|
|平均n*|34.6|
|最大单体体积|91920|
|五seed最大单体radiation|1096.7582033835115|
|radiation基准BR|7016.758203383512|

|强度|weight绝对容量，factor=1|weight/箱体积|radiation容量|
|---|---:|---:|---:|
|strong|505532.9|0.11036770909544695|7016.758203383512|
|medium|758299.35|0.16555156364317042|10525.137305075268|
|weak|1011065.8|0.2207354181908939|14033.516406767025|

公式：weight比例=max(c×平均U*, 最大单体体积/箱体积)，c=.5/.75/1；BR=rmax+1000×max(0,.2n*−1)，radiation倍率=1/1.5/2。完整30组数值包含在JSON中。

## 当前运行命令

激活AIRE packing环境，在repo根目录：

```bash
# 小实例完整流程
bash bin_packing/hpc/paper_suite/run_smoke.sh

# 大实例32G内存测试，三算法各3600秒
bash bin_packing/hpc/paper_suite/run_memory_smoke.sh
```

内存测试在 `bin_packing/hpc` 目录也可以用 `bash paper_suite/run_memory_smoke.sh`。与旧 `submit_memory_probe.sh` 区分：新入口使用当前无丢弃CA预热的paper_suite worker。

脚本打印BUNDLE和JOBID；查看实时内存或最终记账（将JOBID替换为实际值）：

```bash
sstat -j JOBID_0 --allsteps --format=JobID,AveRSS,MaxRSS
sacct -j JOBID --units=G --format=JobID,State,Elapsed,ReqMem,MaxRSS,ExitCode
python BUNDLE/driver/memory_probe.py check --bundle BUNDLE
```

三任务依次对应GRASP、ILS、GRASP_ILS。实时查看时将数组下标改为正在运行的0/1/2。GNU time报告在 `logs/memory_0_time.txt` 等。Slurm状态OUT_OF_MEMORY或cgroup主存OOM记录说明32G主存不足；CUDA显存OOM需要另行区分。只看GPU显存或VIRT不能判断主存够不够。成功结果还需查看MaxRSS与32G的余量，不能把贴近上限视为充足余量。
