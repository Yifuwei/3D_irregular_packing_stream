# 三种改进版 metaheuristics 的迭代参数实验方案

## 1. 目标与实验范围

比较 improved_GRASP、improved_ILS、improved_GRASP_ILS 在不同搜索深度或 kick 阈值下的表现。正式实验每次运行的算法墙钟时间上限为 3600 秒。

本实验不是三种算法使用相同总迭代次数的比较：GRASP 调整每个起点的 LS 深度，ILS 调整连续失败后的 kick 阈值，GRASP_ILS 固定每个起点的 LS 深度并调整 kick 阈值。分别分析算法内部参数敏感性，再在相同时间预算下比较算法。

使用改进版函数，不使用旧 GRASP、ILS_from_the_first_piece 或 GRASP_ILS 替代。先采用 CA 实验中的 15 个 dataset、cube/cylinder 两种箱体及每组 5 个配对随机重复；运行前核对实例清单是否确实包含这 30 个 dataset/shape 组合。

## 2. 正式实验配置：9 组

| 配置 ID | 函数 | 每轮 LS 上限 | 连续失败 kick 阈值 | 整次运行 iteration_limit | GRASP 连续失败重启阈值 |
|---|---|---:|---:|---|---:|
| G100 | improved_GRASP | 100 | 无 | 不适用 | 150 |
| G125 | improved_GRASP | 125 | 无 | 不适用 | 150 |
| G150 | improved_GRASP | 150 | 无 | 不适用 | 150 |
| I100 | improved_ILS | 不适用 | 100 | None | 不适用 |
| I125 | improved_ILS | 不适用 | 125 | None | 不适用 |
| I150 | improved_ILS | 不适用 | 150 | None | 不适用 |
| GI300_K100 | improved_GRASP_ILS | 300 | 100 | 不适用 | 无：停滞时 kick |
| GI300_K125 | improved_GRASP_ILS | 300 | 125 | 不适用 | 无：停滞时 kick |
| GI300_K150 | improved_GRASP_ILS | 300 | 150 | 不适用 | 无：停滞时 kick |

共同配置：time_limit=3600；kick_level="medium"；GRASP 和混合算法 restart_limit=None；rcl_tolerance=0.10；random_CA_threshold=5（在启用 tolerance RCL 时保留该兼容参数）。

GRASP 显式设置 no_improvement_limit=150，避免它默认随每轮 LS 上限一起变化。由于三组 LS 上限均不超过 150，该停滞条件不会先于评估上限结束本轮。这使 G100/G125/G150 只改变每轮 LS 预算。

ILS 的 100/125/150 是 kick 阈值，不是总评估预算。iteration_limit=None，防止在第 100/125/150 次评估时提前停止。当前 improved_ILS 若设置总迭代上限，则该计数包含邻域评估和 kick；本方案不启用该上限。

混合算法每轮允许 300 次 LS；每次 kick 作为一个额外 iteration 单独计数，包括达到第 300 次 LS 时触发的 kick。全程失败、未超时且未提前结束本轮时：

| 配置 | kick 在累计 LS 次数 | kick 次数 | 本轮总 iteration |
|---|---|---:|---:|
| GI300_K100 | 100、200、300 | 3 | 303 |
| GI300_K125 | 125、250 | 2 | 302 |
| GI300_K150 | 150、300 | 2 | 302 |

接受一次改善会清零连续失败计数，实际 kick 次数可能更少。第 300 次 LS 后的 kick 结果仍参与最优解更新，但其后不再做本轮 LS。该边界行为必须在报告中说明。

## 3. 固定条件与可复现性

- 固定实例几何、物体数量、箱体尺寸、约束模式、朝向集合、packing_alg、PES、候选位置范围、accessibility 和 NFV 设置；禁止按算法单独调整。
- 初始建议使用 SCH、minimum_aabb_volume、bottom_top；若已经根据独立 CA 参数实验选定公共配置，则统一采用该配置并记录选择依据。不要用正式 metaheuristic 测试集结果再反复选择公共配置。
- 同一 dataset/shape 的 9 组配置使用相同的输入顺序及同一重复编号。每个任务单独进程，避免 RNG 和全局变量串扰。
- 重复编号使用 seed={3,7,13,19,53}。建议明确记录 order_seed=seed、piece_seed=101+seed、position_seed=202+seed、kick_seed=303+seed、global_seed=404+seed。
- improved_ILS 没有独立 kick_seed 参数，其 kick 使用全局 NumPy RNG：为它设置 np.random.seed(303+seed)。混合算法显式传 kick_seed=303+seed。其余全局随机源也要在任务入口初始化并记录。
- 不在每个 restart 或 kick 重新设种子；RNG 流随搜索推进。同种子不会保证三种算法经历同一轨迹，但能保证各配置可复现。
- 单任务使用相同型号 GPU、相同 CPU 数与线程设置；禁止多个装箱任务竞争同一 GPU。固定软件版本并保存源代码快照/提交及未提交差异。

## 4. 执行阶段

### A. 接入与小规模验证

先修改 HPC runner，使其直接调用三个 improved 函数，并显式传入所有参数。当前旧 runner/launcher 不应被当作已支持新混合函数。

为每个参数生成配置 ID，支持 --algorithm、--time-limit、--ls-limit、--kick-trigger、--restart-limit、各随机种子和结果目录。每次结果保存完整参数，避免依赖默认值或只用“iterations”一个字段表达不同含义。

计时起点使用各算法函数内部的 perf_counter；几何加载和 GPU 编译/初始化时间另记为 setup_s。为各任务执行一致的初始化流程，不对某一种算法单独预热。SLURM 的 walltime 应覆盖初始化、3600 秒搜索、一条未完成装箱调用的超时余量及结果保存；根据 smoke test 实测确定，不能直接等于 3600 秒。

选择小、中、大代表实例，对全部 9 组做功能 smoke test。短时间测试仅用于检查调用、结果保存和约束验证，不能据此选最优参数。额外使用控制流程测试确认 300/100、300/125、300/150 的边界 kick 次数。

### B. 正式配对实验

15 dataset × 2 shapes × 9 configurations × 5 seeds = 1350 个任务。

若每个任务独占一张 GPU 且完整使用一小时，搜索预算合计 1350 GPU 小时，不含初始化和超时调用。理想情况下 20 张 GPU 约需 67.5 小时，实际排队、任务长尾及重跑会增加时间。

使用 SLURM job array 和并发上限；混合安排配置，减少某类配置集中运行在特定节点或时段的偏差。超时、异常、无效布局保留状态，不当作正常结果也不静默删除。用原配置和原种子重跑基础设施失败任务，保留重跑记录；算法自身异常需先排查原因。

## 5. 每次运行应保存的结果

识别与环境：dataset、shape、bin_size、piece_count、config_id、algorithm、全部种子、源代码版本、GPU/CPU、软件环境。

配置：time_limit_s、ls_limit_per_restart、kick_trigger_failures、kick_level、total_iteration_limit、restart_limit、no_improvement_limit、PES、selection_range、rcl_tolerance、constraint_mode。

最终指标：best_N、best_U、best_U_star、best_N_minus_U_star、initial_N、initial_U_star、algorithm_wall_s、setup_s、elapsed_s、validation_ok、status、error。

搜索计数：construction_count、ls_evaluation_count、kick_count、total_iteration_count=ls_evaluation_count+kick_count、accepted_ls_count、best_found_at_s、stop_reason；构造不计入 total_iteration_count。

每轮日志：restart_id、initial_score、best_score、ls_count、kick_count、wall_s。改善事件保存时间、N、U*、N-U*、事件类型，以构建 anytime 曲线。

同时记录 construction_s、ls_s、kick_s，判断预算是否被随机构造、局部改善或扰动主导。计数与时间统计需要 runner/算法 instrumentation，新函数现有返回值不包含全部这些信息。

保存最优布局的物体 ID、朝向、位置、每个 bin 的物体体积总和，便于独立复算指标与验证。验证至少检查每件物体恰好一次、无重叠、满足箱体边界，以及本实验启用的其他约束。

## 6. 指标定义与汇总

令 u_i 为第 i 个 bin 的 filling rate，N 为所用 bin 数，使用算法现有的物体 volume 和箱体容量定义：

U = sum(u_i)/N。

U* = [sum(u_i)-min(u_i)]/(N-1)，即去除最空 bin 后的平均 filling rate；N=1 时按现有实现使用 U 作为 U* 的回退，并在结果中注明。

目标 f=N-U*，越小越好。相同 N 时，较高 U* 表示体积更集中在其余 N-1 个 bin，最空 bin 的体积更少。

每个 dataset/shape/config 汇总 5 个种子的 N、U*、f 的平均值、标准差、最佳值；同时汇总实际运行时间、构造轮数、kick 次数和最优解出现时间。主结果以平均 f 排名，并报告平均 N 和 U*，不能只展示一项指标。

跨 dataset 不把不同规模的原始 f 平均值作为唯一排名依据，因为大 N 实例会扩大尺度。先在每个 dataset/shape 内按各配置的平均 f 排名（并列取平均名次），再等权平均这些排名；cube/cylinder 分别报告，并可额外报告 Overall。原始平均 f 作为辅助信息保留。

首先比较每个算法内部的三组配置，再比较三种算法。胜/平/负按相同 dataset、shape、seed 的 f 比较；浮点相等采用绝对容差 1e-9。报告 N 的配对变化，避免只报告一个总胜率。

正式比较应在不同 seed 重复间保留配对。若做统计检验，以 dataset/shape 的种子均值为实例级观察值，不把同一实例的 5 个 seed 当作 5 个独立 dataset。算法内三配置可用 Friedman 检验；后续成对比较可用 Wilcoxon 配对检验及 Holm 多重比较校正，同时报告效果大小/实际差值。cube 与 cylinder 共用物体实例，主检验宜按 shape 分开，Overall 排名作为描述性统计。

输出 600、1200、1800、2400、3000、3600 秒的 best-so-far 曲线。缺少刚好对应时间点的事件时，取该时间点之前最后一个已完成候选的最好结果；初始构造尚未完成则标为 unavailable，不使用未来结果回填。

## 7. 判定与后续实验

不预先宣布 100、125、150 哪组最好。综合观察实例级平均排名、N、f、跨种子波动和 anytime 表现。

本轮回答：GRASP 的每轮 LS 深度、ILS 的停滞容忍度，以及固定 300 次 LS 的混合算法 kick 阈值如何影响表现。

本轮不回答混合算法的最佳每轮 LS 预算。后续若有必要，固定选定的 kick 阈值，单独测试 LS 上限 300、375、450；不要同时调整 RCL、PES、kick 强度和预算。

若正式结果用于声称某配置优于其他算法，不应只在同一批测试实例上选择参数后报告获胜配置。可以完整公开本次 9 组敏感性结果；要报告调参后的最终算法性能，则另留独立验证实例或采用预先定义的训练/验证划分。

## 8. 本文档的实施状态

这是实验方案，尚未提交或启动 HPC 任务。独立 improved_GRASP_ILS 已存在，并具备上述末次 LS 后允许 kick 的计数规则；HPC runner 接入、统一配置、搜索计数/事件日志和批量汇总仍需实现及 smoke test。
