# 方法与实验要求核对

核对来源：2026-10-16 重新读取的 `EJOR paper update summary.docx`，包括 Word 原生公式。文档只作为方法和实验规格读取，没有修改 Word。下表区分原有实现与本次补充。

|要求|原实现状态|本次处理|
|---|---|---|
|PES 同时评价位置与 orientation，取消独立 OES|三个 improved controller 已支持统一 PES；部分旧驱动仍有历史路径|新 suite 固定使用共享 improved controller 路径，配置从已完成 CA 汇总冻结|
|大物体加权，早期偏前/后期偏后，连续失败决定阶段；找到第一个改善即接受|已有 `improve_pieces_selection_ls` 和 first-improvement LS|已符合；保留并验证三个 controller 的真实函数体，用户再次确认采用 first-improvement|
|有 weight/radiation 时按放入后的 filling × radiation-left ratio 排箱，无约束时 first-fit|有约束共享引擎已实现；无约束走旧的另一条路径|共享引擎增加 first-fit 模式；无约束也保留 run-scoped 缓存和相同 RCL|
|运行内缓存成功/失败，并按物体、orientation、箱布局、搜索设置区别状态|已有确定性缓存|随机 CA 的 key 增加 RNG 状态，防止缓存固定后续重构的随机抽样；LS 继续复用确定性结果|
|ILS 连续失败 100/125/150 触发 kick|controller 已支持|新 tuning 三组均传 `iteration_limit=None`，避免被额外总 iteration limit 提前截断|
|位置 RCL top/bottom 至多 16 个去重点|已有候选与位置 RCL|补上新空箱的随机候选路径：cube 的极端角点、cylinder 的 IFV top/bottom 极值；不再只评价规范起点|
|位置 RCL 后再用相同 tolerance 做 orientation RCL|原共享引擎最终仍取 orientation 的最小 PES|新增 `orientation_rcl.py`，两层按同一 tolerance 各自随机选择|
|GRASP：NI100 和 B100/B125/B150，共四组|函数已有 NI 与独立 budget；旧 iteration 脚本只覆盖三组|新 suite 明确区分四组；NI 改善清零，B 的总预算不清零|
|RCL 测试一次 CA + 50 次实际 LS|旧固定工作量脚本可做部分实验，完整流程没有统一衔接|新 worker 固定一次构造、50 LS、不 kick、不重构；若只需一个箱会提前停止并明确记录|
|GRASP_ILS 每轮 300 LS，连续失败 100/125/150 kick|已有；kick 不占 300|保留并回归验证；最后一次 LS 可 kick，因此持续失败最大 kick 数是 3/2/2|
|三算法整体 3600 秒，包括 CA|controller 已计 CA，旧 HPC runner 有额外完整 CA warm-up|新 worker 删除 warm-up；共享 wrapper 的初始化/复制也从 controller 可用预算扣除，报告完整算法入口耗时|
|五顺序 seed 分 dataset/shape 标定容量，排最空箱|CSV缺各箱物体数；后来找到1800个原始JSON|新增历史placements恢复与校验；可导入150条reference，计算平均U*/n*和三档容量，不必GPU重跑|
|内部随机，输入顺序不重洗|旧脚本已有部分实现|worker 用 time_ns/host/pid/task 派生独立种子，保存实际值；不 shuffle 输入|
|180 个调参任务 + 4050 个完整任务均能 HPC 执行|旧脚本不覆盖最新四组 GRASP 和容量网格|新增四阶段 suite、Slurm 数组、CPU 汇总依赖、冻结包、断点恢复、单命令提交|
|结果可核对且不能混协议|旧实验身份检查存在，但不覆盖新流程|逐任务 bundle/case 身份和源码/输入 SHA256；独立验证物体、边界、碰撞和 N/U/U*；容量另验|

## 论文表述需要明确的边界

1. 用户已明确确认 LS 使用 first-improvement：按顺序尝试替代 orientation，出现第一个严格改善即接受、重新选物体；若一直没有改善才尝试完全部替代 orientation。这与预期方法一致，不是代码差异。有限次随机 LS 没有改善不等于穷举邻域后证明局部最优。
2. “CA+50” 按**一次重新 packing 并评价的邻居**计数，不是选 50 次物体。一次物体选择可能评价多个 orientation。单箱时 LS 早停：N 已为 1，U*=U 在物体总体积和箱容量固定时不变，目标值不能进一步改善。GRASP 仍可继续下一次构造直至整体时间预算或 restart limit，RCL 测试则固定仅一次构造。
3. 相同 tolerance 用于两级 RCL，并不保证最终抽到的位置在所有 orientation-position 的全局最优 PES 的单层 tolerance 内。非负 PES 时 10% 两级最坏可能为 21%。
4. 选 tolerance 按 dataset/shape 分别做；文档只说 per dataset，这里将 bin shape 作为同一参数选择分组的一部分。CA PES/reference range 也按 dataset/shape 从历史 12 组结果选择。
5. 时间上限是软上限：完成已经开始的一次 packing，保存其结果，再停止。文件读取、CUDA context 初始检测、结果独立验证属于任务开销；算法入口内初始化/JIT/CA/LS/kick/重构属于 3600 秒。
6. 正式调参仍为 geometry-only，full 才启用容量和 radiation bin priority。tolerance 与 iteration 配置在 full 前固定，不在 full 数据上重新调优。
7. 调参 seed 为 3/7/13，full 为 3/7/13/19/53，部分顺序重叠；这是文档指定的当前方案，不是完全独立的 hold-out 评估。内部搜索使用不同实际种子。若论文需要独立测试顺序，可扩展新实例顺序再创建新协议包。

## 验证范围

已做 CPU 控制流、RCL、缓存、容量公式、冻结输入、依赖图、断点筛选和汇总测试，以及真实 15 数据集全部输入准备与 submit dry-run。本机没有完整 CUDA/numba/scipy 环境，未声称完成 AIRE GPU 的端到端运行；README 给出一个独立 smoke 包流程用于集群验证。旧 HPC 结果、CA 结果和历史冻结包保持原样。
