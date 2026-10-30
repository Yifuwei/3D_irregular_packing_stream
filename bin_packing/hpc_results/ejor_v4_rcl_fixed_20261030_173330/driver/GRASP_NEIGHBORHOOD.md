# GRASP 顺序邻域搜索

2026-10-18 根据用户确认更新 `improved_GRASP` 的默认 LS。这里是算法变更说明，不修改已提交的冻结实验包。

## 邻域定义

固定输入 sequence、当前所有物体的 orientation 和当前被接受布局。对每个原序号 i=0..n-1，逐个尝试所有允许且不同于当前标签的 orientation。每个邻居保留 i 之前的已装布局，改变 i 的 orientation，并按原 sequence 重新装 i 及后面的物体。比较完整结果的 N−U*。

物体按原输入顺序扫描，包含最后一箱的物体。orientations_list 去重后保持给定顺序；当前十个标签通常每个物体最多九个替代标签。这里不包含重排 sequence 或枚举全部可能位置；repacking 使用确定的既有 SCH/PES 规则。

## 默认策略 ls_policy='systematic'

1. 从第一个物体开始，逐个尝试替代 orientation。
2. 找到第一个严格改善就立即接受，不再继续当前物体剩余的 orientation。
3. 针对新解，从第一个物体重新开始检查。
4. 最近一次改善后完整扫描所有物体、所有替代 orientation，均无改善，才确认该邻域局部最优；GRASP 随后重新随机构造。

time_limit 仍是整体软预算。显式 ls_evaluation_limit 也可截断；这两种截断均不宣称已确认局部最优。单箱达到该目标的下界时可以提前停止LS。没有替代orientation的空邻域按已完整检查处理。

原 iter_limit_per_iter/no_improvement_limit 参数为兼容旧接口保留并校验，但在systematic模式下不作为停止阈值；alpha/piece_seed同样不决定扫描顺序。顺序扫描替代加权随机选物体，仅修改GRASP默认LS，不改变ILS/hybrid接受策略。

## 结束原因记录

可传 ls_observer 接收每轮LS结束事件：restart、ls_policy、stop_reason、local_optimum、ls_evaluations、objective。

|stop_reason|是否确认局部最优|
|---|---|
|local_optimum|是，完整邻域无改善|
|single_bin_bound|是，单箱达到目标下界；不声称穷举了邻域|
|time_limit|否|
|evaluation_limit|否|
|legacy_stall_or_empty_neighborhood|否，旧策略|

新worker将这些事件写入结果JSON的ls_end_events，以及events.jsonl的ls_end行。

## 当前实验协议

10月20日 v2正式协议使用systematic；RCL实验仅构造一次，ls_evaluation_limit=0。
旧legacy_stochastic只保留为历史API兼容，已不用于新版正式实验。
参见OCT20_ALIGNMENT.md和EXPERIMENT_PARAMETERS_V2.md。
