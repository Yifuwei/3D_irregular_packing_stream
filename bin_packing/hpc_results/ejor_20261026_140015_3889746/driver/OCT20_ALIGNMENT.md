# 2026年10月20日方法与实现对应

依据桌面文档 `20261020 EJOR paper update summary.docx`，并以用户随后确认的计数、全局参数选择和不评价多样性为准。没有修改原 Word。

| 文档条目 | 当前实现 | 验证 |
|---|---|---|
| CA 四种 PES、三种 selection range；PES 同时评价方向与位置 | 保留 SCH 四种评价和既有 reference CA 参数；不增加 OES 调参 | 原 SCH 边界测试、真实 GPU 构造 |
| 辐射/重量硬约束后重新排序 bin | `radio_bin_priority.post_placement_score`：放入后 R/Rmax + 放入后 U/Umax，降序；无约束阶段 first fit | 两种 ranked engine 及三个控制器集成测试 |
| Umax 定义 | `min(1, rho, max_weight/(weight_factor*bin_volume))`；与实际体积/重量容量一致 | 归一化容量测试 |
| 成功/失败缓存 | 保留 run-scoped、精确 prefix/orientation/policy 缓存；RCL 缓存全部坐标，重复访问仍重新抽样 | 缓存状态隔离与重复抽样测试 |
| ILS 使用加权概率选择物体 | 保留 size/order/stage 选择器，首次严格改善即接受 | 控制器状态测试 |
| ILS kick 条件 | 默认 `auto` = ceil(n/2) 次完整物体搜索失败，改善后清零；截止时未搜完的物体不累计失败 | 5物体/9方向、43物体取整测试 |
| kick 强度 | small/medium/large = 25%/50%/75%，ceil(p*n) 个不同物体，每个更换方向标签 | kick 改变数量测试 |
| GRASP RCL | 当前物体、当前 bin，10方向各至多16个底/顶极值位置；合并至多160个配对后一次筛选抽样 | 全池筛选、Top-k 截断测试 |
| 四种 RCL 参数 | 10%、20%、Top10、Top30；Top-k 不足时使用实际候选数 | 参数规模与实际 GPU RCL |
| RCL 实验 | 只构造一次，LS=0，kick=0；仅以 N-U* 评价质量 | worker 计数检查与全流程测试 |
| GRASP LS | 从第一个到最后一个物体，逐个方向；首次改善从头搜索；完整无改善遍历才重构 | systematic GRASP 六项测试 |
| GRASP_ILS | 相同加权 ILS；每轮 9*n 次 LS repacking 调用，改善不重置；kick 不计入 LS 预算 | 有改善、kick、两次重构的预算测试 |
| GRASP_ILS 预算边界 | 预算耗尽立即重构，不在末尾额外 kick；不把截断物体当完整失败 | 末尾预算测试 |
| 整体时间 | 最终实验3600秒；ILS调参1800秒；构造、LS、kick及包装器开销共用计时；不预热CA | 时间边界测试与实际报告 |
| Weight calibration | 五个 sequence seeds 的 U*；0.5/0.75/1.0 U*，下限最大物体重量 | 历史 JSON 重算与 calibration 测试 |
| Radiation calibration | Strong=rmax，Medium=(rmax+RUB)/2，Weak=RUB；RUB公式保留 | calibration 测试 |
| 重合 radiation 等级 | n*<=5 时标注重合；保留九个名义 W/R 组合，不静默改变实验规模 | 参数报告 `coincident_radio_levels` |
| CA sensitivity | 使用已选 PES/range，增加九种 W/R 组合；输出相对 geometry reference 的 N-U* 变化 | 五阶段收集与数据传递测试 |
| 统一参数选择 | 所有匹配 dataset/shape/seed/W/R cell 的算术平均 N-U*；选择一个 RCL 和一个 kick 强度 | 全局均值选择及缺失候选拒绝测试 |
| Hybrid 参数继承 | 全局 RCL；ILS最佳强度；ceil(n/2)触发；9n预算，无独立调参 | selection.json 自动传递测试 |
| 内部随机性 | time_ns+PID+host+task 生成独立 piece/position/kick/global seeds；不 shuffle 已有 sequence | 报告保留内部 seeds 和 order_seed |
| 一次性提交 | calibration → sensitivity → rcl → tuning → full，GPU阵列后 CPU 完整性 gate，afterok 传递参数 | 提交 dry run、恢复及全流程测试 |

## 计数与时间的准确含义

`n` 是物体数；`N` 是 bin 数。一次 LS repacking 指改变某一物体方向并对后续 sequence 重装箱的一次完整邻域评估，不是单个物体/bin 的几何查询。缓存命中仍属于这次评估。kick 内部重装箱不计入 9n，但计入运行时间。

一个完整失败的物体搜索最多九次替代方向尝试；有改善则提前接受。GRASP_ILS 固定的是 repacking 总数，不是物体选择总数。旧 API 可显式指定预算/触发值供诊断使用，正式 v2 协议只使用上述规则。

时间限制是停止启动下一次完整 packing 调用的界限。已经开始的调用会完成并评价，因此墙钟可能超过预算一个调用的耗时；初始 CA 至少完成一次。Slurm walltime 是额外的作业上限。这里没有声称可在任意 CUDA 调用中途精确截断到3600.000秒。

## 与 Word 原文仍有差别的文字

这些差别采用用户最新确认，不需要继续运行旧实验：

- LS 总述应区别顺序扫描 GRASP 和概率选择 ILS/Hybrid。
- `4.5*n` 改为完整失败时至多 `9*ceil(n/2)`；一般物体搜索是 up to 9 次。
- Hybrid 外层明确写为 **9n次LS repacking**，而不是 n 次物体选择。
- 删除 best iteration limit 和 100/125/150、300/100等调参说法。
- RCL 与 kick strength 暂时都使用全局统一参数，只评价质量。
- RCL 总量应是3240；n*等于5时三个radiation等级也重合。

## 不自动推断的事项

CA sensitivity 只评价已选 CA 在约束下的表现，不会自动重选12种 CA 参数。旧版冻结 bundle 应继续用自己的 driver；v2 结果放入新 bundle，不能把旧参数实验报告当作新版结果续跑。
