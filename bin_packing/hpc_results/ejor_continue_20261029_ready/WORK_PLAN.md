# 当前 Work Plan — 2026-10-29

本计划以本次对话最终确认和已验证的续跑包为准。N表示物体数；一次LS iteration指一次候选方向的完整repacking评估，不是一次piece selection，也不是一次piece/bin几何查询。

## 当前状态

- CA参数测试已完成：四种PES × 三种selection range；按dataset/shape使用已选定参数。保留历史数据，不重跑。
- 容量校准已完成：保留sequence seeds 3、7、13、19、53的150份参考结果，固定各dataset/shape的weight/radiation容量。后续不重新校准。
- CA sensitivity已完成：原结果1350条全部成功，其中3、7、13的810条已导入续跑包，不重跑GPU。
- 正式续跑包：`bin_packing/hpc_results/ejor_continue_20261029_ready`。只使用这个包，忽略第一次准备时留下的未完成目录。
- 本地已完成24项HPC流程测试、真实结果导入、冻结文件哈希检查和全流程提交dry run。没有由本次本地操作提交HPC作业；未查询HPC实时队列。

## 后续实验顺序

| 顺序 | 阶段 | 实验规模 | 任务数 | 单次算法预算 | 产出/进入下一阶段条件 |
|---|---|---|---:|---|---|
| 已完成 | Calibration | 15 datasets × 2 shapes × 5 reference seeds | 150 | 复用已有结果 | 原固定容量，已收集验证 |
| 已完成 | CA sensitivity | 15 × 2 × 3 seeds × 9 W/R组合 | 810 | 复用已有结果 | 已筛选3、7、13并收集验证 |
| 1 | RCL比较 | 15 × 2 × 3 seeds × 4规则 × 9 W/R组合 | 3240 | 每次只运行一次随机CA；无LS、无kick | 从10%、20%、Top10、Top30中选全局最佳RCL |
| 2 | ILS kick强度调参 | 3 selected datasets × 2 × 3 seeds × 3强度 × 9 W/R组合 | 486 | 整体1800秒 | 从25%、50%、75%中选全局最佳kick强度 |
| 3 | Full comparison | 15 × 2 × 3 seeds × 9 W/R组合 × 3算法 | 2430 | 整体3600秒 | 三算法最终结果、均值/标准差与对比 |

待运行共6156个任务；7116为含已复用150+810条记录的整个计划条目数。RCL/tuning/full的sequence seeds均只有3、7、13。

调参数据集：shapesnew（小）、Merged4_normal（中）、st04_Example5_normal（大）。两种bin shape为cube/cylinder。W/R分别有strong/medium/weak，共9个组合。容量重合时仍保留名义组合并标注。

参数选择使用匹配cells的原始N−U*算术平均值，越小越好。RCL采用一个全局最佳规则；ILS选一个全局最佳kick强度；GRASP×ILS继承两者，不另做独立迭代预算调参。每阶段完整性检查通过后才运行下一阶段。

## 当前算法规则

| 算法 | Piece selection | LS停止/kick/重构规则 |
|---|---|---|
| ILS | size + 动态order评分；从本轮未访问物体中选最高分，早期偏向靠前、停滞后偏向靠后 | 连续ceil(N/2)次LS repacking无改善触发kick；改善或kick重置停滞计数与访问集合；总搜索继续到时间预算 |
| GRASP | 固定输入顺序，不使用动态评分 | 每个物体尝试替代方向，first improvement后从第一个物体重启；完整一轮无改善达到该邻域local optimum后重新构造 |
| GRASP×ILS | 与ILS相同的动态评分和轮内无重复选择 | 每次构造最多N次LS repacking，改善不重置此预算；连续ceil(N/2)次无改善触发kick；N次预算耗尽直接重构，末尾不额外kick |

每个物体最多尝试9个替代方向，找到改善提前接受。repacking的预算或kick阈值可在一个物体的方向未全部尝试完时到达；不再把N/2或N乘以9。Kick不占hybrid的N次LS预算，但消耗整体时间。单箱达到目标下界可以提前结束LS。截止或显式预算截断不宣称完成局部最优搜索。

构造、LS、kick及包装器开销共用算法时间预算。已开始的packing调用允许完成，之后不再启动新调用，故可能超出软预算一次调用时间。Slurm作业walltime默认12小时，是额外的作业上限。

## HPC资源

| 任务 | 每任务主机RAM | CPU | GPU |
|---|---:|---:|---:|
| chess、shapesnew | 16G | 4核 | 1张 |
| 其余datasets | 32G | 4核 | 1张 |
| 阶段完整性检查/收集 | 2G | 1核 | 无 |

默认最多同时运行6个GPU任务。主机RAM不等于GPU显存。16G和32G任务分组提交，阶段间通过Slurm依赖自动串接。

## HPC执行命令

将整个`ejor_continue_20261029_ready`目录上传至HPC仓库的`bin_packing/hpc_results/`下。进入HPC仓库根目录，激活现有packing环境后执行：

```bash
BUNDLE=bin_packing/hpc_results/ejor_continue_20261029_ready

# 查看提交图，不占GPU、不提交作业
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --dry-run

# 正式提交；只会为RCL、tuning、full创建新的GPU任务
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full

# 中断后恢复：先确认没有同一包的活动作业，再只补缺失/失败任务
bash "$BUNDLE/driver/submit_all.sh" "$BUNDLE" --through full --pending
```

必须显式写`--through full`；不写时默认终点是sensitivity，不会启动后续实验。已有calibration/sensitivity仍运行CPU完整性gate，不重复运行GPU。

可选资源覆盖：在提交命令后加`--max-parallel 6 --small-memory 32 --large-memory 32`可把全部算法任务主机RAM设为32G。当前计划仍保持16G/32G默认值，未自动提高。

结果保存于`$BUNDLE/results/rcl`、`$BUNDLE/results/tuning`和`$BUNDLE/results/full`，运行日志在`$BUNDLE/logs`。每阶段保存completion.json、runs.csv和selection.json；full另有summary.csv。不要手工改动已冻结的driver、runtime或inputs文件，以免哈希校验失败。
