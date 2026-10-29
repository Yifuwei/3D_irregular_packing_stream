> Smoke calibration update: run_smoke.sh now reuses hpc_results/CA_jsons_results automatically (or CA_JSON_DIR). If missing, it stops before submission; it no longer starts GPU calibration by default.

> Current submission default (2026-10-26): stop after calibration and CA sensitivity. RCL/tuning/full are not submitted. Use --through full only when intentionally resuming the later experiments. run_smoke.sh also stops at sensitivity by default; inspect results/sensitivity/completion.json. Full-workflow smoke verification requires an explicitly completed full workflow.

# 新版实验参数与运行顺序

协议：`ejor-three-seed-repacking-v3`。依据10月29日PDF及本次对话的最新确认。完整参数和每一个任务见同目录 `EXPERIMENT_PARAMETERS.json`；每次 prepare 后还会生成该 bundle 专属的 `parameter_plan.json` 和 `manifest.json`。

## 共用设置

- 15 datasets，cube/cylinder；容量校准保留原五个 seeds 3、7、13、19、53；所有后续实验只用3、7、13。已有CA_sensitivity_check只筛选这三个实验seeds的810条结果，不重跑；smoke实验使用seed3。
- 选定 CA 的 PES/range 按 dataset/shape 从已完成的12组 geometry-only CA 中读取。
- 输入顺序不再 shuffle。内部 piece/position/kick/global RNG 独立于 sequence seed，并记录实际 seeds。
- 强/中/弱 weight = max(0.5/0.75/1.0 × U*，最大物体体积比) × bin volume，weight factor=1。
- RUB = rmax + 1000 max(0,0.2n*−1)；radiation 强/中/弱 = rmax、(rmax+RUB)/2、RUB。
- U* 是删除体积最少 bin 后的平均 filling rate；单箱采用 U*=U。n* 也删除同一箱后计算。
- n*≤5时保留九个名义组合，但标注 radiation capacities 重合。全零 radiation 使用1e-12作为实现容量。
- 有约束时先过滤不合法 bin，再按放入后 R/Rmax + U/Umax 降序；Umax=min(1,rho,Wmax/(weight factor×Vbin))。无约束使用 first fit。
- 成功/失败缓存仅在单次算法运行内保留；CA候选池缓存坐标，抽样每次重新执行。
- 全局参数选择以所有匹配 cell 的平均 N−U* 最小为准。平局顺序：RCL10、RCL20、TOP10、TOP30；kick 25%、50%、75%。不使用rank，不衡量多样性。

## 各阶段

| 阶段 | 参数 | 正式任务数 | 算法预算 |
|---|---|---:|---|
| calibration | 每dataset/shape五个reference sequence；优先恢复已有CA JSON | 150个reference；恢复后无需GPU | 固定CA一次 |
| sensitivity | 已选CA；15×2×3×9约束组合 | 810 | 固定CA一次，无LS |
| rcl | 15×2×3×4规则×9约束组合 | 3240 | 随机构造一次，LS=0 |
| tuning | shapesnew、Merged4_normal、st04_Example5_normal；2形状×3seeds×3强度×9约束 | 486 | 每次1800秒整体时间 |
| full | 15×2×3×9约束×3算法 | 2430 | 每次3600秒整体时间 |

RCL规则为10%、20%、Top10、Top30。每个物体在当前bin的10个方向各评价至多16个底/顶极值位置，合并池后一次筛选和均匀抽样。位置数不足时Top-k取实际数量。

ILS：按size与动态order评分，选择本轮未访问物体中的最高分；每次最多9个替代方向，first improvement即接受；连续ceil(n/2)次LS repacking无改善触发kick，改善或kick后清零。kick强度ceil(p*n)，p=25%、50%、75%。

GRASP：顺序完整邻域搜索，first improvement后从第一个物体重启；完整一遍无改善才重构，无100/125/150调参。

GRASP_ILS：继承统一RCL和ILS强度，触发ceil(n/2)；每轮n次LS repacking，改善不清零；kick不占n次LS预算，但占整体时间。预算耗尽直接重新构造，无独立iteration实验。

时间上限包括构造、LS、kick，不执行丢弃的CA预热。已启动的packing调用允许完成，所以可能超出软预算一次调用的时间。

## HPC 命令

在AIRE进入 `bin_packing/hpc/paper_suite`，先加载既有packing环境。

```bash
# 小实例全流程：两种形状，seed3，算法整体时间30秒
bash run_smoke.sh

# 大实例内存：st04_Example5_normal / cylinder / seed3
# 三个算法各3600秒；每任务1GPU、4CPU、32G；并发1
bash run_memory_smoke.sh

# 正式默认仅到sensitivity；自动从已有CA JSON恢复calibration
bash run_all.sh

# 只查看提交图，不提交
bash run_all.sh --dry-run
```

默认正式并发6；shapesnew/chess任务16G，其余32G；每任务1GPU、4CPU。普通任务Slurm walltime默认12小时；内存实验6小时。CPU gate为1CPU、2G、30分钟。

小smoke默认任务数：calibration2、sensitivity18、rcl72、tuning54、full54，共200；如果指定CA_JSON_DIR恢复历史reference，calibration会是10个已恢复reference（保留原五-seed容量）。仅cube时各阶段为1、9、36、27、27，共100。

```bash
# BUNDLE 替换为提交输出的实际路径
python "$BUNDLE/driver/suite.py" verify-smoke --bundle "$BUNDLE"
python "$BUNDLE/driver/memory_probe.py" check --bundle "$BUNDLE"

# 恢复：只提交缺失/失败任务，保留已完成结果
bash submit_all.sh "$BUNDLE" --pending
```

每阶段GPU结束后CPU gate核对完整结果，写入runs.csv、summary.csv（sensitivity/full）和selection.json。下一阶段只有gate成功才启动；full读取统一RCL、统一kick强度及calibration容量。不需要提前手动知道最佳参数。

旧版冻结bundle继续用其自身driver；新版必须prepare新目录。不要把旧100/125/150或CA+50结果混入本协议。

2026-10-29：正式全阶段任务数为 calibration 150、sensitivity 810、RCL 3240、ILS tuning 486、full 2430，共7116；完整实验计算预算为2430小时，ILS调参为243小时。新协议需prepare新bundle，旧冻结包不自动改变。LS iteration统一为一次候选方向的完整repacking评估，N表示物体数；不再将N/2乘以9。

## 从已有 sensitivity 结果继续（当前推荐）

保留 CA_sensitivity_check 的五-seed容量；复用其中3、7、13的810条sensitivity结果。原始结果目录不变。run_all.sh默认检测该目录并用最新算法代码冻结一个新续跑包，核对输入、CA参数、容量和结果后导入，不提交calibration/sensitivity GPU任务。

```bash
# 在paper_suite目录；后面只运行RCL、ILS调参和full，实验seeds均为3、7、13
bash run_all.sh --through full

# 原结果在其他路径时
SENSITIVITY_BUNDLE=/path/to/CA_sensitivity_check bash run_all.sh --through full
```

不加 --through full 的默认终点仍为sensitivity；在复用场景只会核对已完成阶段。不要直接用旧结果目录的冻结driver运行新算法实验。
