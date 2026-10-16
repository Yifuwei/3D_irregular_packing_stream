# 一键 GPU smoke run

用途：在 AIRE 上用真实 GPU 跑通四阶段，并自动检查参数选择和传递。结果放入独立 `ejor_smoke_*` 包，不用于正式论文结论。

这是小实例流程测试。最大实例32GiB内存测试请运行 `bash bin_packing/hpc/paper_suite/run_memory_smoke.sh`，三算法各3600秒、串行；全部参数和监控命令见 [EXPERIMENT_PARAMETERS.md](EXPERIMENT_PARAMETERS.md)。

|阶段|小规模设置|任务数|
|---|---|---:|
|容量标定|shapesnew，seed=3，cube/cylinder|2|
|RCL|三个 tolerance；一次构造 + 3 次 LS|6|
|Iteration|保留 GRASP 四组、ILS 三组、hybrid 三组；整体 30 秒/任务|20|
|Full|三算法 × 三档 weight × 三档 radiation × 两种箱形；整体 30 秒/任务|54|

合计 82 个任务。默认同时一张 GPU，每任务 4 CPU、16G 主存，Slurm walltime 为 1 小时/任务。30 秒是整体算法预算；初始构造或已启动的 packing 可能超过软上限。单箱时允许 LS 提前停止。短预算不保证触发 100/125/150 次 kick，控制流已有独立 CPU 测试。

## 提交

将最新脚本同步到 AIRE，激活已有 packing 环境后，在 repo 根目录执行：

```bash
bash bin_packing/hpc/paper_suite/run_smoke.sh
```

如果当前就在 `bin_packing/hpc`，执行 `bash paper_suite/run_smoke.sh` 即可。脚本自动推导 repo 路径，不依赖当前工作目录。

显式指定输出位置（必须为新目录）：

```bash
EJOR_OUTPUT=/mnt/scratch/pmywe/GPU/ejor_flow_smoke_v1 \
bash bin_packing/hpc/paper_suite/run_smoke.sh
```

CA 汇总不在默认路径时增加 `CA_SUMMARY=/path/to/summary_by_dataset.csv`。可选 `SMOKE_TIME_LIMIT=60` 改为 60 秒，`MAX_PARALLEL=2` 改为两张并行 GPU。默认保持 30 秒、并行 1。不预估固定完成时间：还受排队、CUDA/JIT 和 packing 耗时影响。

仅查看提交命令（会准备冻结包，不会提交）：

```bash
bash bin_packing/hpc/paper_suite/run_smoke.sh --dry-run
```

dry-run 后如要运行同一个已准备的包，应使用输出路径，不要再次 run_smoke 创建新包：

```bash
python BUNDLE/driver/suite.py submit-all --bundle BUNDLE \
  --max-parallel 1 --walltime 01:00:00
```

## 自动流程与验收

```text
calibration → 容量 JSON
    → RCL → 最优 tolerance JSON
    → iteration → 最优 iteration JSON
    → full → CSV 汇总 → 自动 smoke 验收
```

阶段之间使用 Slurm afterok 依赖；每个阶段 GPU 任务结束后由 CPU 汇总检查完整性并选参数。后续 worker 在实际启动时读取选择文件，不需要在提交时预知最优值。

full 汇总成功后自动生成 `BUNDLE/smoke_verification.json`，预期：

```json
{
  "status": "PASS",
  "counts": {
    "calibration": {"expected": 2, "success": 2},
    "rcl": {"expected": 6, "success": 6},
    "tuning": {"expected": 20, "success": 20},
    "full": {"expected": 54, "success": 54}
  },
  "problems": []
}
```

报告还包含 bundle_id 和检查项目。检查包括：冻结代码/输入哈希；所有 task 身份和验证标记；阶段完整性；从原始 task 重新计算选定参数；检查每个后续任务记录的真实 search_settings 与选定参数一致；reference/RCL 工作量；无约束/有约束模式；full 各任务的容量；最终 CSV 存在。

查看报告：

```bash
python -m json.tool BUNDLE/smoke_verification.json
```

报告不存在表示最终验收尚未运行，不表示通过。手工检查（未完成时会输出 FAIL 并非零退出）：

```bash
python BUNDLE/driver/suite.py verify-smoke --bundle BUNDLE
```

某阶段失败会阻止下游；先查看 `logs/` 和 `results/<stage>/completion.json`，以及失败 task 的 error。取消同包剩余任务后，按 README 的 unlock 和 `submit-all --pending` 恢复，仍使用 `--max-parallel 1 --walltime 01:00:00`。

## 已验证范围

新增自动验收及四阶段参数交接通过本地 12 项 suite 测试，其中模拟完整四阶段能得到 PASS，错误 tolerance 和缺少 task 均得到 FAIL。这些本地测试使用模拟搜索结果；真实 GPU smoke 需要在 AIRE 执行上述提交命令。
