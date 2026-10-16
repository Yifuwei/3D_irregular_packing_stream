# Smoke 与内存实验提交前核对

2026-10-16 重新读取用户最新版 `EJOR paper update summary.docx`，包括原生数学公式，逐项核对实际本地 main repo。Word 未修改。实验完整参数见 `EXPERIMENT_PARAMETERS.md` 和 JSON；本表说明核对结果及测试缩减。

|Word条目|实际实现与正式实验|结论|
|---|---|---|
|1 CA：位置/orientation同一PES|共享packing引擎以同一PES评分；不独立使用OES|符合|
|2 LS：大小、顺序概率加权，连续失败决定阶段|improve_pieces_selection_ls；三个improved controller均调用；找到第一个严格改善即接受|符合用户已明确确认的first-improvement|
|2 辐射/重量硬过滤与排箱|full先检查硬容量，再按放入后的filling×radiation-left比例降序|符合|
|2 无约束用first-fit|calibration、RCL、tuning和memory均设置radio_priority=False、cache_first_fit=True|符合|
|2 全运行成功/失败缓存|按物体、orientation、箱布局、搜索设置区分；缓存跨LS和重构；随机构造另含RNG状态|符合|
|3 ILS|一次CA开始，kick阈值100/125/150，medium；总iteration_limit=None|符合|
|4 位置RCL|top/bottom极值点，去重至多16个；按best+tau×abs(best)均匀抽取|符合|
|4 orientation RCL|在各orientation抽出的位置之间，再按同一tau过滤并随机抽取|符合|
|4 tolerance实验|正式.10/.15/.20，一次CA+50实际LS，restart_limit=1，无kick|符合|
|4 GRASP四组|NI100改善清零；B100/B125/B150总LS预算改善不清零|符合|
|5 GRASP_ILS|每轮300实际LS，连续失败100/125/150触发kick；kick不占300|符合|
|6 标定|五seed历史reference，排最空箱计算平均U*/n*；weight=.5/.75/1 U*加单体下界；BR公式及1/1.5/2倍|符合；已支持原始JSON恢复，无需GPU重跑|
|实验规模|RCL270；ILS54、GRASP72、hybrid54，共180调参；full4050|符合|
|整体时间|正式三算法每次3600秒，包括算法内初始化、CA、LS、kick、重构，无丢弃CA预热|符合软预算约定；允许在途packing结束|

口径：一次LS iteration指实际repack并评价一个orientation邻居，不是选一次物体。有限次随机LS无改善不构成邻域穷举证明。tolerance按dataset/shape分别选取；两个RCL层各有同一tau，而非对所有orientation-position全局统一做一次过滤。

## 两套提交入口与有意缩减

|项|小实例smoke|大实例内存测试|
|---|---|---|
|入口|run_smoke.sh|run_memory_smoke.sh|
|dataset|shapesnew|st04_Example5_normal|
|物体数|43|50|
|箱形/网格|cube16³、cylinder19³|cylinder180³|
|顺序seed|3|3|
|calibration|每形1次新CA，验证完整流程；仅smoke专用容量|不提交|
|RCL|tau .10/.15/.20；CA+3 LS，替代正式50|不提交；固定tau=.15作资源测试|
|iteration|保留十组配置，整体30秒/任务|只取GRASP NI100、ILS K100、hybrid300/100，整体3600秒/任务|
|full|三算法×九容量组合×两箱形，整体30秒/任务|不提交；geometry-only|
|数量|2+6+20+54=82个GPU任务|3个GPU任务|
|Slurm|1GPU、4CPU、16G；每包并行1；walltime1h/任务|1GPU、4CPU、32G硬限；array0-2%1；walltime6h/任务|
|验收|最终smoke_verification.json应PASS|三任务正常完成，sacct检查状态/MaxRSS，检查每次算法时间和实际LS/kick次数|

小smoke结果不能作为正式调参结论；大内存测试不能证明所有带容量约束full配置的峰值上界。主存OOM与CUDA显存OOM需区分。两个脚本同时提交时可能同时运行两张GPU；顺序执行可以保持只运行一张。

## 本次实际测试

* 本地main：76项算法、LS控制流、RCL、缓存、边界和RNG回归通过。
* 当前流程：20项准备/阶段交接/历史恢复/失败阻断/内存参数/radiation精度测试通过。
* 重新建立小smoke和大memory两个冻结包，所有源码与输入SHA256校验通过。
* 两个新包的function_lib.py均与main逐字节一致，包含实际生产版凸包IFV、uint8和XY缓存；不是旧worktree的IFV。
* 小包submit dry-run得到4个GPU数组和4个CPU汇总依赖；大包恰好三个算法、seed3、cylinder、3600秒。
* main中四个实际执行Shell/Slurm文件逐个通过bash -n。
* 前一轮已实际运行生产IFV的5项Numba等价性测试及11项边界回归，全部通过。

本次修正：旧code_launcher将radiation舍入到6位，但历史标定使用原始精度。新worker加载后恢复原始实例radiation，并检查物体数量、类型和体积顺序；新增回归通过。避免BR恰为最大单体radiation时出现舍入超限。没有改原始instances、历史JSON或几何算法。

以上是本地测试和提交准备验证，尚未将任务提交到AIRE，不能替代集群真实GPU smoke。需上传最新代码后重新创建冻结包，旧冻结包不会自动获得IFV或worker修正。

## 提交命令

在AIRE repo根目录，激活packing环境并确保最新源码及 `bin_packing/hpc_results/CA param test result/summary_by_dataset.csv` 已上传：

```bash
module load miniforge
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate packing

# 先验证小实例全流程
bash bin_packing/hpc/paper_suite/run_smoke.sh

# 小实例通过后，启动最大实例32G内存测试
bash bin_packing/hpc/paper_suite/run_memory_smoke.sh
```

如果当前目录为 `bin_packing/hpc`，命令分别是 `bash paper_suite/run_smoke.sh`、`bash paper_suite/run_memory_smoke.sh`。两脚本会创建全新目录并打印路径。不要同时使用同一个EJOR_OUTPUT环境变量指向一个目录。
