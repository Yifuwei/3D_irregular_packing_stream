# Cylinder边界计算优化与独立基准

本测试不替换function_lib.py中的原函数，也不改变正在运行的metaheuristic实验。新候选方法位于hpc/cylinder_ifv_candidate.py，benchmark通过AST提取当前真实的njit_get_ifv_cylinder与max_along_axis_2，避免导入整个GPU装箱环境。

## 方法

原方法对每个XY平移分配完整二维trans_projection，再与圆外mask相加并扫描所有元素。候选方法先提取物体投影凸包，再对每个XY平移检查凸包顶点到圆心的距离。圆盘是凸集，全部凸包顶点在圆内当且仅当整套投影点在圆内。

使用与原代码相同的体素中心坐标和圆周包含规则，整数加倍坐标处理半整数圆心，避免新增浮点误差。保留旧函数的XY中心约定（包括非方形网格）；不在本测试中修正它。新方法要求投影处于传入的、以原点归一化的bounding dimensions中，遇到可能导致旧实现截断的异常输入会报错。

输出仍为完整int32三维IFV。物体同类型/朝向、箱体同XY尺寸时，二维mask可复用；只有Z范围需要重新展开，特别适合严格accessibility反复生成扩展高度IFV。缓存未接入生产代码，基准的reuse_xy_mask是一张预先计算mask的命中场景，不计首次建mask成本；首次成本在hull_uncached中单独测量。

## 正确性和测量

先执行60个小规模全掩膜比较，包括空投影、单点、线、矩形、随机形状、奇偶直径、非方形网格、正常/扩展/不足高度。正式计时前每个真实物体/朝向/高度再次逐元素比较，任何不一致直接停止。

计时方法：

- original：原Numba方法。
- hull_uncached：新方法，包含投影、凸包、顶点判定及完整3D结果分配。
- reuse_xy_mask：已算好的二维mask复用，但仍包含完整3D结果分配。

两种方法均在计时前编译/预热，重复顺序交替，报告多次运行中位数和原始采样。使用同一int64输入、同一int32输出。不能用带JIT编译的首次耗时与已编译方法比较。

默认从最大实例st04_Example5_normal/cylinder/seed3的去重物体中，按体积分布选择最多3种；朝向x_0、z_90；每种分别测试正常高度和accessibility扩展高度，最多12个case，每个3次计时。不是整次CA耗时测试，因此不能将边界函数加速倍数当成整次CA的加速倍数。

## AIRE运行

上传三个Python文件/说明及Slurm脚本到bin_packing/hpc，不更新生产src目录。边界计算在CPU上执行，此基准不申请GPU。

```bash
cd /mnt/scratch/pmywe/GPU/3D_irregular_packing_stream/bin_packing

# 编译环境中先做正确性验证；只验证也可在登录节点小规模运行
python hpc/benchmark_cylinder_ifv.py --verify-only

# 在计算节点执行真实数据计时，不在登录节点运行完整基准
sbatch hpc/run_cylinder_ifv_benchmark.slurm

# 若有交互式CPU计算节点，可直接运行
python hpc/benchmark_cylinder_ifv.py --dataset st04_Example5_normal --seed 3 \
  --repeats 3 --max-pieces 3 --output hpc_results/ifv_test_1
```

Slurm默认1 CPU、8GB主机RAM、2小时，无GPU；使用packing环境。输出为hpc_results/cylinder_ifv_JOBID/timings.csv和samples.json。uncached_speedup=原方法中位耗时/新方法中位耗时；cached_speedup=原方法/复用mask耗时。大于1表示加速。

本地已通过60个小例子正确性检查。当前本地工具Python没有Numba，因此尚未产生公平的Numba计时结果；完整计时需在AIRE已有环境执行。
