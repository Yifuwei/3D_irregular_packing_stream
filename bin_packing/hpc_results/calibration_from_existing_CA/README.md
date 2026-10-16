# 从既有 CA 结果恢复的容量标定

2026-10-16 检查 `bin_packing/hpc_results/CA_jsons_results/all_instances_geometry/`，共索引 1800 个成功且 validation.ok=true 的 geometry-only fixed_CA JSON，PES 与 OES 一致。

按正式 suite 当前规则，为每 dataset/shape 从历史 CA 汇总的 12 组中选平均 N−U* 最低、再按运行时间最短的配置。取五个顺序 seed 3/7/13/19/53，共恢复并验证 150 个 reference，生成 30 个 dataset/shape 的三档 weight/radiation capacity。

`references.json`：每个参考运行的来源、SHA256、历史配置/commit、各箱体积、各箱物体数、最空箱、U*/n* 和单体最大体积/辐射。

`selection.json`：30 组最终容量及比例，使用 weight_factor=1。

恢复过程检查箱形/尺寸、物体类型及数量、同类型体积一致性，并由 placements 和当前实例体积复算 N/U/U*，与原 JSON 核对。几何无碰撞/边界验证沿用历史 validation，不重新运行 GPU 几何搜索；当前无法仅凭旧 JSON 证明历史与当前 binvox 文件字节完全相同。

无需补跑150个CA即可取得所需统计。当前suite已支持 `prepare --ca-json-dir PATH` 导入并冻结原始reference，submit会跳过已导入的calibration GPU任务；run_all.sh会自动检测默认CA_jsons_results目录。仅放置本文件不足以导入，需让prepare读取原始JSON。
