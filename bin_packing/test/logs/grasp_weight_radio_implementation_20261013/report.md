# improved_GRASP_weight_radio implementation verification

11 CPU regression tests passed in the main repository. The production empty-bin geometry adapter also passed cube and cylinder PES smoke checks without launching a GPU task.

Tests cover bin priority, radio and weight capacities, oversized pieces, repeated failures, successful restoration, state/orientation/search-policy invalidation, capacity checks before cache reuse, cache lifetime across constructions and LS evaluations, isolated cached records, bounds checks, and the eleven-value GRASP controller API.

The cache belongs to one algorithm run and persists across restarts and LS. Its exact key contains piece ID, orientation, bin ID, ordered bin contents (IDs/orientations/translations), and geometry/search policy. Successful entries store PES and translation rather than voxel arrays. CA and LS have distinct policy keys. Cache hits reuse the first successful construction RCL draw; identical construction queries are not independently resampled. This can reduce restart diversification.

All pieces are eligible for LS because ranked-bin construction invalidates the earlier first-fit last-bin exclusion assumption. Weight is piece volume times weight_factor. max_weight is an absolute bin weight capacity.

Full GPU NFV/AFR searches and end-to-end GPU performance were not tested: the all-dataset tolerance batch is currently using the GPU. Its seven source hashes remain unchanged. No existing launcher or experiment algorithm was replaced.

Files: bin_packing/src/improved_GRASP_weight_radio.py; bin_packing/test/test_grasp_weight_radio.py. Import the function from the new module.
