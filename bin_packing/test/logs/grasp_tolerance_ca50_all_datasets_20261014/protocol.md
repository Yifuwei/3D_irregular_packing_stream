# Fixed-work RCL tolerance protocol

15 datasets x 2 shapes x 3 order seeds x 3 tolerances = 270 fresh independent workers. No 180-second batch scores are reused.

Each run uses exactly one complete CA and 50 actual LS neighbor evaluations. LS starts from its own CA and continues from the currently accepted layout. Accepted and rejected candidates each count once. There are no restarts and no elapsed-time cutoff. Fifty consecutive rejected candidates coincide with the complete LS budget rather than triggering an earlier restart. A single-bin solution stops early with actual counts and an explicit reason.

Tolerance rates are 10%, 15%, 20%; order seeds 3/7/13; piece_seed 101; position_seed 202. Cube PES maximal_residual_box, cylinder PES minimum_aabb_edges_len; bottom_top candidates. The algorithm is the original improved_GRASP with its original dynamic piece exclusion and improved_repack, not the separate ranked weight-radio implementation.

Construction and final N-U*, improvement, LS checkpoints 10/20/50, runtime, actual call counts, RCL histograms, and geometry validation are recorded. Each dataset/shape gets a provisional best tolerance only after all nine cases succeed. Best means minimum mean final N-U* across matched seeds; tied rates are retained. CA-only best rates are reported separately. Missing and failed cases are never zero-filled.

The seven CPU protocol regression tests passed. They exercise positional and keyword launcher binding, a construction lasting longer than 180 seconds followed by all 50 LS evaluations, rejected candidates, accepting improvements without resetting total work, single-bin stops, call-count validation, and partial/failure reporting. Test output is in protocol_tests.log.

Equal work does not imply equal time. Three order seeds are parameter-selection evidence, not independent generalization validation. Geometry checks do not certify complete insertion trajectories. All runtime algorithm source hashes and inputs match the frozen original experiment; the new driver is separately hashed in manifest.json.
