def improved_ILS(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations,
                 orientations_list, packing_alg, selection_type, selection_range, accessible_check,
                 SCH_nesting_strategy, orien_evaluation, ALG, container_size, container_shape,
                 iteration_limit=None, time_limit=None, kick_trigger_time=None,
                 kick_level=None, flag_NFV_POOL=False, visualisation=False, _TRACE=False):
    
    """Run randomized descent and kicks with independent accepted/best states.

    iteration_limit counts evaluated neighbors and kick solutions, excluding
    construction. None disables a limit. Wall time includes construction.
    A running packing call is allowed to finish and its result is evaluated
    before stopping; no further candidate is started after the budget expires.
    """

    global TRACE
    TRACE = _TRACE

    if ALG not in ("ILS", "fixed_CA", "random_CA", "BLF"):
        raise ValueError("Unsupported ALG")
    if not object_info_total:
        raise ValueError("At least one piece is required")
    if iteration_limit is not None and (not isinstance(iteration_limit, (int, np.integer)) or iteration_limit < 0):
        raise ValueError("iteration_limit must be a nonnegative integer or None")
    if time_limit is not None and (not math.isfinite(time_limit) or time_limit < 0):
        raise ValueError("time_limit must be finite and nonnegative or None")
    if ALG == "ILS":
        if kick_trigger_time is not None and (not isinstance(kick_trigger_time, (int, np.integer)) or kick_trigger_time <= 0):
            raise ValueError("kick_trigger_time must be a positive integer or None")
        if kick_trigger_time is not None and kick_level not in ("small", "medium", "large"):
            raise ValueError("kick_level must be small, medium or large when kicks are enabled")
        if kick_trigger_time is not None and len(set(orientations_list)) < 2:
            raise ValueError("Kicks require at least two distinct orientations")
        if not orientations_list:
            raise ValueError("orientations_list must not be empty")

    overall_start = time.perf_counter()
    original_object_info_total = copy.deepcopy(object_info_total)
    data_pool = {}
    local_best_list, local_change_iter_list = [], []
    common = dict(density=5, axis='z', _accessible_check=accessible_check,
                  _encourage_dbl=True, _select_range=selection_range,
                  flag_NFV_POOL=flag_NFV_POOL, _TRACE=False)

    # Construction may modify its input; preserve unplaced original geometry.
    start = time.perf_counter()
    layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(
        copy.deepcopy(original_object_info_total), nfv_pool, ifv_pool,
        orientations, container_size, container_shape, rho, max_radio,
        packing_alg, orien_evaluation, SCH_nesting_strategy,
        _type=selection_type, random_CA=ALG == "random_CA",
        random_CA_threshold=5 if ALG == "random_CA" else None, **common)
    T = time.perf_counter() - start

    def evaluate(layout, topos_layout, pieces_order):
        # Reject invalid candidates before they can update accepted/best states.
        indices = [i for ids in pieces_order for i in ids]
        if sorted(indices) != list(range(len(object_info_total))):
            raise ValueError("Packing must contain every piece exactly once")
        if len(layout) != len(topos_layout) or len(layout) != len(pieces_order):
            raise ValueError("Inconsistent bin records")
        if any(len(objects) != len(ids) for objects, ids in zip(layout, pieces_order)):
            raise ValueError("Inconsistent piece records")
        if any(np.any(array > 1) for array in topos_layout):
            raise ValueError("Packing contains overlapping voxels")
        N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
        U_star = U if U_star is None else U_star
        return N, U, U_star, N - U_star

    def read_orientations(layout, pieces_order):
        values = [None] * len(object_info_total)
        for each_bin, indices in enumerate(pieces_order):
            for j, index in enumerate(indices):
                values[index] = layout[each_bin][j]["orientation"]
        return values

    N, U, U_star, local_best_perform = evaluate(layout, topos_layout, pieces_order)
    initial_orientations_list_no_bin = read_orientations(layout, pieces_order)
    local_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
    global_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
    global_best_perform = local_best_perform
    local_best_iteration = global_best_iteration = "origin"
    update_data(data_pool, layout, topos_layout, radio_layout, pieces_order,
                U_star, local_best_perform, T, "None", "origin")
    if visualisation:
        visualize_voxel_model(layout, pieces_order, container_size, container_shape)

    n_iter = 0
    no_change_time = iter_piece_selection = 0

    def budget_exhausted():
        return ((iteration_limit is not None and n_iter >= iteration_limit) or
                (time_limit is not None and time.perf_counter() - overall_start >= time_limit))

    # Exclude the last bin only while the accepted layout matches construction.
    exclude_last_bin = True

    def accept_candidate(layout, topos_layout, radio_layout, pieces_order, T,
                         selected_info, is_kick=False):
        
        # Store layout, actual orientations and objective as one consistent state.
        nonlocal local_best_iteration, global_best_iteration
        nonlocal local_best_perform, global_best_perform
        nonlocal local_best_orientations_list_no_bin, global_best_orientations_list_no_bin
        nonlocal no_change_time, iter_piece_selection, exclude_last_bin

        N, U, U_star, perform_this_iter = evaluate(layout, topos_layout, pieces_order)
        accepted = is_kick or perform_this_iter < local_best_perform
        globally_better = perform_this_iter < global_best_perform
        
        if accepted or globally_better:
            candidate_orientations = read_orientations(layout, pieces_order)
            # Rejected trials do not change eligibility. Once an accepted layout
            # changes, earlier CA failures no longer justify excluding the last bin.
            if accepted and exclude_last_bin:
                previous = data_pool[local_best_iteration]
                same_layout = (pieces_order == previous["pieces_order"]
                               and candidate_orientations == local_best_orientations_list_no_bin
                               and all(np.array_equal(a["array"], b["array"])
                                       for new_bin, old_bin in zip(layout, previous["bin_real_layout"])
                                       for a, b in zip(new_bin, old_bin)))
                exclude_last_bin = same_layout
            update_data(data_pool, layout, topos_layout, radio_layout, pieces_order,
                        U_star, perform_this_iter, T, selected_info, n_iter)
            if accepted:
                local_best_iteration = n_iter
                local_best_perform = perform_this_iter
                local_best_orientations_list_no_bin = list(candidate_orientations)
                local_best_list.append(perform_this_iter)
                local_change_iter_list.append(n_iter)
                no_change_time = iter_piece_selection = 0
            if globally_better:
                global_best_iteration = n_iter
                global_best_perform = perform_this_iter
                global_best_orientations_list_no_bin = list(candidate_orientations)
            prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
        else:
            no_change_time += 1
        trace(f"Evaluation {n_iter}: N-U*={perform_this_iter}, accepted={accepted}, kick={is_kick}")
        return accepted, N

    nonstop = ALG == "ILS"
    while nonstop and not budget_exhausted():
        # Select from the accepted local solution, never from a rejected trial.
        filtered_data = data_pool[local_best_iteration]
        num_used_bin = len(filtered_data["bin_real_layout"])
        selection_info = [None] * len(object_info_total)
        for each_bin, indices in enumerate(filtered_data["pieces_order"]):
            for j, index in enumerate(indices):
                selection_info[index] = filtered_data["bin_real_layout"][each_bin][j]
        LS_stage_var = ((kick_trigger_time - no_change_time) / kick_trigger_time
                        if kick_trigger_time is not None else 1.0)
        selected_index = improve_pieces_selection_ls(
            selection_info, iter_piece_selection, LS_stage_var,
            num_used_bin, seed=None, deterministic=False,
            exclude_last_bin=exclude_last_bin)
        iter_piece_selection += 1
        other_orientations = [value for value in orientations_list
                              if value != local_best_orientations_list_no_bin[selected_index]]
        if not other_orientations:
            break

        for each_orientation in other_orientations:
            if budget_exhausted():
                nonstop = False
                break
            selected_info = (selected_index, each_orientation)
            start = time.perf_counter()
            layout, topos_layout, radio_layout, pieces_order = improved_repack(
                original_object_info_total, selected_info, nfv_pool, ifv_pool,
                local_best_orientations_list_no_bin, data_pool, local_best_iteration,
                ils_orientations, container_size, container_shape, rho, max_radio,
                packing_alg, orien_evaluation, SCH_nesting_strategy,
                _type="bounding_box", **common)
            T = time.perf_counter() - start
            n_iter += 1
            accepted, N = accept_candidate(layout, topos_layout, radio_layout,
                                          pieces_order, T, selected_info)
            if budget_exhausted():
                nonstop = False
                break
            if accepted:
                break

            if kick_trigger_time is not None and no_change_time >= kick_trigger_time:
                # Retain the existing global-best-based perturbation policy.
                tem = list(global_best_orientations_list_no_bin)
                fraction = {"small": 0.25, "medium": 0.5, "large": 0.75}[kick_level]
                change_n = int(np.ceil(len(object_info_total) * fraction))
                change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)
                for each_position in change_position:
                    # Each selected piece must change its orientation label.
                    # Remove duplicates to sample distinct alternatives uniformly.
                    other_orientations_kick = [value for value in dict.fromkeys(orientations_list)
                                               if value != tem[each_position]]
                    tem[each_position] = np.random.choice(other_orientations_kick)
                start = time.perf_counter()
                layout, topos_layout, radio_layout, pieces_order = kick_repacking(
                    copy.deepcopy(original_object_info_total), nfv_pool, ifv_pool,
                    orientations, tem, container_size, container_shape, rho, max_radio,
                    packing_alg, orien_evaluation, SCH_nesting_strategy,
                    _type=selection_type, **common)
                T = time.perf_counter() - start
                n_iter += 1
                _, N = accept_candidate(layout, topos_layout, radio_layout,
                                        pieces_order, T, "kick", is_kick=True)
                if budget_exhausted():
                    nonstop = False
                # Rebuild piece selection and orientation candidates after every kick.
                break

    best_data = data_pool[global_best_iteration]
    original_data = data_pool["origin"]
    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]
    best_topos_layout = best_data["bin_topos_layout"]
    best_pieces_order = best_data["pieces_order"]
    
    best_N, best_U, best_U_star, _ = evaluate(best_current_layout, best_topos_layout, best_pieces_order)
    origin_N, origin_U, origin_U_star, _ = evaluate(
        origin_current_layout, original_data["bin_topos_layout"], original_data["pieces_order"])
    if visualisation:
        visualize_voxel_model(best_current_layout, best_pieces_order, container_size, container_shape)
    overall_time_cost = time.perf_counter() - overall_start

    return (best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,
            best_current_layout, origin_current_layout, best_topos_layout,
            best_pieces_order, list(initial_orientations_list_no_bin),
            list(global_best_orientations_list_no_bin), local_best_list,
            local_change_iter_list, overall_time_cost)
    
    # df = pd.DataFrame(data_pool)
