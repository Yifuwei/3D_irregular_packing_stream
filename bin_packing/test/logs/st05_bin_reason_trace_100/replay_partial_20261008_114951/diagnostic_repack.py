def improved_repack(original_object_info_total, selected_info, nfv_pool, ifv_pool,
                    best_orientations_list_no_bin, old_data_pool,
                    local_best_iteration, orientation, container_size,
                    container_shape, rho, max_radio, packing_alg, _evaluation,
                    SCH_nesting_strategy, density, axis, _type,
                    _accessible_check, _encourage_dbl, _select_range,
                    flag_NFV_POOL, _TRACE):
    """Repack from p, searching only bins whose packing has changed.

    Preserve the prefix before p. For later pieces, skip unchanged earlier bins
    and restore their placement in unchanged original bins. Search later bins.
    A bin remains changed once an orientation, placement or membership changes.
    """
    # Load the accepted local-best solution as the reference for repacking.
    filtered_data = old_data_pool[local_best_iteration]

    # Find a placement using the existing empty-bin policy or SC_heuristic.
    def search(info, current_layout, topos_layout, each_bin):
        # Handle an empty bin directly instead of searching against fixed pieces.
        if not current_layout[each_bin]:
            # Check the normalized voxel geometry before placing it at the cuboid origin.
            if container_shape == "cube":
                occupied = np.argwhere(info["array"] == 1)
                if not len(occupied) or np.any(occupied.max(axis=0) >= container_size):
                    return False
                return 0, (0, 0, 0)
            # For a cylindrical bin, obtain the inner-fit volume and choose a starting position.
            ifv = ifv_pool.retrieve_ifv(info, container_size, container_shape)
            x, y, z = np.where(ifv == 1)
            if not len(x):
                return False
            min_x = np.min(x)
            return 0, (min_x, np.min(y[x == min_x]), 0)
        # Search for a feasible position among the pieces already packed in this bin.
        return SC_heuristic(
            nfv_pool, ifv_pool, info, current_layout, topos_layout, each_bin,
            container_size, SCH_nesting_strategy, density, axis,
            container_shape, _type, _accessible_check, _encourage_dbl,
            _select_range, flag_NFV_POOL, _TRACE)

    # Identify the selected piece p and its candidate orientation.
    new_selected_index, new_selected_orientation = selected_info
    best_current_layout = filtered_data["bin_real_layout"]
    pieces_order_matrix = filtered_data["pieces_order"]
    # Map each sequence index to its original bin and complete placement record.
    # Piece indices distinguish individual objects even when their types are identical.
    old_piece_info = {}
    for each_bin, indices in enumerate(pieces_order_matrix):
        for j, num_piece in enumerate(indices):
            old_piece_info[num_piece] = (each_bin, best_current_layout[each_bin][j])

    # Initialize the reconstructed solution while retaining the original bin indices.
    current_layout = [[] for _ in best_current_layout]
    topos_layout = [np.zeros(container_size) for _ in best_current_layout]
    radio_list = [[] for _ in best_current_layout]
    pieces_order = [[] for _ in best_current_layout]
    # False means the processed prefix in this bin still matches the old solution.
    # Removing unprocessed suffix pieces alone does not mark a bin as changed.
    bin_state_flag_list = [False for _ in best_current_layout]

    # Update the real layout, occupancy, radioactivity and piece-order records together.
    def place(num_piece, each_bin, info):
        # Copy the object dictionary so final renumbering does not modify the reference solution.
        current_layout[each_bin].append(dict(info))
        topos_layout[each_bin] += info["array"]
        radio_list[each_bin].append(info["radio"])
        pieces_order[each_bin].append(num_piece)

    # Restore the prefix before p with its original positions and orientations.
    # The selected piece p is excluded and will be repacked with the suffix.
    for num_piece in range(new_selected_index):
        each_bin, info = old_piece_info[num_piece]
        place(num_piece, each_bin, info)

    # Removing/changing p changes its old bin even if it moves elsewhere.
    # Mark the original bin of p as changed: its orientation or membership will change.
    probe.mark(bin_state_flag_list, old_piece_info[new_selected_index][0], "SELECTED_ORIGINAL_BIN", new_selected_index)
    # Compute container volume for the capacity check controlled by rho.
    volume = (np.prod(container_size) if container_shape == "cube" else
              np.pi * (container_size[0] / 2) ** 2 * container_size[2])

    # Process p and all subsequent pieces in their original sequence order.
    for num_piece in range(new_selected_index, len(original_object_info_total)):
        # Retrieve this piece's original bin and placement for comparison and possible reuse.
        original_bin_index, best_current_info = old_piece_info[num_piece]
        # Use the candidate orientation for p and the accepted orientations for later pieces.
        orientation = (new_selected_orientation if num_piece == new_selected_index else
                       best_orientations_list_no_bin[num_piece])
        selected_axis, selected_degree = orientation.split("_")
        selected_degree = int(selected_degree)
        # Build trial geometry from the original piece, without modifying its input record.
        try_info_tem = dict(original_object_info_total[num_piece])
        try_info_tem["array"] = rotate_voxel(try_info_tem["array"], selected_degree, selected_axis)
        try_info_tem["orientation"] = orientation
        # Normally only p changes orientation; keep restoration safe if a
        # caller supplies a different orientation for another piece as well.
        # Invalidate restoration if the requested orientation differs from the old placement.
        if orientation != best_current_info["orientation"]:
            # Mark the original bin as changed after an orientation change or a departure.
            probe.mark(bin_state_flag_list, original_bin_index, "ORIENTATION_CHANGE_OR_DEPARTURE", num_piece)

        # Visit bins from index 0; stop as soon as the piece is placed.
        # The extra iteration opens a new bin only after existing bins fail or are skipped.
        for each_bin in range(len(current_layout) + 1):
            is_new = each_bin == len(current_layout)
            # Allocate a new bin when needed; a failed search in this bin is handled below.
            if is_new:
                current_layout.append([])
                topos_layout.append(np.zeros(container_size))
                radio_list.append([])
                pieces_order.append([])
                # A new bin has no old placement to reuse and is marked as changed.
                bin_state_flag_list.append(False)
                probe.mark(bin_state_flag_list, each_bin, "NEW_BIN", num_piece)
                
            # Core skip rule: p tries all eligible bins; later pieces reuse unchanged bins.
            # Skip unchanged earlier bins, restore the original bin, and search later bins.
            probe.visit(num_piece, new_selected_index, original_bin_index, each_bin, bin_state_flag_list, current_layout)
            if num_piece != new_selected_index and not bin_state_flag_list[each_bin]:
                if each_bin < original_bin_index:
                    # Reuse an earlier first-fit rejection in an unchanged bin.
                    probe.event("SKIP_UNCHANGED_EARLIER")
                    continue
                if each_bin == original_bin_index:
                    # Check original membership before allowing a fresh search.
                    # The unchanged original bin can restore this piece directly.
                    probe.event("RESTORE_UNCHANGED_ORIGINAL")
                    place(num_piece, each_bin, best_current_info)
                    break
                # Later bins were not tested after the old first-fit placement.
                # Search them without temporarily changing their state flags.
                # Success marks arrival/departure below; failure preserves flags.

            diagnostic_searched = False
            if not bin_state_flag_list[each_bin] and each_bin > original_bin_index:
                probe.event("VISIT_UNCHANGED_LATER")
            result = False
            if (try_info_tem["volume"] <= volume * rho - probe.timed("capacity_scan", np.count_nonzero)(topos_layout[each_bin])
                    and try_info_tem["radio"] <= max_radio - sum(radio_list[each_bin])):
                # Perform geometric search only after the skip rule and capacity checks allow it.
                diagnostic_searched = True
                probe.event("SEARCH")
                result = search(try_info_tem, current_layout, topos_layout, each_bin)

            # A failed attempt does not alter any change flag; continue to the next bin.
            # If even a new empty bin cannot accommodate the piece, report failure.
            if not isinstance(result, tuple):
                probe.event("FAILED_ATTEMPT")
                probe.emit("PLACEMENT_FAILED", piece=int(num_piece), bin=int(each_bin)+1, reason="GEOMETRY_SEARCH_FAILED" if diagnostic_searched else "CAPACITY_OR_RADIO_REJECTED", flags=[bool(v) for v in bin_state_flag_list])
                # Allocate a new bin when needed; a failed search in this bin is handled below.
                if is_new:
                    raise ValueError(f"Piece {num_piece} cannot fit in an empty bin")
                # A failed attempt changes no bin. In particular, bins before
                # p's original bin stay False after trying p.
                continue

            # Convert the successful search result into a placed object and commit its records.
            packed_tem = dict(try_info_tem)
            packed_tem["translation"] = result[1]
            packed_tem["array"] = translate_voxel(try_info_tem["array"], result[1])
            packed_tem["bin_position"] = each_bin
            place(num_piece, each_bin, packed_tem)

            # Compare the target bin with this piece's original bin.
            if each_bin != original_bin_index:
                # Mark the destination bin as changed so subsequent pieces reconsider it.
                probe.mark(bin_state_flag_list, each_bin, "ARRIVAL_OR_POSITION_CHANGE", num_piece)
                # Mark the original bin as changed after an orientation change or a departure.
                probe.mark(bin_state_flag_list, original_bin_index, "ORIENTATION_CHANGE_OR_DEPARTURE", num_piece)

            # Even without a bin transfer, a different orientation or occupied position is a change.
            # Once True, a change flag remains True for the rest of this repacking call.
            elif (orientation != best_current_info["orientation"] or
                  not probe.timed("array_equal", np.array_equal)(packed_tem["array"], best_current_info["array"])):
                # Mark the destination bin as changed so subsequent pieces reconsider it.
                probe.mark(bin_state_flag_list, each_bin, "ARRIVAL_OR_POSITION_CHANGE", num_piece)
            break

    # Old bin indices are needed throughout the search. Compact only after
    # all pieces are placed so empty bins do not inflate the objective N.
    # Remove empty bins only after repacking, preserving reference indices during the search.
    used = [each_bin for each_bin, layout in enumerate(current_layout) if layout]
    current_layout = [current_layout[each_bin] for each_bin in used]
    topos_layout = [topos_layout[each_bin] for each_bin in used]
    radio_list = [radio_list[each_bin] for each_bin in used]
    pieces_order = [pieces_order[each_bin] for each_bin in used]
    # Renumber the remaining bins and update each placed object's bin_position.
    for each_bin, layout in enumerate(current_layout):
        for info in layout:
            info["bin_position"] = each_bin
    # Return the same four layout records expected by improved_ILS.
    return current_layout, topos_layout, radio_list, pieces_order
