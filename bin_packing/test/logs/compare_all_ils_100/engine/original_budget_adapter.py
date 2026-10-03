def ILS_from_the_first_piece(object_info_total, nfv_pool, ifv_pool, max_radio, rho, orientations, ils_orientations, orientations_list,
                            packing_alg, selection_type, selection_range, accessible_check,
                            SCH_nesting_strategy, orien_evaluation, ALG,
                            container_size, container_shape,
                            iteration_limit=None, time_limit=None, alpha=None, kick_trigger_time=None, kick_level=None, flag_NFV_POOL=False,  visualisation=False, _TRACE=False):
    
    global TRACE 
    TRACE = _TRACE

    # structure of object_info_total
    # =======================================================================================
    # key --      value                     --    value example
    # =======================================================================================
    # "array"     -- current 3D binary array        --    np.array((0 0 0),(1,1,1)...) (np array)     
    # "translation"     -- translation (for retrieve nfv) --    (11,13,67) (tuple)
    # "orientation"     -- orientation (for retrieve nfv) --    "x_180" (string)
    # "bin_position"     -- bin position                   --    5 (6 th bin) (int)
    # "volume"     -- volume after filling holes     --    800 (int)
    # "radio"     -- radioactivity                  --    1100 (float)
    # "piece_type"     -- piece type (for retrieve nfv)  --    777 (a number represents a group of item) (int)
    # =======================================================================================
    original_object_info_total = copy.deepcopy(object_info_total)

    data_pool = {}

    # trace("============================================================")
    trace(f"Algorithm is {ALG}")
    trace(f"Objects number: {len(object_info_total)}")
    trace(f"The container type: {container_shape}")
    trace(f"Container size is: {container_size}")
    trace(f"Packing algorithm is {packing_alg}") 
    trace(f"Nesting Strategy is: {SCH_nesting_strategy}")
    trace(f"Evaluation for orientation is {orien_evaluation}")
    trace(f"Accessibility check is {accessible_check}")
    trace("============================================================")

    # ============================================================================
    # Constructive algortithm
    trace("Solution loading..")
    # original_object_info = copy.deepcopy(object_info_total)

    if ALG == "fixed_CA" or ALG == "ILS" or ALG == "BLF":
        random_CA = False
        random_CA_threshold = None

    elif ALG == "random_CA":
        random_CA = True
        random_CA_seed = 42 # to decide the seed of the packing positon seletion
        random_CA_threshold = 5 # this means best five packing position is selected randomly.
    
    overall_time_cost = 0 

    check_1 = time.time()
    # orientations_start = ([0,90],["x"])
    # 

    layout, topos_layout, radio_layout, pieces_order = packing_3D_voxel_lookback(original_object_info_total, nfv_pool, ifv_pool, orientations, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range, random_CA= random_CA, random_CA_threshold = random_CA_threshold,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False) # for SCH
    
                                                                                                                                            # def packing_3D_voxel_lookback(polobject_info_total, nfv_pool, orientation, container_size, 
                                                                                                                                            #                         container_shape, rho, max_radio, 
                                                                                                                                            #                         packing_alg, _evaluation,
                                                                                                                                            #                         SCH_nesting_strategy, density, axis, 
                                                                                                                                            #                         _type, _accessible_check, _encourage_dbl,
                                                                                                                                            #                         _select_range, GRASP, GRASP_threhold, _TRACE):

    
    check_2 = time.time()   

    # print('pieces orientation pool is: ',pieces_orientation)
    # print('pieces order pool is: ',pieces_order)
    
    # this is for the ILS to select the 
    initial_orientations_list_no_bin = []

    for each_piece in range(len(object_info_total)):
        index = find_element_index(pieces_order, each_piece, path=None)
        each_object_info = layout[index[0]][index[1]]
        initial_orientations_list_no_bin.append(each_object_info["orientation"])

    if visualisation:
        visualize_voxel_model(layout, pieces_order, container_size, container_shape)
    
    N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)

    T = check_2 - check_1
    overall_time_cost += T

    # print(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U_star is {U_star}")         
    trace(f"Constructive algorithm finished, cost {T} s in total, {N} bins are used, U is {U}, U_star is {U_star}")
    
    no_overlap = all(np.all(each_bin <= 1.5) for each_bin in topos_layout)
    trace(f"Constructive algorithm - Overlap check: {'Pass' if no_overlap else 'NOT pass!'}")
    
    if U_star == None: 
        U_star = U
        
    N_U = N - U_star
    
    # data_pool is to track result of each iteration
    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                        pieces_order, U_star, N_U, T, selected_info = "None", iteration="origin")
    
    trace("Database updated!")

    # data_pool, layout, topos_layout, radio_layout,
    # pieces_order, pieces_orientation, pieces_orientation_value_pool, 
    # pieces_packing_position, pieces_packing_position_pool,
    # U_star, N_U, T, selected_info, iteration, neighbour_type

    # draw the bar chart for the nfv
    # x_labels = [str(k) for k in nfv_pool.keys()]
    # y_values = list(nfv_pool.values())
    # 
    # plt.figure(figsize=(6, 4))
    # plt.bar(x_labels, y_values)
    # plt.xlabel("Tuple keys")
    # plt.ylabel("calculation times")
    # plt.title(f"NFV calculation times {sum(y_values)}")
    # plt.xticks(rotation=30)
    # plt.tight_layout()
    # plt.show()

    # =========================================================================================================
    # if we activate nfv_ppol, this is to count the number of replicated of NFV
    # print("all calculation time of nfv is", nfv_pool.all_nfv_cal)
    # print("validate nfv cal is",nfv_pool.val_nfv_cal)
    # print("repilcated nfv cal is",nfv_pool.rep_nfv_cal)
    # print(f"Replicated NFVs: {nfv_pool.rep_nfv_cal/nfv_pool.all_nfv_cal * 100} %")
    
    if ALG == "fixed_CA" or ALG == "random_CA" or ALG == "BLF":
        return N, U, U_star, N, U, U_star, \
            layout, topos_layout, topos_layout,  \
            pieces_order, initial_orientations_list_no_bin, initial_orientations_list_no_bin,None,None,overall_time_cost

    # ==========================================================================================================
    trace("Iterative Local Search started!")
    local_best_iteration = "origin"
    global_best_iteration = local_best_iteration

    local_best_perform = N_U
    global_best_perform = local_best_perform
    
    n_iter = 1
    n_kick = 1
    
    nonstop = True
    
    # aabb_volume_list = [get_aabb_volume(i[0]) for i in object_info]
    # aabb_sorted = np.argsort(aabb_volume_list)[::-1]

    # to track the orientaions of constructive algorithm
    # initial_selected_info = []
    # for each_piece in range(len(object_total)):
    #     index = find_element_index(pieces_order,each_piece)
    #     initial_selected_info.append((each_piece,pieces_orientation[index[0]][index[1]]))
    

    local_best_orientations_list_no_bin = list(initial_orientations_list_no_bin)
    global_best_orientations_list_no_bin = list(local_best_orientations_list_no_bin)

    # print(local_best_orientations_list_no_bin)

    local_best_list = []
    local_change_iter_list = []
    
    no_change_time = 0
    n_kick = 1  
    # kick_flag = False
    # kick_iteration = 0
    
    original_object_info_total = copy.deepcopy(object_info_total)

    while nonstop and _test_budget["count"] < _test_limit:   
        

        # decide which piece
        selected_index = pieces_selection_ls(object_info_total, alpha = alpha, seed=None)

        # decide which orientation
        # selected_orientation = orientation_selection_ls(orientations)

        other_orientations = list(orientations_list)
        other_orientations.remove(local_best_orientations_list_no_bin[selected_index])

        for each_orientation in other_orientations:
            if _test_budget["count"] >= _test_limit:
                nonstop = False
                break 

            trace(f"===================== iteration {n_iter} =========================")
            
            selected_info = (selected_index, each_orientation)

            trace (f"Object: {selected_index}, orientation: {each_orientation}, original orientation: {local_best_orientations_list_no_bin[selected_index]}")
            trace (f"Current local best iteration is {local_best_iteration}")
            trace (f"Current global best iteration is {global_best_iteration}")

            start = time.time()
            
            layout, topos_layout, radio_layout, pieces_order = repacking_new_ILS(original_object_info_total, selected_info, nfv_pool, ifv_pool, local_best_orientations_list_no_bin, data_pool, local_best_iteration, ils_orientations, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = "bounding_box",_accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL,  _TRACE = False)

            end = time.time()

            # kick_flag = False
            kick_iteration = 0

            T = end - start
            overall_time_cost += T
            
            trace(f"=== Re-pack finished! Cost {T} s in this iteration, It takes {overall_time_cost} s overall ===")
            # trace(pieces_order)
            
            # no_overlap = all(np.all(each_bin <= 1.5) for each_bin in layout)
            # trace(f"Repack - Overlap: {'No' if no_overlap else 'Yes'}")

            if overall_time_cost > time_limit:
                trace("Has reached the time limit, stop!")
                nonstop = False
                break
            
            elif n_iter == iteration_limit and overall_time_cost < time_limit:
                
                
                N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                
                
                if perform_this_iter < global_best_perform: 
                    trace(f"A GLOBAL best is found in iteration - {n_iter}")
                    no_change_time = 0
                    old_best = global_best_perform
                    global_best_iteration = n_iter
                    global_best_perform = perform_this_iter

                    local_best_orientations_list_no_bin[selected_index] = each_orientation
                    global_best_orientations_list_no_bin = list(local_best_orientations_list_no_bin)
                    trace("GLOBAL best updated") 
                    
                
                if perform_this_iter < local_best_perform: 
                    no_change_time = 0
                    # smaller values are better 
                    old_best = local_best_perform
                    local_best_iteration = n_iter
                    local_best_perform = perform_this_iter
                    local_best_list.append(local_best_perform)
                    local_change_iter_list.append(n_iter)
                    trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")

                    local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

                    trace(f"A better LOCAL result is found in iteration - {n_iter}") 
                    trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
                    trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")

                    data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                    prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                    trace("Database updated!")

                trace("Has reached the iteration limit, stop!")
                nonstop = False
                break
            
            else:
                
                N, U, U_star = get_final_performance(object_info_total,container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                    
                # data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                #             pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                
                # trace("Database updated!")
            
            if perform_this_iter < global_best_perform: 
                trace(f"A GLOBAL best is found in iteration - {n_iter}")
                no_change_time = 0
                old_best = global_best_perform
                global_best_iteration = n_iter
                global_best_perform = perform_this_iter

                local_best_orientations_list_no_bin[selected_index] = each_orientation
                global_best_orientations_list_no_bin = list(local_best_orientations_list_no_bin)
                trace("GLOBAL best updated") 

            if perform_this_iter < local_best_perform: 
                no_change_time = 0
                # smaller values are better 
                old_best = local_best_perform
                local_best_iteration = n_iter
                local_best_perform = perform_this_iter
                local_best_list.append(local_best_perform)
                local_change_iter_list.append(n_iter)
                trace(f"Previous orientations of pieces are: {local_best_orientations_list_no_bin}")

                local_best_orientations_list_no_bin[selected_index] = each_orientation # UPDATE THE BEST OPRIENTATION IF IT'S BETTER

                trace(f"A better LOCAL result is found in iteration - {n_iter}") 
                trace(f"Previous best (N-U*) is {old_best}, now is {local_best_perform}")
                trace(f"Now best orientations of pieces are: {local_best_orientations_list_no_bin}")

                data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, T, selected_info, iteration=n_iter)
                prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                trace("Database updated!")

                n_iter += 1
                break # break to the next pieces
                
            else:
                n_iter += 1 
                no_change_time += 1
                trace("This iteration NOT makes result better!")     
                trace(f"No-improvement Rep - {no_change_time}")     

            

            if no_change_time >= kick_trigger_time and _test_budget["count"] < _test_limit: 
                trace(f"===================== Kick {n_kick} =========================") 
                trace(f"!!!!Solution no change for {no_change_time} replications, a kick is triggered!!!!")
                # kick_flag = True
                no_change_time = 0 # re-count the no change time 
                trace("No change times return 0!")

                kick_iteration = n_iter

                # can be parameterised
                # if after an amount of time, the solution is unchanged
                # we give a kick to the orientations

                tem = list(global_best_orientations_list_no_bin)

                if kick_level == "small":
                    change_n = int(np.ceil(len(object_info_total)/4))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)

                elif kick_level == "medium":
                    change_n = int(np.ceil(len(object_info_total)/2))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)

                elif kick_level == "large":
                    change_n = int(np.ceil(3*len(object_info_total)/4))
                    change_position = np.random.choice(len(object_info_total), size=change_n, replace=False)   
                    for each_position in change_position:
                        tem[each_position] = np.random.choice(orientations_list)
                
                local_best_orientations_list_no_bin = tem

                trace("Kick repacking started")
                start = time.time()
                
                layout, topos_layout, radio_layout, pieces_order = kick_repacking(original_object_info_total, nfv_pool, ifv_pool, orientations, local_best_orientations_list_no_bin, container_size,
                                                                                container_shape, rho, max_radio, 
                                                                                packing_alg, orien_evaluation,
                                                                                SCH_nesting_strategy, density = 5, axis = 'z', 
                                                                                _type = selection_type, _accessible_check = accessible_check, _encourage_dbl = True, 
                                                                                _select_range = selection_range,flag_NFV_POOL=flag_NFV_POOL, _TRACE = False)
                end = time.time()
                
                overall_time_cost += (end-start)

                N, U, U_star = get_final_performance(object_info_total, container_size, container_shape, layout)
                
                if U_star == None: 
                    U_star = U
                    
                perform_this_iter = N - U_star
                
                # kick data must be the local best for the new iterations
                data_pool = update_data(data_pool, layout, topos_layout, radio_layout,
                            pieces_order, U_star, perform_this_iter, "kick", selected_info = "None", iteration=n_iter)

                local_best_list.append(local_best_perform)
                local_change_iter_list.append(kick_iteration-1)

                local_best_iteration = kick_iteration
                prune_data_pool(data_pool, "origin", local_best_iteration, global_best_iteration)
                trace("Database updated!")
                
                local_best_list.append(perform_this_iter)
                local_change_iter_list.append(local_best_iteration)

                trace(f"Current local best (N-U*) is {local_best_perform}, kick leads to {perform_this_iter}")
                trace(f"Local best iteration updated to the kick iteration {kick_iteration}!")

                if perform_this_iter < global_best_perform:
                    # highly unlikely
                    old_best = global_best_perform
                    global_best_iteration = n_iter
                    global_best_perform = perform_this_iter
                    global_best_orientations_list_no_bin = list(tem)
                
                local_best_perform = perform_this_iter
                
                n_iter += 1
                n_kick += 1
                break

        if nonstop == False:
            break
        
        
        
    # ==================================================================================
    # read the final results
    best_data = data_pool[global_best_iteration]
    original_data = data_pool["origin"]

    best_pieces_order = best_data["pieces_order"]
    best_topos_layout = best_data["bin_topos_layout"]
    # origin_topos_layout = original_data["bin_topos_layout"]

    best_current_layout = best_data["bin_real_layout"]
    origin_current_layout = original_data["bin_real_layout"]


    best_N, best_U, best_U_star = get_final_performance(object_info_total, container_size, container_shape, best_current_layout)
    origin_N, origin_U, origin_U_star = get_final_performance(object_info_total, container_size, container_shape, origin_current_layout)

    if best_U_star == None:
        best_U_star = best_U

    if origin_U_star == None:
        origin_U_star = origin_U

    # trace("============================================================")
    # trace(f"ILS finished, cost {overall_time_cost} s, num of iterations is {n_iter}")
    # trace(f"Time limit is {time_limit} s, iteration limit is {iteration_limit}")
    # trace(f"Best iteration is {best_iteration}")
    # trace(f"Constructive algorithm, {origin_N} bins are used, U_star is {origin_U_star}")
    # trace(f"After ILS, {best_N} bins are used, U_star is {best_U_star}")
    # trace(f" U_star Improvement: {(best_U_star-origin_U_star)/origin_U_star * 100}%")
    # trace(f" U Improvement: {(best_U-origin_U)/origin_U * 100}%")
    # trace("============================================================")

    
    if visualisation:
        visualize_voxel_model(best_current_layout, best_pieces_order, container_size, container_shape)
    

    # draw the bar chart for the nfv
    # x_labels = [str(k) for k in nfv_pool.keys()]
    # y_values = list(nfv_pool.values())
    # plt.figure(figsize=(6, 4))
    # plt.bar(x_labels, y_values)
    # plt.xlabel("Tuple keys")
    # plt.ylabel("calculation times")
    # plt.title(f"NFV calculation times {sum(y_values)}")
    # plt.xticks(rotation=30)
    # plt.tight_layout()
    # plt.show()

    return best_N, best_U, best_U_star, origin_N, origin_U, origin_U_star,  \
            best_current_layout, origin_current_layout, best_topos_layout,  \
            best_pieces_order, initial_orientations_list_no_bin, global_best_orientations_list_no_bin,local_best_list,local_change_iter_list,overall_time_cost
    
    # df = pd.DataFrame(data_pool)
