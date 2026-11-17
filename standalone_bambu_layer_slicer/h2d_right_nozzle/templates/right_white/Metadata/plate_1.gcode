; HEADER_BLOCK_START
; BambuStudio 02.05.00.66
; model printing time: 5m 5s; total estimated time: 11m 29s
; total layer number: 50
; total filament length [mm] : 704.85
; total filament volume [cm^3] : 1695.37
; total filament weight [g] : 2.14
; filament_density: 1.26,1.26
; filament_diameter: 1.75,1.75
; max_z_height: 10.00
; filament: 2
; HEADER_BLOCK_END

; CONFIG_BLOCK_START
; accel_to_decel_enable = 0
; accel_to_decel_factor = 50%
; activate_air_filtration = 0,0
; additional_cooling_fan_speed = 75,75
; apply_scarf_seam_on_circles = 1
; apply_top_surface_compensation = 0
; auxiliary_fan = 1
; avoid_crossing_wall_includes_support = 0
; bed_custom_model = 
; bed_custom_texture = 
; bed_exclude_area = 
; bed_temperature_formula = by_highest_temp
; before_layer_change_gcode = 
; best_object_pos = 0.3,0.5
; bottom_color_penetration_layers = 3
; bottom_shell_layers = 3
; bottom_shell_thickness = 0
; bottom_surface_density = 100%
; bottom_surface_pattern = monotonic
; bridge_angle = 0
; bridge_flow = 1
; bridge_no_support = 0
; bridge_speed = 50,50
; brim_object_gap = 0.1
; brim_type = auto_brim
; brim_width = 5
; chamber_temperatures = 0,0
; change_filament_gcode = ;======== H2D ========\n;===== 20260116 =====\nM993 A2 B2 C2 ; nozzle cam detection allow status save.\nM993 A0 B0 C0 ; nozzle cam detection not allowed.\n\n{if (filament_type[next_extruder] == "PLA") ||  (filament_type[next_extruder] == "PETG")\n ||  (filament_type[next_extruder] == "PLA-CF")  ||  (filament_type[next_extruder] == "PETG-CF")}\nM1015.4 S1 K0 ;disable E air printing detect\n{else}\nM1015.4 S0 ; disable E air printing detect\n{endif}\n\nM620 S[next_extruder]A\nM1002 gcode_claim_action : 4\nM204 S9000\n\nG1 Z{max_layer_z + 3.0} F1200\n\nM400\nM106 P1 S0\nM106 P2 S0\n\n{if toolchange_count == 2}\n; get travel path for change filament\n;M620.1 X[travel_point_1_x] Y[travel_point_1_y] F21000 P0\n;M620.1 X[travel_point_2_x] Y[travel_point_2_y] F21000 P1\n;M620.1 X[travel_point_3_x] Y[travel_point_3_y] F21000 P2\n{endif}\n\n{if ((filament_type[current_extruder] == "PLA") || (filament_type[current_extruder] == "PLA-CF") || (filament_type[current_extruder] == "PETG")) && (nozzle_diameter[current_extruder] == 0.2)}\nM620.10 A0 F74.8347 L[flush_length] H{nozzle_diameter[current_extruder]} T{flush_temperatures[current_extruder]} P[old_filament_temp] S1\n{else}\nM620.10 A0 F{flush_volumetric_speeds[current_extruder]/2.4053*60*0.8} L[flush_length] H{nozzle_diameter[current_extruder]} T{flush_temperatures[current_extruder]} P[old_filament_temp] S1\n{endif}\n\n{if ((filament_type[next_extruder] == "PLA") || (filament_type[next_extruder] == "PLA-CF") || (filament_type[next_extruder] == "PETG")) && (nozzle_diameter[next_extruder] == 0.2)}\nM620.10 A1 F74.8347 L[flush_length] H{nozzle_diameter[next_extruder]} T{flush_temperatures[next_extruder]} P[new_filament_temp] S1\n{else}\nM620.10 A1 F{flush_volumetric_speeds[next_extruder]/2.4053*60*0.8} L[flush_length] H{nozzle_diameter[next_extruder]} T{flush_temperatures[next_extruder]} P[new_filament_temp] S1\n{endif}\n\n{if long_retraction_when_cut}\nM620.11 P1 I[current_extruder] E-{retraction_distance_when_cut} F{max((flush_volumetric_speeds[current_extruder]/2.4053*60), 200)}\n{else}\nM620.11 P0 I[current_extruder] E0\n{endif}\n\n{if long_retraction_when_ec}\nM620.11 K1 I[current_extruder] R{retraction_distance_when_ec} F{max((flush_volumetric_speeds[current_extruder]/2.4053*60), 200)}\n{else}\nM620.11 K0 I[current_extruder] R0\n{endif}\n\nM620.15 C{new_filament_temp - filament_cooling_before_tower[next_extruder]}\n\nM628 S1\n{if filament_type[current_extruder] == "TPU"}\nM620.11 S0 L0 I[current_extruder] E-{retraction_distances_when_cut[current_extruder]} F{max((flush_volumetric_speeds[current_extruder]/2.4053*60), 200)}\n{else}\n{if (filament_type[current_extruder] == "PA") || (filament_type[current_extruder] == "PA-GF")}\nM620.11 S1 L0 I[current_extruder] R4 D2 E-{retraction_distances_when_cut[current_extruder]} F{max((flush_volumetric_speeds[current_extruder]/2.4053*60), 200)}\n{else}\nM620.11 S1 L0 I[current_extruder] R10 D8 E-{retraction_distances_when_cut[current_extruder]} F{max((flush_volumetric_speeds[current_extruder]/2.4053*60), 200)}\n{endif}\n{endif}\nM629\n\n{if (filament_type[current_extruder] == "TPU" || filament_type[next_extruder] == "TPU") && (old_extruder_variant != "Direct Drive TPU High Flow")}\nM620.11 H2 C331\n{else}\nM620.11 H0\n{endif}\n\n{if  (old_extruder_variant == "Direct Drive TPU High Flow") && (filament_map[current_extruder] == 2) && (filament_map[next_extruder] == 1)}\n;debug log pe:{previous_extruder} ce:{current_extruder} ne:{next_extruder} oev: {old_extruder_variant} nev:{new_extruder_variant}\n;debug fm-curr:{filament_map[current_extruder]} fm-next:{filament_map[next_extruder]}\n;sw from R2L&TPU kit, travel run a distance for sketch TPU\nG1 X30 Y30 F5000\nM400\nG1 X300 Y30 F5000\nM400\n{endif}\n\nT[next_extruder]\n\n;deretract\n{if filament_type[next_extruder] == "TPU"}\n{else}\n{if (filament_type[next_extruder] == "PA") || (filament_type[next_extruder] == "PA-GF")}\n;VG1 E1 F{max(new_filament_e_feedrate, 200)}\n;VG1 E1 F{max(new_filament_e_feedrate/2, 100)}\n{else}\n;VG1 E4 F{max(new_filament_e_feedrate, 200)}\n;VG1 E4 F{max(new_filament_e_feedrate/2, 100)}\n{endif}\n{endif}\n\n; VFLUSH_START\n\n{if flush_length>41.5}\n;VG1 E41.5 F{min(old_filament_e_feedrate,new_filament_e_feedrate)}\n;VG1 E{flush_length-41.5} F{new_filament_e_feedrate}\n{else}\n;VG1 E{flush_length} F{min(old_filament_e_feedrate,new_filament_e_feedrate)}\n{endif}\n\nSYNC T{ceil(flush_length / 125) * 5}\n\n; VFLUSH_END\n\nM1002 set_filament_type:{filament_type[next_extruder]}\n\nM400\nM83\n{if next_extruder < 255}\n\nM620.10 R{new_extruder_retracted_length}\nM628 S0\n;VM109 S[new_filament_temp]\nM629\nM400\n\n;prime_tower_interface\n{if is_prime_tower_interface && filament_tower_interface_purge_volume !=0}\nG150.1\nM620.13 W0 L{filament_tower_interface_purge_volume} T{filament_tower_interface_print_temp} R0.0\n{endif}\n;prime_tower_interface\n\nM983.3 F{filament_max_volumetric_speed[next_extruder]/2.4} A0.4 R{new_extruder_retracted_length}\n\nM400\n{if wipe_avoid_perimeter}\nG1 Y320 F30000\nG1 X{wipe_avoid_pos_x} F30000\n{endif}\nG1 Y295 F30000\nG1 Y265 F18000\nG1 Z{max_layer_z + 3.0} F3000\n{if layer_z <= (initial_layer_print_height + 0.001)}\nM204 S[initial_layer_acceleration]\n{else}\nM204 S[default_acceleration]\n{endif}\n{else}\nG1 X[x_after_toolchange] Y[y_after_toolchange] Z[z_after_toolchange] F12000\n{endif}\nM621 S[next_extruder]A\n\nM993 A3 B3 C3 ; nozzle cam detection allow status restore.\n\n{if (filament_type[next_extruder]  == "TPU")}\nM1015.3 S1;enable tpu clog detect\n{else}\nM1015.3 S0;disable tpu clog detect\n{endif}\n\n{if (filament_type[next_extruder] == "PLA") ||  (filament_type[next_extruder] == "PETG")\n ||  (filament_type[next_extruder] == "PLA-CF")  ||  (filament_type[next_extruder] == "PETG-CF")}\nM1015.4 S1 K1 H[nozzle_diameter] ;enable E air printing detect\n{else}\nM1015.4 S0 ; disable E air printing detect\n{endif}\n\nM620.6 I[next_extruder] W1 ;enable ams air printing detect\nM1002 gcode_claim_action : 0
; circle_compensation_manual_offset = 0
; circle_compensation_speed = 200,200
; close_fan_the_first_x_layers = 1,1
; complete_print_exhaust_fan_speed = 70,70
; cool_plate_temp = 35,35
; cool_plate_temp_initial_layer = 35,35
; cooling_filter_enabled = 0
; cooling_perimeter_transition_distance = 10,10
; cooling_slowdown_logic = uniform_cooling,uniform_cooling
; counter_coef_1 = 0,0
; counter_coef_2 = 0.003,0.003
; counter_coef_3 = 0.01,0.01
; counter_limit_max = 0.088,0.088
; counter_limit_min = -0.035,-0.035
; curr_bed_type = Textured PEI Plate
; default_acceleration = 8000,8000
; default_filament_colour = ;
; default_filament_profile = "Bambu PLA Basic @BBL H2D"
; default_jerk = 0
; default_nozzle_volume_type = Standard,Standard
; default_print_profile = 0.20mm Standard @BBL H2D
; deretraction_speed = 30,30
; detect_floating_vertical_shell = 1
; detect_narrow_internal_solid_infill = 1
; detect_overhang_wall = 1
; detect_thin_wall = 0
; diameter_limit = 50,50
; draft_shield = disabled
; during_print_exhaust_fan_speed = 70,70
; elefant_foot_compensation = 0.15
; embedding_wall_into_infill = 0
; enable_arc_fitting = 1
; enable_circle_compensation = 0
; enable_height_slowdown = 0,0
; enable_long_retraction_when_cut = 2
; enable_overhang_bridge_fan = 1,1
; enable_overhang_speed = 1,1
; enable_pre_heating = 1
; enable_pressure_advance = 0,0
; enable_prime_tower = 0
; enable_support = 0
; enable_support_ironing = 0
; enable_tower_interface_features = 1
; enable_wrapping_detection = 0
; enforce_support_layers = 0
; eng_plate_temp = 55,55
; eng_plate_temp_initial_layer = 55,55
; ensure_vertical_shell_thickness = enabled
; exclude_object = 1
; extruder_ams_count = 1#0|4#0;1#0|4#0
; extruder_clearance_dist_to_rod = 50
; extruder_clearance_height_to_lid = 201
; extruder_clearance_height_to_rod = 47.4
; extruder_clearance_max_radius = 96
; extruder_colour = #018001;#018001
; extruder_max_nozzle_count = 1,1
; extruder_nozzle_stats = Standard#1;Standard#1
; extruder_offset = 0x0,0x0
; extruder_printable_area = 0x0,325x0,325x320,0x320#25x0,350x0,350x320,25x320
; extruder_printable_height = 320,325
; extruder_type = Direct Drive,Direct Drive
; extruder_variant_list = "Direct Drive Standard,Direct Drive High Flow";"Direct Drive Standard,Direct Drive High Flow,Direct Drive TPU High Flow"
; fan_cooling_layer_time = 100,100
; fan_direction = left
; fan_max_speed = 80,80
; fan_min_speed = 60,60
; filament_adaptive_volumetric_speed = 0,0
; filament_adhesiveness_category = 100,100
; filament_bridge_speed = 25,25
; filament_change_length = 4,4
; filament_change_length_nc = 10,10
; filament_colour = #000000;#FFFFFF
; filament_colour_type = 1;1
; filament_cooling_before_tower = 10,10
; filament_cost = 24.99,24.99
; filament_density = 1.26,1.26
; filament_dev_ams_drying_ams_limitations = 1;0;1;0
; filament_dev_ams_drying_heat_distortion_temperature = 45,45
; filament_dev_ams_drying_temperature = 45,45,45,45,45,45,45,45
; filament_dev_ams_drying_time = 12,12,12,12,12,12,12,12
; filament_dev_chamber_drying_bed_temperature = 70,70
; filament_dev_chamber_drying_time = 12,12
; filament_dev_drying_cooling_temperature = 45,45
; filament_dev_drying_softening_temperature = 50,50
; filament_diameter = 1.75,1.75
; filament_enable_overhang_speed = 1,1
; filament_end_gcode = "; filament end gcode \n";"; filament end gcode \n"
; filament_extruder_variant = "Direct Drive Standard";"Direct Drive Standard"
; filament_flow_ratio = 0.98,0.98
; filament_flush_temp = 0,0
; filament_flush_volumetric_speed = 0,0
; filament_ids = GFA00;GFA00
; filament_is_support = 0,0
; filament_map = 1,2
; filament_map_2 = 0,1
; filament_map_mode = Auto For Flush
; filament_max_volumetric_speed = 25,25
; filament_minimal_purge_on_wipe_tower = 15,15
; filament_multi_colour = #000000;#FFFFFF
; filament_notes = 
; filament_nozzle_map = 0,1
; filament_overhang_1_4_speed = 0,0
; filament_overhang_2_4_speed = 50,50
; filament_overhang_3_4_speed = 30,30
; filament_overhang_4_4_speed = 10,10
; filament_overhang_totally_speed = 10,10
; filament_pre_cooling_temperature = 0,0
; filament_pre_cooling_temperature_nc = 0,0
; filament_prime_volume = 30,30
; filament_prime_volume_nc = 60,60
; filament_printable = 3,3
; filament_ramming_travel_time = 0,0
; filament_ramming_travel_time_nc = 0,0
; filament_ramming_volumetric_speed = -1,-1
; filament_ramming_volumetric_speed_nc = -1,-1
; filament_retract_length_nc = 14,14
; filament_retraction_length = 0.4,0.4
; filament_scarf_gap = 0%,0%
; filament_scarf_height = 10%,10%
; filament_scarf_length = 10,10
; filament_scarf_seam_type = none,none
; filament_self_index = 1,1,2,2
; filament_settings_id = "Bambu PLA Basic @BBL H2D";"Bambu PLA Basic @BBL H2D"
; filament_shrink = 100%,100%
; filament_soluble = 0,0
; filament_start_gcode = "; filament start gcode\n";"; filament start gcode\n"
; filament_tower_interface_pre_extrusion_dist = 10,10
; filament_tower_interface_pre_extrusion_length = 0,0
; filament_tower_interface_print_temp = -1,-1
; filament_tower_interface_purge_volume = 20,20
; filament_tower_ironing_area = 4,4
; filament_type = PLA;PLA
; filament_velocity_adaptation_factor = 1,1
; filament_vendor = "Bambu Lab";"Bambu Lab"
; filament_volume_map = 0,0
; filament_wipe = 1,1
; filament_wipe_distance = 1,1
; filament_z_hop_types = Spiral Lift,Spiral Lift
; filename_format = {input_filename_base}_{filament_type[0]}_{print_time}.gcode
; fill_multiline = 1
; filter_out_gap_fill = 0
; first_layer_print_sequence = 0
; first_x_layer_fan_speed = 0,0
; flush_into_infill = 0
; flush_into_objects = 0
; flush_into_support = 1
; flush_multiplier = 1,1
; flush_volumes_matrix = 0,900,90,0,356,900,90,0
; flush_volumes_vector = 140,140,140,140
; full_fan_speed_layer = 0,0
; fuzzy_skin = none
; fuzzy_skin_point_distance = 0.8
; fuzzy_skin_thickness = 0.3
; gap_infill_speed = 250,250
; gcode_add_line_number = 0
; gcode_flavor = marlin
; grab_length = 0,0
; group_algo_with_time = 0
; has_scarf_joint_seam = 0
; head_wrap_detect_zone = 
; hole_coef_1 = 0,0
; hole_coef_2 = -0.008,-0.008
; hole_coef_3 = 0.18,0.18
; hole_limit_max = 0.22,0.22
; hole_limit_min = 0.088,0.088
; host_type = octoprint
; hot_plate_temp = 55,55
; hot_plate_temp_initial_layer = 55,55
; hotend_cooling_rate = 2,2
; hotend_heating_rate = 3.6,3.6
; impact_strength_z = 13.8,13.8
; independent_support_layer_height = 1
; infill_combination = 0
; infill_direction = 45
; infill_instead_top_bottom_surfaces = 0
; infill_jerk = 9
; infill_lock_depth = 1
; infill_rotate_step = 0
; infill_shift_step = 0.4
; infill_wall_overlap = 15%
; initial_layer_acceleration = 500,500
; initial_layer_flow_ratio = 1
; initial_layer_infill_speed = 105,105
; initial_layer_jerk = 9
; initial_layer_line_width = 0.5
; initial_layer_print_height = 0.2
; initial_layer_speed = 50,50
; initial_layer_travel_acceleration = 6000,6000
; inner_wall_acceleration = 0,0
; inner_wall_jerk = 9
; inner_wall_line_width = 0.45
; inner_wall_speed = 300,300
; interface_shells = 0
; interlocking_beam = 0
; interlocking_beam_layer_count = 2
; interlocking_beam_width = 0.8
; interlocking_boundary_avoidance = 2
; interlocking_depth = 2
; interlocking_orientation = 22.5
; internal_bridge_support_thickness = 0.8
; internal_solid_infill_line_width = 0.42
; internal_solid_infill_pattern = zig-zag
; internal_solid_infill_speed = 250,250
; ironing_direction = 45
; ironing_flow = 10%
; ironing_inset = 0.21
; ironing_pattern = zig-zag
; ironing_spacing = 0.15
; ironing_speed = 30
; ironing_type = no ironing
; is_infill_first = 0
; layer_change_gcode = ;======== H2D 20250710 layer_change ========\n; layer num/total_layer_count: {layer_num+1}/[total_layer_count]\n; update layer progress\nM73 L{layer_num+1}\nM991 S0 P{layer_num} ;notify layer change\n
; layer_height = 0.2
; line_width = 0.42
; locked_skeleton_infill_pattern = zigzag
; locked_skin_infill_pattern = crosszag
; long_retractions_when_cut = 0,0
; long_retractions_when_ec = 1,1
; machine_end_gcode = ;========== H2D end ==========\n;===== date: 2025/12/26 =====\n\nG392 S0 ;turn off nozzle clog detect\nM993 A0 B0 C0 ; nozzle cam detection not allowed.\n\nM400 ; wait for buffer to clear\nG92 E0 ; zero the extruder\nG1 E-0.8 F1800 ; retract\nM400\nM211 Z1\nG1 Z{max_layer_z + 0.4} F900 ; lower z a little\n\nM1002 judge_flag timelapse_record_flag\nM622 J1\n    G150.3\n    M400 ; wait all motion done\n    M991 S0 P-1 ;end smooth timelapse at safe pos\n    M400 S5 ;wait for last picture to be taken\nM623  ;end of "timelapse_record_flag"\n\nG90\nG1 Z{max_layer_z + 10} F900 ; lower z a little\n\nG90\nM141 S0 ; turn off chamber heating\nM140 S0 ; turn off bed\nM106 S0 ; turn off fan\nM106 P2 S0 ; turn off remote part cooling fan\nM106 P3 S0 ; turn off chamber cooling fan\nM106 P9 S0 ; turn off ext toodhead cooling fan\n; pull back filament to AMS\nM620 S65535\nT65535\nG150.2\nM621 S65535\n\nM620 S65279\nT65279\nG150.2\nM621 S65279\n\nG150.3\n\nM104 S0 T0; turn off hotend\nM104 S0 T1; turn off hotend\n\nM400 ; wait all motion done\nM17 S\nM17 Z0.4 ; lower z motor current to reduce impact if there is something in the bottom\n{if (100.0 - max_layer_z/2) > 0}\n    {if (max_layer_z + 100.0 - max_layer_z/2) < 320}\n        G1 Z{max_layer_z + 100.0 - max_layer_z/2} F600\n        G1 Z{max_layer_z + 98.0 - max_layer_z/2}\n    {else}\n        G1 Z320 F600\n        G1 Z320\n    {endif}\n{else}\n    {if (max_layer_z + 4.0) < 320}\n        G1 Z{max_layer_z + 4.0} F600\n        G1 Z{max_layer_z + 2.0}\n    {else}\n        G1 Z320 F600\n        G1 Z320\n    {endif}\n{endif}\nM400 P100\nM17 R ; restore z current\n\nM220 S100  ; Reset feedrate magnitude\nM201.2 K1.0 ; Reset acc magnitude\nM73.2   R1.0 ;Reset left time magnitude\nM1002 set_gcode_claim_speed_level : 0\n\nM1015.4 S0 K0 ;disable air printing detect\n\n;=====printer finish air purification=========\nM622.1 S0\nM1002 judge_flag print_finish_air_filt_flag\n\nM622 J1\nM1002 gcode_claim_action : 66\nM145 P1\nM106 P6 S255\nM400 S180\nM106 P6 S0\nM623\n\nM622 J2\nM1002 gcode_claim_action : 66\nM145 P0\nM106 P3 S127\nM400 S180\nM106 P3 S0\nM623\n;=====printer finish air purification=========\n\n\n;=====printer finish  sound=========\nM17\nM400 S1\nM1006 S1\nM1006 A53 B10 L99 C53 D10 M99 E53 F10 N99 \nM1006 A57 B10 L99 C57 D10 M99 E57 F10 N99 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A53 B10 L99 C53 D10 M99 E53 F10 N99 \nM1006 A57 B10 L99 C57 D10 M99 E57 F10 N99 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A48 B10 L99 C48 D10 M99 E48 F10 N99 \nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 \nM1006 A60 B10 L99 C60 D10 M99 E60 F10 N99 \nM1006 W\n;=====printer finish  sound=========\nM400\nM18\n\n
; machine_hotend_change_time = 0
; machine_load_filament_time = 30
; machine_max_acceleration_e = 5000,5000,5000,5000
; machine_max_acceleration_extruding = 20000,20000,20000,20000
; machine_max_acceleration_retracting = 5000,5000,5000,5000
; machine_max_acceleration_travel = 9000,9000,9000,9000
; machine_max_acceleration_x = 20000,20000,20000,20000
; machine_max_acceleration_y = 20000,20000,20000,20000
; machine_max_acceleration_z = 500,500,500,500
; machine_max_jerk_e = 2.5,2.5,2.5,2.5
; machine_max_jerk_x = 9,9,9,9
; machine_max_jerk_y = 9,9,9,9
; machine_max_jerk_z = 3,3,3,3
; machine_max_speed_e = 50,50,50,50
; machine_max_speed_x = 1000,1000,1000,1000
; machine_max_speed_y = 1000,1000,1000,1000
; machine_max_speed_z = 30,30,30,30
; machine_min_extruding_rate = 0,0
; machine_min_travel_rate = 0,0
; machine_pause_gcode = M400 U1
; machine_prepare_compensation_time = 260
; machine_start_gcode = ;===== machine: H2D =========================\n;===== date: 20260116 =====================\n\n;M1002 set_flag extrude_cali_flag=1\n;M1002 set_flag g29_before_print_flag=1\n;M1002 set_flag auto_cali_toolhead_offset_flag=1\n;M1002 set_flag build_plate_detect_flag=1\n\nM993 A0 B0 C0 ; nozzle cam detection not allowed.\n\nM400\n;M73 P99\n\nM960 S10 P1 ; ext fan led\n\n;=====printer start sound ===================\nM17\nM400 S1\nM1006 S1\nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99 \nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99 \nM1006 A61 B9 L99 C61 D9 M99 E61 F9 N99 \nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99 \nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99 \nM1006 A61 B18 L99 C61 D18 M99 E61 F18 N99 \nM1006 W\n;=====printer start sound ===================\n\n;===== reset machine status =================\nM204 S10000\nM630 S0 P0\n\nG90\nM17 D ; reset motor current to default\nM960 S5 P1 ; turn on logo lamp\nG90\nM1002 set_gcode_claim_speed_level 5 ;Reset speed level\nM220 S100 ;Reset Feedrate\nM221 S100 ;Reset Flowrate\nM73.2   R1.0 ;Reset left time magnitude\nG29.1 Z{+0.0} ; clear z-trim value first\nM983.1 M1 \nM901 D4\nM481 S0 ; turn off cutter pos comp\nG28.140 D0; reset pre-extrude z pos\n;===== reset machine status =================\n\nM620 M ;enable remap\n\n;===== avoid end stop =================\nG91\nG380 S2 Z42 F1200\nG380 S2 Z-12 F1200\nG90\n;===== avoid end stop =================\n\n;==== set airduct mode ==== \n\n{if (overall_chamber_temperature >= 40)}\n\n    M145 P1 ; set airduct mode to heating mode for heating\n    M106 P2 S0 ; turn off auxiliary fan\n    M106 P3 S0 ; turn off chamber fan\n\n{else}\n    M145 P0 ; set airduct mode to cooling mode for cooling\n    M106 P2 S178 ; turn on auxiliary fan for cooling\n    M106 P3 S127 ; turn on chamber fan for cooling\n    M140 S0 ; stop heatbed from heating\n\n    M1002 gcode_claim_action : 29\n    M191 S0 ; wait for chamber temp\n    M106 P2 S0 ; turn off auxiliary fan\n    {if (min_vitrification_temperature <= 50)}\n        {if (nozzle_diameter == 0.2)}\n            M142 P1 R30 S35 T40 U0.3 V0.5 W0.8 O40 ; set PLA/TPU ND0.2 chamber autocooling\n        {else}\n            M142 P1 R30 S40 T45 U0.3 V0.5 W0.8 O45; set PLA/TPU ND0.4 chamber autocooling\n        {endif}\n    {else}\n        {if (!is_all_bbl_filament)}\n            M142 P1 R35 S40 T45 U0.3 V0.5 W0.8 O45 L1 ; set third-party PETG chamber autocooling\n        {else}\n            {if (nozzle_diameter == 0.2)}\n                M142 P1 R35 S45 T50 U0.3 V0.5 W0.8 O50 L1 ; set PETG ND0.2 chamber autocooling\n            {else}\n                M142 P1 R35 S50 T55 U0.3 V0.5 W0.8 O55 L1 ; set PETG ND0.4 chamber autocooling\n            {endif}\n        {endif}\n    {endif}\n    {if(cooling_filter_enabled)}\n        M145.2 P0 F0\n    {else}\n        M145.2 P0 F1\n    {endif}\n{endif}\n;==== set airduct mode ==== \n\n;===== start to heat heatbed & hotend==========\n\n    M1002 set_filament_type:{filament_type[initial_no_support_extruder]}\n\n    M104 S140 A\n    M140 S[bed_temperature_initial_layer_single]\n\n    ;===== set chamber temperature ==========\n    {if (overall_chamber_temperature >= 40)}\n        M145 P1 ; set airduct mode to heating mode\n        M141 S[overall_chamber_temperature] ; Let Chamber begin to heat\n    {endif}\n    ;===== set chamber temperature ==========\n\n;===== start to heat heatbead & hotend==========\n\n;====== cog noise reduction=================\nM982.2 S1 ; turn on cog noise reduction\n\n;===== first homing start =====\nM1002 gcode_claim_action : 13\n\nG28 X T300\n\nG150.1 F18000 ; wipe mouth to avoid filament stick to heatbed\nG150.3 F18000\nM400 P200\nM972 S24 P0 T2000\n\nM1002 gcode_claim_action : 74 ; Heatbed surface foreign object detection\n{if curr_bed_type=="Textured PEI Plate"}\nM972 S26 P0 C0\n{else}\nM972 S36 P0 C0 X1\n{endif}\nM972 S35 P0 C0\n\nM972 S41 P0 T5000 ; trash can anti-collision\n\nM1009 Q1 L1\nG91\nG380 S2 Z30 F1200 ; lower heatbed to move toolhead\nG90\nG1 X175 Y160 F30000\nG28 Z P0 T250\nM1009 Q1 L0\n\n;===== first homing end =====\n\nM400\n;M73 P99\n\n;===== detection start =====\n    \nM1002 judge_flag build_plate_detect_flag\nM622 S1\n    ;M1002 gcode_claim_action : 11 ; Indentifying build plate type\n    M972 S19 P0 C0    ; heatbed presence detection\n    M972 S31 P0 T5000 ; toolhead camera dirty detection\n    ;M1002 gcode_claim_action : 73 ; Build plate alignment detection\n    M972 S34 P0 T5000 ; heatbed plate offset detection\nM623\n\nM1002 gcode_claim_action : 72 ; Hotend Type Detection\nT1001\nM972 S14 P0 T5000 ; nozzle type detection\n\nM104 S{nozzle_temperature_initial_layer[initial_no_support_extruder]} T{filament_map[initial_no_support_extruder] % 2} ; rise temp in advance\n\nG151 P{filament_map[initial_no_support_extruder] % 2} M ; plug the heat nozzle\n\n{if max_print_z >= 145}\nM1002 gcode_claim_action : 75 ; Heatbed underside foreign object detection\nG3811 Z{max_print_z}  ; Detect obstacles at the bottom of the heated bed\n{endif}\n\n;===== detection end =====\n\nM400\n;M73 P99\n\n;===== prepare print temperature and material ==========\nM400\nM211 X0 Y0 Z0 ;turn off soft endstop\nM975 S1 ; turn on input shaping\n\nG29.2 S0 ; avoid invalid abl data\n\n{if ((filament_type[initial_no_support_extruder] == "PLA") || (filament_type[initial_no_support_extruder] == "PLA-CF") || (filament_type[initial_no_support_extruder] == "PETG")) && (nozzle_diameter[initial_no_support_extruder] == 0.2)}\nM620.10 A0 F74.8347 H{nozzle_diameter[initial_no_support_extruder]} T{flush_temperatures[initial_no_support_extruder]} P{nozzle_temperature_initial_layer[initial_no_support_extruder]} S1\nM620.10 A1 F74.8347 H{nozzle_diameter[initial_no_support_extruder]} T{flush_temperatures[initial_no_support_extruder]} P{nozzle_temperature_initial_layer[initial_no_support_extruder]} S1\n{else}\nM620.10 A0 F{flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60*0.8} H{nozzle_diameter[initial_no_support_extruder]} T{flush_temperatures[initial_no_support_extruder]} P{nozzle_temperature_initial_layer[initial_no_support_extruder]} S1\nM620.10 A1 F{flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60*0.8} H{nozzle_diameter[initial_no_support_extruder]} T{flush_temperatures[initial_no_support_extruder]} P{nozzle_temperature_initial_layer[initial_no_support_extruder]} S1\n{endif}\n\nM620.11 P0 I[initial_no_support_extruder] E0\n\n{if long_retraction_when_ec }\nM620.11 K1 I[initial_no_support_extruder] R{retraction_distance_when_ec} F{max((flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60), 200)}\n{else}\nM620.11 K0 I[initial_no_support_extruder] R0\n{endif}\n\nM628 S1\n{if filament_type[initial_no_support_extruder] == "TPU"}\n    M620.11 S0 L0 I[initial_no_support_extruder] E-{retraction_distances_when_cut[initial_no_support_extruder]} F{flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60}\n{else}\n{if (filament_type[initial_no_support_extruder] == "PA") ||  (filament_type[initial_no_support_extruder] == "PA-GF")}\n    M620.11 S1 L0 I[initial_no_support_extruder] R4 D2 E-{retraction_distances_when_cut[initial_no_support_extruder]} F{flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60}\n{else}\n    M620.11 S1 L0 I[initial_no_support_extruder] R10 D8 E-{retraction_distances_when_cut[initial_no_support_extruder]} F{flush_volumetric_speeds[initial_no_support_extruder]/2.4053*60}\n{endif}\n{endif}\nM629\n\nM620 S[initial_no_support_extruder]A   ; switch material if AMS exist\nM1002 gcode_claim_action : 4\nM1002 set_filament_type:UNKNOWN\nM400\nT[initial_no_support_extruder]\nM400\nM628 S0\nM629\nM400\nM1002 set_filament_type:{filament_type[initial_no_support_extruder]}\nM621 S[initial_no_support_extruder]A\n\nM104 S{nozzle_temperature_initial_layer[initial_no_support_extruder]}\nM400\nM106 P1 S0\n\nG29.2 S1\n;===== prepare print temperature and material ==========\n\nM400\n;M73 P99\n\n;===== auto extrude cali start =========================\nM975 S1\nM1002 judge_flag extrude_cali_flag\n\nM622 J0\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4} A0.4 ; cali dynamic extrusion compensation\nM623\n\nM622 J1\n    M1002 set_filament_type:{filament_type[initial_no_support_extruder]}\n    M1002 gcode_claim_action : 8\n\n    M109 S{nozzle_temperature[initial_no_support_extruder]}\n\n    G90\n    M83\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4} A0.4 ; cali dynamic extrusion compensation\n\n    M400\n    M106 P1 S255\n    M400 S5\n    M106 P1 S0\n    G150.3\nM623\n\nM622 J2\n    M1002 set_filament_type:{filament_type[initial_no_support_extruder]}\n    M1002 gcode_claim_action : 8\n\n    M109 S{nozzle_temperature[initial_no_support_extruder]}\n\n    G90\n    M83\n    M983.3 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4} A0.4 ; cali dynamic extrusion compensation\n\n    M400\n    M106 P1 S255\n    M400 S5\n    M106 P1 S0\n    G150.3\nM623\n\n;===== auto extrude cali end =========================\n\n{if filament_type[initial_no_support_extruder] == "TPU"}\n    G150.2\n    G150.1\n    G150.2\n    G150.1\n    G150.2\n    G150.1\n{else}\n    M106 P1 S0\n    M400 S2\n    M109 S{nozzle_temperature[initial_no_support_extruder]} ; wait tmpr to extrude\n    M83\n    {if(nozzle_diameter == 0.8)}\n        G1 E60 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4053*60}\n    {else}\n        G1 E45 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4053*60}\n    {endif}\n    G1 E-3 F1800\n    M400 P500\n    G150.2\n    G150.1\n{endif}\n\nG91\nG1 Y-16 F12000 ; move away from the trash bin\nG90\n\nM400\n;M73 P99\n\n;===== wipe right nozzle start =====\n\nM1002 gcode_claim_action : 14\n    G150 T{nozzle_temperature_initial_layer[initial_no_support_extruder]}\n    {if (overall_chamber_temperature >= 40)}\n        G150 T{nozzle_temperature_initial_layer[initial_no_support_extruder] - 80}\n    {endif}\nM106 S255 ; turn on fan to cool the nozzle\n\n;===== wipe left nozzle end =====\n\nM400\n;M73 P99\n\n{if (overall_chamber_temperature >= 40)}\n    M1002 gcode_claim_action : 49\n    M191 S[overall_chamber_temperature] ; wait for chamber temp\n{endif}\n\nM400\n;M73 P99\n\n;===== bed leveling ==================================\n\nM1002 judge_flag g29_before_print_flag\n\nM190 S[bed_temperature_initial_layer_single]; ensure bed temp\nM109 S140 A\nM106 S0 ; turn off fan , too noisy\n\nG91\nG1 Z5 F1200\nG90\nG1 X175 Y160 F30000\n\nM622 J1\n    M1002 gcode_claim_action : 1\n    G29.20 A3\n    G29 A1 O X{first_layer_print_min[0]} Y{first_layer_print_min[1]} I{first_layer_print_size[0]} J{first_layer_print_size[1]} R \n    M400\n    M500 ; save cali data\nM623\n    \nM622 J2\n    M1002 gcode_claim_action : 1\n    {if has_tpu_in_first_layer}\n        G29.20 A3\n        G29 A1 O X{first_layer_print_min[0]} Y{first_layer_print_min[1]} I{first_layer_print_size[0]} J{first_layer_print_size[1]} R\n    {else}\n        G29.20 A4\n        G29 A2 O X{first_layer_print_min[0]} Y{first_layer_print_min[1]} I{first_layer_print_size[0]} J{first_layer_print_size[1]} R\n    {endif}\n    M400\n    M500 ; save cali data\nM623\n\nM622 J0\n    G28 R\nM623\n\n;===== bed leveling end ================================\n\n;===== z ofst cali start =====\n\n    M190 S[bed_temperature_initial_layer_single]; ensure bed temp\n\n    G383 O0 M2 T140\n    M500\n\n;===== z ofst cali end =====\n\nG39.1 ; cali nozzle wrapped detection pos\nM500\n\nG90\nG1 Z5 F1200\nG1 X270 Y-0.5 F60000\nG28.140 S0 ; cali pre-extrude z pos\n\nM141 S[overall_chamber_temperature]\nM104 S{nozzle_temperature_initial_layer[initial_no_support_extruder]} A\n\n;===== mech mode sweep start =====\n    M1002 gcode_claim_action : 3\n\n    G90\n    G1 Z5 F1200\n    G1 X187 Y160 F20000\n    T1000\n    M400 P200\n\n    M970.3 Q1 A5 K0 O1\n    M974 Q1 S2 P0\n\n    M970.3 Q0 A5 K0 O1\n    M974 Q0 S2 P0\n\n    M970.2 Q2 K0 W38 Z0.01\n    M974 Q2 S2 P0\n    M500\n\n    M975 S1\n;===== mech mode sweep end =====\n\nM400\n;M73 P99\n\nG150.3 ; move to garbage can to wait for temp\nM1026\nG29.9\n\n;===== xy ofst cali start =====\n\nM1002 judge_flag auto_cali_toolhead_offset_flag\n\nM622 J0\n    M1012.5 N1 R1\n    M500\nM623\n\nM622 J1\n    M1002 gcode_claim_action : 39\n    M141 S0\n    M620.17 T0 S{nozzle_temperature_initial_layer[(first_non_support_filaments[0] != -1 ? first_non_support_filaments[0] : first_filaments[0])]} L{(first_non_support_filaments[0] != -1 ? first_non_support_filaments[0] : first_filaments[0])}\n    M620.17 T1 S{nozzle_temperature_initial_layer[(first_non_support_filaments[1] != -1 ? first_non_support_filaments[1] : first_filaments[1])]} L{(first_non_support_filaments[1] != -1 ? first_non_support_filaments[1] : first_filaments[1])}\n    G383 O1 T{nozzle_temperature_initial_layer[initial_no_support_extruder]} L{initial_no_support_extruder}\n    M500\n    M141 S[overall_chamber_temperature]\nM623\n\nM622 J2\n    M1002 gcode_claim_action : 39\n    M141 S0\n    M620.17 T0 S{nozzle_temperature_initial_layer[(first_non_support_filaments[0] != -1 ? first_non_support_filaments[0] : first_filaments[0])]} L{(first_non_support_filaments[0] != -1 ? first_non_support_filaments[0] : first_filaments[0])}\n    M620.17 T1 S{nozzle_temperature_initial_layer[(first_non_support_filaments[1] != -1 ? first_non_support_filaments[1] : first_filaments[1])]} L{(first_non_support_filaments[1] != -1 ? first_non_support_filaments[1] : first_filaments[1])}\n    G383.3 T{nozzle_temperature_initial_layer[initial_no_support_extruder]} L{initial_no_support_extruder}\n    M500\n    M141 S[overall_chamber_temperature]\nM623\n;===== xy ofst cali end =====\n\nM400\n;M73 P99\n\nM1002 gcode_claim_action : 0\nM400\n\n;============switch again==================\n\nM211 X0 Y0 Z0 ;turn off soft endstop\nG91\nG1 Z6 F1200\nG90\nM1002 set_filament_type:{filament_type[initial_no_support_extruder]}\nM620 S[initial_no_support_extruder]A\nM400\nT[initial_no_support_extruder]\nM400\nM628 S0\nM629\nM400\nM621 S[initial_no_support_extruder]A\n\n;============switch again==================\n\nM400\n;M73 P99\n\n;===== wait temperature reaching the reference value =======\n\nM104 S{nozzle_temperature_initial_layer[initial_no_support_extruder]} ; rise to print tmpr\n\nM140 S[bed_temperature_initial_layer_single] \nM190 S[bed_temperature_initial_layer_single] \n\n    ;========turn off light and fans =============\n    M960 S1 P0 ; turn off laser\n    M960 S2 P0 ; turn off laser\n    M106 S0 ; turn off fan\n    M106 P2 S0 ; turn off big fan\n    ;==== set ext toodhead cooling fan ==== \n    {if (min_vitrification_temperature <= 50)}\n    M106 P9 S255\n    {endif}\n    ;============set motor current==================\n    M400 S1\n\n;===== wait temperature reaching the reference value =======\n\nM400\n;M73 P99\n\n;===== for Textured PEI Plate , lower the nozzle as the nozzle was touching topmost of the texture when homing ==\n    {if curr_bed_type=="Textured PEI Plate"}\n        {if nozzle_diameter[initial_no_support_extruder] == 0.2}\n            G29.1 Z{-0.01} ; for Textured PEI Plate\n        {else}\n            G29.1 Z{-0.02} ; for Textured PEI Plate\n        {endif}\n    {else}\n        {if nozzle_diameter[initial_no_support_extruder] == 0.2}\n            G29.1 Z{0.01} ; for Textured PEI Plate\n        {endif}\n    {endif}\n    \nG150.1\n\nM975 S1 ; turn on mech mode supression\nM983.4 S1 ; turn on deformation compensation \nG29.2 S1 ; turn on pos comp\nG29.7 S1\n\nG90\nG1 Z5 F1200\nG1 Y295 F30000\nG1 Y265 F18000\n\n;===== nozzle load line ===============================\n    G29.2 S1 ; ensure z comp turn on\n    G90\n    M83\n    G1 Z5 F1200\n    G1 X270 Y-0.5 F60000\n    G28.14 R0\n    G29.2 S0\n    G91\n    G1 Z0.8 F1200\n    G90\n    G1 X250 F60000\n    M109 S{nozzle_temperature_initial_layer[initial_no_support_extruder]}\n    M83\n{if (filament_type[initial_no_support_extruder] == "TPU")}\n    G1 E5 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4053*60}\n{endif}\n    G1 E5 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4053*60}\n    G1 X290 E10 F{filament_max_volumetric_speed[initial_no_support_extruder]/2.4053*60}\n    G91\n    G3 Z0.4 I1.217 J0 P1 F60000\n    G90\n    M83\n    G29.2 S1 ; ensure z comp turn on\n;===== noozle load line end ===========================\n\nM400\n;M73 P99\n\nM993 A1 B1 C1 ; nozzle cam detection allowed.\n\n{if (filament_type[initial_no_support_extruder] == "TPU")}\nM1015.3 S1;enable tpu clog detect\n{else}\nM1015.3 S0;disable tpu clog detect\n{endif}\n\n{if (filament_type[initial_no_support_extruder] == "PLA") ||  (filament_type[initial_no_support_extruder] == "PETG")\n ||  (filament_type[initial_no_support_extruder] == "PLA-CF")  ||  (filament_type[initial_no_support_extruder] == "PETG-CF")}\nM1015.4 S1 K1 H[nozzle_diameter] ;enable E air printing detect\n{else}\nM1015.4 S0 K0 H[nozzle_diameter] ;disable E air printing detect\n{endif}\n\nM620.6 I[initial_no_support_extruder] W1 ;enable ams air printing detect\n\nM211 Z1\nG29.99\n\n\n
; machine_switch_extruder_time = 5.6
; machine_unload_filament_time = 30
; master_extruder_id = 2
; max_bridge_length = 0
; max_layer_height = 0.28,0.28
; max_travel_detour_distance = 0
; min_bead_width = 85%
; min_feature_size = 25%
; min_layer_height = 0.08,0.08
; minimum_sparse_infill_area = 15
; mmu_segmented_region_interlocking_depth = 0
; mmu_segmented_region_max_width = 0
; no_slow_down_for_cooling_on_outwalls = 0,0
; nozzle_diameter = 0.4,0.4
; nozzle_flush_dataset = 1,1
; nozzle_height = 4
; nozzle_temperature = 220,220
; nozzle_temperature_initial_layer = 220,220
; nozzle_temperature_range_high = 240,240
; nozzle_temperature_range_low = 190,190
; nozzle_type = hardened_steel,hardened_steel
; nozzle_volume = 130,145
; nozzle_volume_type = Standard,Standard
; only_one_wall_first_layer = 0
; ooze_prevention = 0
; other_layers_print_sequence = 0
; other_layers_print_sequence_nums = 0
; outer_wall_acceleration = 5000,5000
; outer_wall_jerk = 9
; outer_wall_line_width = 0.42
; outer_wall_speed = 200,200
; overhang_1_4_speed = 0,0
; overhang_2_4_speed = 50,50
; overhang_3_4_speed = 30,30
; overhang_4_4_speed = 10,10
; overhang_fan_speed = 100,100
; overhang_fan_threshold = 50%,50%
; overhang_threshold_participating_cooling = 95%,95%
; overhang_totally_speed = 10,10
; override_filament_scarf_seam_setting = 0
; override_process_overhang_speed = 0,0
; physical_extruder_map = 1,0
; post_process = 
; pre_start_fan_time = 2,2
; precise_outer_wall = 0
; precise_z_height = 0
; pressure_advance = 0.02,0.02
; prime_tower_brim_width = -1
; prime_tower_enable_framework = 0
; prime_tower_extra_rib_length = 0
; prime_tower_fillet_wall = 1
; prime_tower_flat_ironing = 1
; prime_tower_infill_gap = 150%
; prime_tower_lift_height = -1
; prime_tower_lift_speed = 90
; prime_tower_max_speed = 90
; prime_tower_rib_wall = 1
; prime_tower_rib_width = 8
; prime_tower_skip_points = 1
; prime_tower_width = 60
; prime_volume_mode = Default
; print_compatible_printers = "Bambu Lab H2D 0.4 nozzle"
; print_extruder_id = 1,2
; print_extruder_variant = "Direct Drive Standard";"Direct Drive Standard"
; print_flow_ratio = 1
; print_sequence = by layer
; print_settings_id = 0.20mm Standard @BBL H2D
; printable_area = 0x0,350x0,350x320,0x320
; printable_height = 325
; printer_extruder_id = 1,2
; printer_extruder_variant = "Direct Drive Standard";"Direct Drive Standard"
; printer_model = Bambu Lab H2D
; printer_notes = 
; printer_settings_id = Bambu Lab H2D 0.4 nozzle
; printer_structure = corexy
; printer_technology = FFF
; printer_variant = 0.4
; printhost_authorization_type = key
; printhost_ssl_ignore_revoke = 0
; printing_by_object_gcode = 
; process_notes = 
; raft_contact_distance = 0.1
; raft_expansion = 1.5
; raft_first_layer_density = 90%
; raft_first_layer_expansion = -1
; raft_layers = 0
; reduce_crossing_wall = 0
; reduce_fan_stop_start_freq = 1,1
; reduce_infill_retraction = 1
; required_nozzle_HRC = 3,3
; resolution = 0.012
; retract_before_wipe = 0%,0%
; retract_length_toolchange = 2,2
; retract_lift_above = 0,0
; retract_lift_below = 319,319
; retract_restart_extra = 0,0
; retract_restart_extra_toolchange = 0,0
; retract_when_changing_layer = 1,1
; retraction_distances_when_cut = 10,10
; retraction_distances_when_ec = 10,10
; retraction_length = 0.8,0.8
; retraction_minimum_travel = 1,1
; retraction_speed = 30,30
; role_base_wipe_speed = 1
; scan_first_layer = 0
; scarf_angle_threshold = 155
; seam_gap = 15%
; seam_placement_away_from_overhangs = 0
; seam_position = aligned
; seam_slope_conditional = 1
; seam_slope_entire_loop = 0
; seam_slope_gap = 0
; seam_slope_inner_walls = 1
; seam_slope_min_length = 10
; seam_slope_start_height = 10%
; seam_slope_steps = 10
; seam_slope_type = none
; silent_mode = 0
; single_extruder_multi_material = 1
; skeleton_infill_density = 15%
; skeleton_infill_line_width = 0.45
; skin_infill_density = 15%
; skin_infill_depth = 2
; skin_infill_line_width = 0.45
; skirt_distance = 2
; skirt_height = 1
; skirt_loops = 0
; slice_closing_radius = 0.049
; slicing_mode = regular
; slow_down_for_layer_cooling = 1,1
; slow_down_layer_time = 4,4
; slow_down_min_speed = 20,20
; slowdown_end_acc = 100000,100000
; slowdown_end_height = 400,400
; slowdown_end_speed = 1000,1000
; slowdown_start_acc = 100000,100000
; slowdown_start_height = 0,0
; slowdown_start_speed = 1000,1000
; small_perimeter_speed = 50%,50%
; small_perimeter_threshold = 0,0
; smooth_coefficient = 4
; smooth_speed_discontinuity_area = 1
; solid_infill_filament = 0
; sparse_infill_acceleration = 100%,100%
; sparse_infill_anchor = 400%
; sparse_infill_anchor_max = 20
; sparse_infill_density = 15%
; sparse_infill_filament = 0
; sparse_infill_lattice_angle_1 = -45
; sparse_infill_lattice_angle_2 = 45
; sparse_infill_line_width = 0.45
; sparse_infill_pattern = grid
; sparse_infill_speed = 350,350
; spiral_mode = 0
; spiral_mode_max_xy_smoothing = 200%
; spiral_mode_smooth = 0
; standby_temperature_delta = -5
; start_end_points = 30x-3,54x245
; supertack_plate_temp = 40,40
; supertack_plate_temp_initial_layer = 40,40
; support_air_filtration = 0
; support_angle = 0
; support_base_pattern = default
; support_base_pattern_spacing = 2.5
; support_bottom_interface_spacing = 0.5
; support_bottom_z_distance = 0.2
; support_chamber_temp_control = 1
; support_cooling_filter = 1
; support_critical_regions_only = 0
; support_expansion = 0
; support_filament = 0
; support_interface_bottom_layers = 2
; support_interface_filament = 0
; support_interface_loop_pattern = 0
; support_interface_not_for_body = 1
; support_interface_pattern = auto
; support_interface_spacing = 0.5
; support_interface_speed = 80,80
; support_interface_top_layers = 2
; support_ironing_direction = 0
; support_ironing_flow = 10%
; support_ironing_inset = 0
; support_ironing_pattern = zig-zag
; support_ironing_spacing = 0.15
; support_ironing_speed = 30
; support_line_width = 0.42
; support_object_first_layer_gap = 0.2
; support_object_skip_flush = 0
; support_object_xy_distance = 0.35
; support_on_build_plate_only = 0
; support_remove_small_overhang = 1
; support_speed = 150,150
; support_style = default
; support_threshold_angle = 30
; support_top_z_distance = 0.2
; support_type = tree(auto)
; symmetric_infill_y_axis = 0
; temperature_vitrification = 45,45
; template_custom_gcode = 
; textured_plate_temp = 55,55
; textured_plate_temp_initial_layer = 55,55
; thick_bridges = 0
; thumbnail_size = 50x50
; time_lapse_gcode = ;======== H2D 20251104========\n; SKIPPABLE_START\n; SKIPTYPE: timelapse\nM622.1 S1 ; for prev firmware, default turned on\n\nM1002 judge_flag timelapse_record_flag\n\n    M622 J1\n    M993 A2 B2 C2\n    M993 A0 B0 C0\n    \n    M622.1 S0 ; for prev firmware, default turn off\n    M1002 set_flag smooth_safe_pos_suppoprt_flag=1\n    M1002 judge_flag smooth_safe_pos_suppoprt_flag\n    \n    M622 J0\n        {if !spiral_mode && !(has_timelapse_safe_pos && timelapse_type == 0) }\n            {if most_used_physical_extruder_id!= curr_physical_extruder_id || timelapse_type == 1}\n                M83\n                G1 Z{max_layer_z + 0.4} F1200\n                M400\n            {endif}\n        {endif}\n\n        {if has_timelapse_safe_pos && timelapse_type == 0 && !spiral_mode}\n            M9711 M{timelapse_type} E{most_used_physical_extruder_id} X{timelapse_pos_x} Y{timelapse_pos_y} Z{layer_z + 0.4} S11 C10 O0 T3000\n        {else}\n            {if spiral_mode}\n                M971 S11 C10 O0\n                M1004 S5 P1  ; external shutter\n            {else}\n                M9711 M{timelapse_type} E{most_used_physical_extruder_id} Z{layer_z + 0.4} S11 C10 O0 T3000\n            {endif}\n        {endif}\n\n        {if !spiral_mode && !(has_timelapse_safe_pos && timelapse_type == 0) }\n            {if most_used_physical_extruder_id!= curr_physical_extruder_id || timelapse_type == 1}\n                G90\n                G1 Z{max_layer_z + 3.0} F1200\n                G1 Y295 F30000\n                G1 Y265 F18000\n                M83\n            {endif}\n        {endif}\n    M623\n\n    M622 J1\n        {if !spiral_mode && !(has_timelapse_safe_pos) }\n            {if most_used_physical_extruder_id!= curr_physical_extruder_id || timelapse_type == 1}\n                M83\n                G1 Z{max_layer_z + 0.4} F1200\n                M400\n            {endif}\n        {endif}\n\n        {if has_timelapse_safe_pos && !spiral_mode}\n            M9711 M{timelapse_type} E{most_used_physical_extruder_id} U{timelapse_pos_x} V{timelapse_pos_y} Z{layer_z + 0.4} S11 C10 O0 T3000\n        {else}\n            {if spiral_mode}\n                M971 S11 C10 O0\n                M1004 S5 P1  ; external shutter\n            {else}\n                M9711 M{timelapse_type} E{most_used_physical_extruder_id} Z{layer_z + 0.4} S11 C10 O0 T3000\n            {endif}\n        {endif}\n\n        {if !spiral_mode && !(has_timelapse_safe_pos) }\n            {if most_used_physical_extruder_id!= curr_physical_extruder_id || timelapse_type == 1}\n                G90\n                G1 Z{max_layer_z + 3.0} F1200\n                G1 Y295 F30000\n                G1 Y265 F18000\n                M83\n            {endif}\n        {endif}\n    M623\n\n    M993 A3 B3 C3\n\nM623\n; SKIPPABLE_END\n
; timelapse_type = 0
; top_area_threshold = 200%
; top_color_penetration_layers = 5
; top_one_wall_type = all top
; top_shell_layers = 5
; top_shell_thickness = 1
; top_solid_infill_flow_ratio = 1,1
; top_surface_acceleration = 2000,2000
; top_surface_density = 100%
; top_surface_jerk = 9
; top_surface_line_width = 0.42
; top_surface_pattern = monotonicline
; top_surface_speed = 200,200
; top_z_overrides_xy_distance = 0
; travel_acceleration = 10000,10000
; travel_jerk = 9
; travel_short_distance_acceleration = 250,250
; travel_speed = 1000,1000
; travel_speed_z = 0,0
; tree_support_branch_angle = 45
; tree_support_branch_diameter = 2
; tree_support_branch_diameter_angle = 5
; tree_support_branch_distance = 5
; tree_support_wall_count = -1
; upward_compatible_machine = "Bambu Lab H2D Pro 0.4 nozzle"
; use_firmware_retraction = 0
; use_relative_e_distances = 1
; vertical_shell_speed = 80%,80%
; volumetric_speed_coefficients = "0 0 0 0 0 0";"0 0 0 0 0 0"
; wall_distribution_count = 1
; wall_filament = 0
; wall_generator = classic
; wall_loops = 2
; wall_sequence = inner wall/outer wall
; wall_transition_angle = 10
; wall_transition_filter_deviation = 25%
; wall_transition_length = 100%
; wipe = 1,1
; wipe_distance = 2,2
; wipe_speed = 80%
; wipe_tower_no_sparse_layers = 0
; wipe_tower_rotation_angle = 0
; wipe_tower_x = 165
; wipe_tower_y = 250
; wrapping_detection_gcode = ;======== H2D 20250729 clumping ========\n{if !spiral_mode}\n    M622.1 S0 ; for previous firmware, default turn off\n    M1002 set_flag g39_forced_detection_flag=1\n    M1002 judge_flag g39_forced_detection_flag\n    M622 J1\n        {if layer_num == 3 || layer_num == 10 || layer_num == 19}\n            M993 A2 B2 C2 ; nozzle cam detection allow status save.\n            M993 A0 B0 C0 ; nozzle cam detection not allowed.\n\n            M400 P100\n\n            G39\n\n            G90\n            G1 Y295 F30000\n            G1 Y265 F18000\n            \n            M993 A3 B3 C3 ; nozzle cam detection allow status restore.\n        {endif}\n    M623\n{endif}\n
; wrapping_detection_layers = 20
; wrapping_exclude_area = 145x310,256x310,256x326,145x326
; xy_contour_compensation = 0
; xy_hole_compensation = 0
; z_direction_outwall_speed_continuous = 1
; z_hop = 0.4,0.4
; z_hop_types = Auto Lift,Auto Lift
; CONFIG_BLOCK_END

; EXECUTABLE_BLOCK_START
M73 P0 R11
M201 X20000 Y20000 Z500 E5000
M203 X1000 Y1000 Z30 E50
M204 P20000 R5000 T20000
M205 X9.00 Y9.00 Z3.00 E2.50
M106 S0
M106 P2 S0
; FEATURE: Custom
;===== machine: H2D =========================
;===== date: 20260116 =====================

;M1002 set_flag extrude_cali_flag=1
;M1002 set_flag g29_before_print_flag=1
;M1002 set_flag auto_cali_toolhead_offset_flag=1
;M1002 set_flag build_plate_detect_flag=1

M993 A0 B0 C0 ; nozzle cam detection not allowed.

M400
;M73 P99

M960 S10 P1 ; ext fan led

;=====printer start sound ===================
M17
M400 S1
M1006 S1
M1006 A53 B9 L99 C53 D9 M99 E53 F9 N99 
M1006 A56 B9 L99 C56 D9 M99 E56 F9 N99 
M1006 A61 B9 L99 C61 D9 M99 E61 F9 N99 
M1006 A53 B9 L99 C53 D9 M99 E53 F9 N99 
M1006 A56 B9 L99 C56 D9 M99 E56 F9 N99 
M1006 A61 B18 L99 C61 D18 M99 E61 F18 N99 
M1006 W
;=====printer start sound ===================

;===== reset machine status =================
M204 S10000
M630 S0 P0

G90
M17 D ; reset motor current to default
M960 S5 P1 ; turn on logo lamp
G90
M1002 set_gcode_claim_speed_level 5 ;Reset speed level
M220 S100 ;Reset Feedrate
M221 S100 ;Reset Flowrate
M73.2   R1.0 ;Reset left time magnitude
G29.1 Z0 ; clear z-trim value first
M983.1 M1 
M901 D4
M481 S0 ; turn off cutter pos comp
G28.140 D0; reset pre-extrude z pos
;===== reset machine status =================

M620 M ;enable remap

;===== avoid end stop =================
G91
G380 S2 Z42 F1200
G380 S2 Z-12 F1200
G90
;===== avoid end stop =================

;==== set airduct mode ==== 


    M145 P0 ; set airduct mode to cooling mode for cooling
    M106 P2 S178 ; turn on auxiliary fan for cooling
    M106 P3 S127 ; turn on chamber fan for cooling
    M140 S0 ; stop heatbed from heating

    M1002 gcode_claim_action : 29
    M191 S0 ; wait for chamber temp
    M106 P2 S0 ; turn off auxiliary fan
    
        
            M142 P1 R30 S40 T45 U0.3 V0.5 W0.8 O45; set PLA/TPU ND0.4 chamber autocooling
        
    
    
        M145.2 P0 F1
    

;==== set airduct mode ==== 

;===== start to heat heatbed & hotend==========

    M1002 set_filament_type:PLA

    M104 S140 A
    M140 S55

    ;===== set chamber temperature ==========
    
    ;===== set chamber temperature ==========

;===== start to heat heatbead & hotend==========

;====== cog noise reduction=================
M982.2 S1 ; turn on cog noise reduction

;===== first homing start =====
M1002 gcode_claim_action : 13

G28 X T300

G150.1 F18000 ; wipe mouth to avoid filament stick to heatbed
G150.3 F18000
M400 P200
M972 S24 P0 T2000

M1002 gcode_claim_action : 74 ; Heatbed surface foreign object detection

M972 S26 P0 C0

M972 S35 P0 C0

M972 S41 P0 T5000 ; trash can anti-collision

M1009 Q1 L1
G91
G380 S2 Z30 F1200 ; lower heatbed to move toolhead
G90
G1 X175 Y160 F30000
G28 Z P0 T250
M1009 Q1 L0

;===== first homing end =====

M400
;M73 P99

;===== detection start =====
    
M1002 judge_flag build_plate_detect_flag
M622 S1
    ;M1002 gcode_claim_action : 11 ; Indentifying build plate type
    M972 S19 P0 C0    ; heatbed presence detection
    M972 S31 P0 T5000 ; toolhead camera dirty detection
    ;M1002 gcode_claim_action : 73 ; Build plate alignment detection
    M972 S34 P0 T5000 ; heatbed plate offset detection
M623

M1002 gcode_claim_action : 72 ; Hotend Type Detection
T1001
M972 S14 P0 T5000 ; nozzle type detection

M104 S220 T0 ; rise temp in advance

G151 P0 M ; plug the heat nozzle



;===== detection end =====

M400
;M73 P99

;===== prepare print temperature and material ==========
M400
M211 X0 Y0 Z0 ;turn off soft endstop
M975 S1 ; turn on input shaping

G29.2 S0 ; avoid invalid abl data


M620.10 A0 F498.898 H0.4 T240 P220 S1
M620.10 A1 F498.898 H0.4 T240 P220 S1


M620.11 P0 I1 E0


M620.11 K1 I1 R10 F623.623


M628 S1


    M620.11 S1 L0 I1 R10 D8 E-10 F623.623


M629

M620 S1A   ; switch material if AMS exist
M1002 gcode_claim_action : 4
M1002 set_filament_type:UNKNOWN
M400
T1
M400
M628 S0
M629
M400
M1002 set_filament_type:PLA
M621 S1A

M104 S220
M400
M106 P1 S0

G29.2 S1
;===== prepare print temperature and material ==========

M400
;M73 P99

;===== auto extrude cali start =========================
M975 S1
M1002 judge_flag extrude_cali_flag

M622 J0
    M983.3 F10.4167 A0.4 ; cali dynamic extrusion compensation
M623

M622 J1
    M1002 set_filament_type:PLA
    M1002 gcode_claim_action : 8

    M109 S220

    G90
    M83
    M983.3 F10.4167 A0.4 ; cali dynamic extrusion compensation

    M400
    M106 P1 S255
    M400 S5
    M106 P1 S0
    G150.3
M623

M622 J2
    M1002 set_filament_type:PLA
    M1002 gcode_claim_action : 8

    M109 S220

    G90
    M83
    M983.3 F10.4167 A0.4 ; cali dynamic extrusion compensation

    M400
    M106 P1 S255
    M400 S5
    M106 P1 S0
    G150.3
M623

;===== auto extrude cali end =========================


    M106 P1 S0
    M400 S2
    M109 S220 ; wait tmpr to extrude
    M83
    
        G1 E45 F623.623
    
    G1 E-3 F1800
    M400 P500
    G150.2
    G150.1


G91
M73 P14 R9
G1 Y-16 F12000 ; move away from the trash bin
G90

M400
;M73 P99

;===== wipe right nozzle start =====

M1002 gcode_claim_action : 14
    G150 T220
    
M106 S255 ; turn on fan to cool the nozzle

;===== wipe left nozzle end =====

M400
;M73 P99



M400
;M73 P99

;===== bed leveling ==================================

M1002 judge_flag g29_before_print_flag

M190 S55; ensure bed temp
M109 S140 A
M106 S0 ; turn off fan , too noisy

G91
M73 P16 R9
G1 Z5 F1200
G90
G1 X175 Y160 F30000

M622 J1
    M1002 gcode_claim_action : 1
    G29.20 A3
    G29 A1 O X165 Y150 I20 J20 R 
    M400
    M500 ; save cali data
M623
    
M622 J2
    M1002 gcode_claim_action : 1
    
        G29.20 A4
        G29 A2 O X165 Y150 I20 J20 R
    
    M400
    M500 ; save cali data
M623

M622 J0
    G28 R
M623

;===== bed leveling end ================================

;===== z ofst cali start =====

    M190 S55; ensure bed temp

    G383 O0 M2 T140
    M500

;===== z ofst cali end =====

G39.1 ; cali nozzle wrapped detection pos
M500

G90
M73 P54 R5
G1 Z5 F1200
G1 X270 Y-0.5 F60000
G28.140 S0 ; cali pre-extrude z pos

M141 S0
M104 S220 A

;===== mech mode sweep start =====
    M1002 gcode_claim_action : 3

    G90
    G1 Z5 F1200
    G1 X187 Y160 F20000
    T1000
    M400 P200

    M970.3 Q1 A5 K0 O1
    M974 Q1 S2 P0

    M970.3 Q0 A5 K0 O1
    M974 Q0 S2 P0

    M970.2 Q2 K0 W38 Z0.01
    M974 Q2 S2 P0
    M500

    M975 S1
;===== mech mode sweep end =====

M400
;M73 P99

G150.3 ; move to garbage can to wait for temp
M1026
G29.9

;===== xy ofst cali start =====

M1002 judge_flag auto_cali_toolhead_offset_flag

M622 J0
    M1012.5 N1 R1
    M500
M623

M622 J1
    M1002 gcode_claim_action : 39
    M141 S0
    M620.17 T0 S220 L1
    M620.17 T1 S220 L-1
    G383 O1 T220 L1
    M500
    M141 S0
M623

M622 J2
    M1002 gcode_claim_action : 39
    M141 S0
    M620.17 T0 S220 L1
    M620.17 T1 S220 L-1
    G383.3 T220 L1
    M500
    M141 S0
M623
;===== xy ofst cali end =====

M400
;M73 P99

M1002 gcode_claim_action : 0
M400

;============switch again==================

M211 X0 Y0 Z0 ;turn off soft endstop
G91
G1 Z6 F1200
G90
M1002 set_filament_type:PLA
M620 S1A
M400
T1
M400
M628 S0
M629
M400
M621 S1A

;============switch again==================

M400
;M73 P99

;===== wait temperature reaching the reference value =======

M104 S220 ; rise to print tmpr

M140 S55 
M190 S55 

    ;========turn off light and fans =============
    M960 S1 P0 ; turn off laser
    M960 S2 P0 ; turn off laser
    M106 S0 ; turn off fan
    M106 P2 S0 ; turn off big fan
    ;==== set ext toodhead cooling fan ==== 
    
    M106 P9 S255
    
    ;============set motor current==================
    M400 S1

;===== wait temperature reaching the reference value =======

M400
;M73 P99

;===== for Textured PEI Plate , lower the nozzle as the nozzle was touching topmost of the texture when homing ==
    
        
            G29.1 Z-0.02 ; for Textured PEI Plate
        
    
    
G150.1

M975 S1 ; turn on mech mode supression
M983.4 S1 ; turn on deformation compensation 
G29.2 S1 ; turn on pos comp
G29.7 S1

G90
G1 Z5 F1200
G1 Y295 F30000
G1 Y265 F18000

;===== nozzle load line ===============================
    G29.2 S1 ; ensure z comp turn on
    G90
    M83
    G1 Z5 F1200
    G1 X270 Y-0.5 F60000
    G28.14 R0
    G29.2 S0
    G91
    G1 Z0.8 F1200
    G90
    G1 X250 F60000
    M109 S220
    M83

    G1 E5 F623.623
M73 P55 R5
    G1 X290 E10 F623.623
    G91
    G3 Z0.4 I1.217 J0 P1 F60000
    G90
    M83
    G29.2 S1 ; ensure z comp turn on
;===== noozle load line end ===========================

M400
;M73 P99

M993 A1 B1 C1 ; nozzle cam detection allowed.


M1015.3 S0;disable tpu clog detect



M1015.4 S1 K1 H0.4 ;enable E air printing detect


M620.6 I1 W1 ;enable ams air printing detect

M211 Z1
G29.99


; MACHINE_START_GCODE_END
M104 T1 S0 N0 ;Multi extruder pre cooling
; filament start gcode
;VT1
G90
G21
M83 ; use relative distances for extrusion
M981 S1 P20000 ;open spaghetti detector
; CHANGE_LAYER
; Z_HEIGHT: 0.2
; LAYER_HEIGHT: 0.2
G1 E-.4 F1800
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 1/50
; update layer progress
M73 L1
M991 S0 P0 ;notify layer change

M106 S0
M106 P2 S0
; OBJECT_ID: 132
G1 X184.143 Y169.143 F60000
M204 S6000
G1 Z.4
G1 Z.2
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.5
G1 F3000
M204 S500
G1 X165.857 Y169.143 E.68108
G1 X165.857 Y150.857 E.68108
G1 X184.143 Y150.857 E.68108
G1 X184.143 Y169.083 E.67884
M204 S6000
G1 X184.6 Y169.6 F60000
; FEATURE: Outer wall
G1 F3000
M204 S500
G1 X165.4 Y169.6 E.71513
G1 X165.4 Y150.4 E.71513
G1 X184.6 Y150.4 E.71513
G1 X184.6 Y169.54 E.71289
; WIPE_START
M73 P56 R5
G1 X183.6 Y169.543 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S6000
G17
G3 Z.6 I1.216 J-.038 P1  F60000
G1 X183.026 Y151.04 Z.6
G1 Z.2
G1 E.4 F1800
; FEATURE: Bottom surface
; LINE_WIDTH: 0.50487
G1 F6300
M204 S500
G1 X183.754 Y151.769 E.03879
G1 X183.754 Y152.422 E.02459
G1 X182.578 Y151.246 E.06262
G1 X181.925 Y151.246 E.02459
G1 X183.754 Y153.075 E.0974
G1 X183.754 Y153.728 E.02459
G1 X181.272 Y151.246 E.13217
G1 X180.618 Y151.246 E.02459
G1 X183.754 Y154.382 E.16695
G1 X183.754 Y155.035 E.02459
G1 X179.965 Y151.246 E.20173
G1 X179.312 Y151.246 E.02459
G1 X183.754 Y155.688 E.23651
G1 X183.754 Y156.342 E.02459
G1 X178.658 Y151.246 E.27129
G1 X178.005 Y151.246 E.02459
G1 X183.754 Y156.995 E.30607
G1 X183.754 Y157.648 E.02459
G1 X177.352 Y151.246 E.34084
G1 X176.699 Y151.246 E.02459
M73 P56 R4
G1 X183.754 Y158.301 E.37562
G1 X183.754 Y158.955 E.02459
G1 X176.045 Y151.246 E.4104
G1 X175.392 Y151.246 E.02459
G1 X183.754 Y159.608 E.44518
G1 X183.754 Y160.261 E.02459
G1 X174.739 Y151.246 E.47996
G1 X174.085 Y151.246 E.02459
G1 X183.754 Y160.915 E.51474
G1 X183.754 Y161.568 E.02459
G1 X173.432 Y151.246 E.54951
G1 X172.779 Y151.246 E.02459
G1 X183.754 Y162.221 E.58429
G1 X183.754 Y162.875 E.02459
G1 X172.125 Y151.246 E.61907
G1 X171.472 Y151.246 E.02459
G1 X183.754 Y163.528 E.65385
G1 X183.754 Y164.181 E.02459
G1 X170.819 Y151.246 E.68863
G1 X170.166 Y151.246 E.02459
M73 P57 R4
G1 X183.754 Y164.834 E.72341
G1 X183.754 Y165.488 E.02459
G1 X169.512 Y151.246 E.75819
G1 X168.859 Y151.246 E.02459
G1 X183.754 Y166.141 E.79296
G1 X183.754 Y166.794 E.02459
G1 X168.206 Y151.246 E.82774
G1 X167.552 Y151.246 E.02459
G1 X183.754 Y167.448 E.86252
G1 X183.754 Y168.101 E.02459
G1 X166.899 Y151.246 E.8973
G1 X166.246 Y151.246 E.02459
G1 X183.754 Y168.754 E.93208
G1 X183.101 Y168.754 E.02458
G1 X166.246 Y151.899 E.89732
G1 X166.246 Y152.552 E.02459
G1 X182.448 Y168.754 E.86254
G1 X181.795 Y168.754 E.02459
G1 X166.246 Y153.205 E.82776
G1 X166.246 Y153.859 E.02459
G1 X181.141 Y168.754 E.79299
G1 X180.488 Y168.754 E.02459
G1 X166.246 Y154.512 E.75821
G1 X166.246 Y155.165 E.02459
G1 X179.835 Y168.754 E.72343
G1 X179.182 Y168.754 E.02459
G1 X166.246 Y155.818 E.68865
G1 X166.246 Y156.472 E.02459
G1 X178.528 Y168.754 E.65387
G1 X177.875 Y168.754 E.02459
G1 X166.246 Y157.125 E.61909
G1 X166.246 Y157.778 E.02459
M73 P58 R4
G1 X177.222 Y168.754 E.58432
G1 X176.568 Y168.754 E.02459
G1 X166.246 Y158.432 E.54954
G1 X166.246 Y159.085 E.02459
G1 X175.915 Y168.754 E.51476
G1 X175.262 Y168.754 E.02459
G1 X166.246 Y159.738 E.47998
G1 X166.246 Y160.392 E.02459
G1 X174.608 Y168.754 E.4452
G1 X173.955 Y168.754 E.02459
G1 X166.246 Y161.045 E.41042
G1 X166.246 Y161.698 E.02459
G1 X173.302 Y168.754 E.37565
G1 X172.649 Y168.754 E.02459
G1 X166.246 Y162.351 E.34087
G1 X166.246 Y163.005 E.02459
G1 X171.995 Y168.754 E.30609
G1 X171.342 Y168.754 E.02459
G1 X166.246 Y163.658 E.27131
G1 X166.246 Y164.311 E.02459
G1 X170.689 Y168.754 E.23653
G1 X170.035 Y168.754 E.02459
G1 X166.246 Y164.965 E.20175
G1 X166.246 Y165.618 E.02459
G1 X169.382 Y168.754 E.16697
G1 X168.729 Y168.754 E.02459
G1 X166.246 Y166.271 E.1322
G1 X166.246 Y166.924 E.02459
G1 X168.076 Y168.754 E.09742
G1 X167.422 Y168.754 E.02459
G1 X166.246 Y167.578 E.06264
G1 X166.246 Y168.231 E.02459
G1 X166.975 Y168.96 E.03881
; WIPE_START
G1 X166.268 Y168.253 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S6000
G17
G3 Z.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z0.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z0.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 0.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 2/50
; update layer progress
M73 L2
M991 S0 P1 ;notify layer change

M106 S201.45
M106 P2 S191
; open powerlost recovery
M1003 S1
; OBJECT_ID: 132
M204 S10000
G1 X184.398 Y169.398 F60000
G1 Z.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F12000
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z.8 I1.022 J-.661 P1  F60000
G1 X183.467 Y169.234 Z.8
G1 Z.4
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.42222
G1 F15000
M204 S8000
G1 X184.065 Y168.637 E.02611
G1 X184.065 Y168.101 E.01658
G1 X183.101 Y169.065 E.04214
G1 X182.564 Y169.065 E.01658
G1 X184.065 Y167.564 E.06559
G1 X184.065 Y167.028 E.01658
G1 X182.028 Y169.065 E.08903
G1 X181.491 Y169.065 E.01658
G1 X184.065 Y166.491 E.11248
G1 X184.065 Y165.955 E.01658
G1 X180.955 Y169.065 E.13593
G1 X180.419 Y169.065 E.01658
G1 X184.065 Y165.419 E.15937
G1 X184.065 Y164.882 E.01658
G1 X179.882 Y169.065 E.18282
G1 X179.346 Y169.065 E.01658
G1 X184.065 Y164.346 E.20627
G1 X184.065 Y163.809 E.01658
G1 X178.809 Y169.065 E.22971
G1 X178.273 Y169.065 E.01658
G1 X184.065 Y163.273 E.25316
G1 X184.065 Y162.737 E.01658
G1 X177.737 Y169.065 E.27661
G1 X177.2 Y169.065 E.01658
M73 P59 R4
G1 X184.065 Y162.2 E.30006
G1 X184.065 Y161.664 E.01658
G1 X176.664 Y169.065 E.3235
G1 X176.127 Y169.065 E.01658
G1 X184.065 Y161.127 E.34695
G1 X184.065 Y160.591 E.01658
G1 X175.591 Y169.065 E.3704
G1 X175.054 Y169.065 E.01658
G1 X184.065 Y160.054 E.39384
G1 X184.065 Y159.518 E.01658
G1 X174.518 Y169.065 E.41729
G1 X173.982 Y169.065 E.01658
G1 X184.065 Y158.982 E.44074
G1 X184.065 Y158.445 E.01658
G1 X173.445 Y169.065 E.46418
G1 X172.909 Y169.065 E.01658
G1 X184.065 Y157.909 E.48763
G1 X184.065 Y157.372 E.01658
G1 X172.372 Y169.065 E.51108
G1 X171.836 Y169.065 E.01658
G1 X184.065 Y156.836 E.53452
G1 X184.065 Y156.3 E.01658
G1 X171.3 Y169.065 E.55797
G1 X170.763 Y169.065 E.01658
G1 X184.065 Y155.763 E.58142
G1 X184.065 Y155.227 E.01658
G1 X170.227 Y169.065 E.60486
G1 X169.69 Y169.065 E.01658
G1 X184.065 Y154.69 E.62831
G1 X184.065 Y154.154 E.01658
G1 X169.154 Y169.065 E.65176
G1 X168.618 Y169.065 E.01658
G1 X184.065 Y153.618 E.67521
G1 X184.065 Y153.081 E.01658
G1 X168.081 Y169.065 E.69865
G1 X167.545 Y169.065 E.01658
G1 X184.065 Y152.545 E.7221
G1 X184.065 Y152.008 E.01658
G1 X167.008 Y169.065 E.74555
G1 X166.472 Y169.065 E.01658
G1 X184.065 Y151.472 E.76899
G1 X184.065 Y150.935 E.01658
G1 X165.935 Y169.065 E.79244
G1 X165.935 Y168.528 E.01657
G1 X183.528 Y150.935 E.76901
G1 X182.992 Y150.935 E.01658
G1 X165.935 Y167.992 E.74556
G1 X165.935 Y167.456 E.01658
G1 X182.456 Y150.935 E.72211
G1 X181.919 Y150.935 E.01658
G1 X165.935 Y166.919 E.69867
G1 X165.935 Y166.383 E.01658
G1 X181.383 Y150.935 E.67522
G1 X180.846 Y150.935 E.01658
G1 X165.935 Y165.846 E.65177
G1 X165.935 Y165.31 E.01658
G1 X180.31 Y150.935 E.62833
G1 X179.774 Y150.935 E.01658
G1 X165.935 Y164.774 E.60488
G1 X165.935 Y164.237 E.01658
G1 X179.237 Y150.935 E.58143
G1 X178.701 Y150.935 E.01658
G1 X165.935 Y163.701 E.55799
G1 X165.935 Y163.164 E.01658
G1 X178.164 Y150.935 E.53454
G1 X177.628 Y150.935 E.01658
G1 X165.935 Y162.628 E.51109
G1 X165.935 Y162.092 E.01658
G1 X177.092 Y150.935 E.48765
G1 X176.555 Y150.935 E.01658
G1 X165.935 Y161.555 E.4642
G1 X165.935 Y161.019 E.01658
G1 X176.019 Y150.935 E.44075
G1 X175.482 Y150.935 E.01658
G1 X165.935 Y160.482 E.4173
G1 X165.935 Y159.946 E.01658
G1 X174.946 Y150.935 E.39386
G1 X174.409 Y150.935 E.01658
G1 X165.935 Y159.409 E.37041
G1 X165.935 Y158.873 E.01658
G1 X173.873 Y150.935 E.34696
G1 X173.337 Y150.935 E.01658
G1 X165.935 Y158.337 E.32352
G1 X165.935 Y157.8 E.01658
G1 X172.8 Y150.935 E.30007
G1 X172.264 Y150.935 E.01658
G1 X165.935 Y157.264 E.27662
G1 X165.935 Y156.727 E.01658
G1 X171.727 Y150.935 E.25318
G1 X171.191 Y150.935 E.01658
G1 X165.935 Y156.191 E.22973
G1 X165.935 Y155.655 E.01658
G1 X170.655 Y150.935 E.20628
G1 X170.118 Y150.935 E.01658
G1 X165.935 Y155.118 E.18284
G1 X165.935 Y154.582 E.01658
G1 X169.582 Y150.935 E.15939
G1 X169.045 Y150.935 E.01658
G1 X165.935 Y154.045 E.13594
G1 X165.935 Y153.509 E.01658
G1 X168.509 Y150.935 E.11249
G1 X167.973 Y150.935 E.01658
G1 X165.935 Y152.973 E.08905
G1 X165.935 Y152.436 E.01658
G1 X167.436 Y150.935 E.0656
G1 X166.9 Y150.935 E.01658
G1 X165.935 Y151.9 E.04215
G1 X165.935 Y151.363 E.01658
G1 X166.533 Y150.766 E.02612
; WIPE_START
G1 X165.935 Y151.363 E-.32117
G1 X165.935 Y151.518 E-.05883
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z0.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z0.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 0.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 3/50
; update layer progress
M73 L3
M991 S0 P2 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4753
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
M73 P60 R4
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1 I1.217 J.03 P1  F60000
G1 X184.234 Y151.533 Z1
G1 Z.6
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.42222
G1 F15000
M204 S8000
G1 X183.637 Y150.935 E.02611
G1 X183.101 Y150.935 E.01658
G1 X184.065 Y151.899 E.04214
G1 X184.065 Y152.436 E.01658
G1 X182.564 Y150.935 E.06559
G1 X182.028 Y150.935 E.01658
G1 X184.065 Y152.972 E.08903
G1 X184.065 Y153.509 E.01658
G1 X181.491 Y150.935 E.11248
G1 X180.955 Y150.935 E.01658
G1 X184.065 Y154.045 E.13593
G1 X184.065 Y154.581 E.01658
G1 X180.419 Y150.935 E.15937
G1 X179.882 Y150.935 E.01658
G1 X184.065 Y155.118 E.18282
G1 X184.065 Y155.654 E.01658
G1 X179.346 Y150.935 E.20627
G1 X178.809 Y150.935 E.01658
G1 X184.065 Y156.191 E.22971
G1 X184.065 Y156.727 E.01658
G1 X178.273 Y150.935 E.25316
G1 X177.737 Y150.935 E.01658
G1 X184.065 Y157.263 E.27661
G1 X184.065 Y157.8 E.01658
G1 X177.2 Y150.935 E.30005
G1 X176.664 Y150.935 E.01658
G1 X184.065 Y158.336 E.3235
G1 X184.065 Y158.873 E.01658
G1 X176.127 Y150.935 E.34695
G1 X175.591 Y150.935 E.01658
G1 X184.065 Y159.409 E.3704
G1 X184.065 Y159.946 E.01658
G1 X175.054 Y150.935 E.39384
G1 X174.518 Y150.935 E.01658
G1 X184.065 Y160.482 E.41729
G1 X184.065 Y161.018 E.01658
G1 X173.982 Y150.935 E.44074
G1 X173.445 Y150.935 E.01658
G1 X184.065 Y161.555 E.46418
G1 X184.065 Y162.091 E.01658
G1 X172.909 Y150.935 E.48763
G1 X172.372 Y150.935 E.01658
G1 X184.065 Y162.628 E.51108
G1 X184.065 Y163.164 E.01658
G1 X171.836 Y150.935 E.53452
G1 X171.3 Y150.935 E.01658
G1 X184.065 Y163.7 E.55797
G1 X184.065 Y164.237 E.01658
G1 X170.763 Y150.935 E.58142
G1 X170.227 Y150.935 E.01658
G1 X184.065 Y164.773 E.60486
G1 X184.065 Y165.31 E.01658
G1 X169.69 Y150.935 E.62831
G1 X169.154 Y150.935 E.01658
G1 X184.065 Y165.846 E.65176
G1 X184.065 Y166.382 E.01658
G1 X168.618 Y150.935 E.67521
G1 X168.081 Y150.935 E.01658
G1 X184.065 Y166.919 E.69865
G1 X184.065 Y167.455 E.01658
G1 X167.545 Y150.935 E.7221
G1 X167.008 Y150.935 E.01658
G1 X184.065 Y167.992 E.74555
G1 X184.065 Y168.528 E.01658
G1 X166.472 Y150.935 E.76899
G1 X165.935 Y150.935 E.01658
G1 X184.065 Y169.065 E.79244
G1 X183.528 Y169.065 E.01657
G1 X165.935 Y151.472 E.76901
G1 X165.935 Y152.008 E.01658
G1 X182.992 Y169.065 E.74556
G1 X182.456 Y169.065 E.01658
G1 X165.935 Y152.544 E.72211
G1 X165.935 Y153.081 E.01658
G1 X181.919 Y169.065 E.69867
G1 X181.383 Y169.065 E.01658
G1 X165.935 Y153.617 E.67522
G1 X165.935 Y154.154 E.01658
G1 X180.846 Y169.065 E.65177
G1 X180.31 Y169.065 E.01658
G1 X165.935 Y154.69 E.62833
G1 X165.935 Y155.226 E.01658
G1 X179.774 Y169.065 E.60488
G1 X179.237 Y169.065 E.01658
G1 X165.935 Y155.763 E.58143
G1 X165.935 Y156.299 E.01658
G1 X178.701 Y169.065 E.55799
G1 X178.164 Y169.065 E.01658
G1 X165.935 Y156.836 E.53454
G1 X165.935 Y157.372 E.01658
G1 X177.628 Y169.065 E.51109
G1 X177.092 Y169.065 E.01658
G1 X165.935 Y157.908 E.48765
G1 X165.935 Y158.445 E.01658
G1 X176.555 Y169.065 E.4642
G1 X176.019 Y169.065 E.01658
G1 X165.935 Y158.981 E.44075
G1 X165.935 Y159.518 E.01658
G1 X175.482 Y169.065 E.4173
G1 X174.946 Y169.065 E.01658
G1 X165.935 Y160.054 E.39386
G1 X165.935 Y160.591 E.01658
G1 X174.409 Y169.065 E.37041
G1 X173.873 Y169.065 E.01658
G1 X165.935 Y161.127 E.34696
G1 X165.935 Y161.663 E.01658
G1 X173.337 Y169.065 E.32352
G1 X172.8 Y169.065 E.01658
G1 X165.935 Y162.2 E.30007
G1 X165.935 Y162.736 E.01658
G1 X172.264 Y169.065 E.27662
G1 X171.727 Y169.065 E.01658
G1 X165.935 Y163.273 E.25318
G1 X165.935 Y163.809 E.01658
G1 X171.191 Y169.065 E.22973
G1 X170.655 Y169.065 E.01658
G1 X165.935 Y164.345 E.20628
G1 X165.935 Y164.882 E.01658
G1 X170.118 Y169.065 E.18283
G1 X169.582 Y169.065 E.01658
G1 X165.935 Y165.418 E.15939
G1 X165.935 Y165.955 E.01658
G1 X169.045 Y169.065 E.13594
G1 X168.509 Y169.065 E.01658
G1 X165.935 Y166.491 E.11249
G1 X165.935 Y167.027 E.01658
G1 X167.973 Y169.065 E.08905
G1 X167.436 Y169.065 E.01658
G1 X165.935 Y167.564 E.0656
G1 X165.935 Y168.1 E.01658
M73 P61 R4
G1 X166.9 Y169.065 E.04215
G1 X166.363 Y169.065 E.01658
G1 X165.766 Y168.467 E.02612
; WIPE_START
G1 X166.363 Y169.065 E-.32117
G1 X166.518 Y169.065 E-.05883
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z1 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z1 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 0.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 4/50
; update layer progress
M73 L4
M991 S0 P3 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F4763
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4752
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z1.2
G1 Z.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4763
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4763
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4763
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z1.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z1.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 1
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 5/50
; update layer progress
M73 L5
M991 S0 P4 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z1
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4752
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
M73 P62 R4
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.4 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z1.4
G1 Z1
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z1.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z1.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 1.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 6/50
; update layer progress
M73 L6
M991 S0 P5 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z1.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.6 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z1.6
G1 Z1.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
M73 P63 R4
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z1.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z1.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 1.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 7/50
; update layer progress
M73 L7
M991 S0 P6 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z1.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.8 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z1.8
G1 Z1.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z1.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z1.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z1.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 1.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 8/50
; update layer progress
M73 L8
M991 S0 P7 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z1.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
M73 P64 R4
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z2
G1 Z1.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 1.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 9/50
; update layer progress
M73 L9
M991 S0 P8 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z1.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.2 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z2.2
G1 Z1.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
M73 P65 R4
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z2.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z2.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 10/50
; update layer progress
M73 L10
M991 S0 P9 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
M73 P65 R3
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.4 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z2.4
G1 Z2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z2.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z2.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 2.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 11/50
; update layer progress
M73 L11
M991 S0 P10 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z2.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
M73 P66 R3
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.6 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z2.6
G1 Z2.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z2.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z2.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 2.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 12/50
; update layer progress
M73 L12
M991 S0 P11 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z2.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.8 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z2.8
G1 Z2.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
M73 P67 R3
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z2.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z2.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z2.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 2.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 13/50
; update layer progress
M73 L13
M991 S0 P12 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z2.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z3
G1 Z2.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z3 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z3 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 2.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 14/50
; update layer progress
M73 L14
M991 S0 P13 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z2.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
M73 P68 R3
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z3.2
G1 Z2.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z3.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z3.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 3
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 15/50
; update layer progress
M73 L15
M991 S0 P14 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z3
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.4 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z3.4
G1 Z3
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
M73 P69 R3
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z3.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z3.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 3.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 16/50
; update layer progress
M73 L16
M991 S0 P15 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z3.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.6 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z3.6
G1 Z3.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z3.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z3.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 3.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 17/50
; update layer progress
M73 L17
M991 S0 P16 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z3.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
M73 P70 R3
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.8 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z3.8
G1 Z3.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z3.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z3.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z3.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 3.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 18/50
; update layer progress
M73 L18
M991 S0 P17 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z3.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z4
G1 Z3.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
M73 P71 R3
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 3.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 19/50
; update layer progress
M73 L19
M991 S0 P18 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z3.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.2 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z4.2
G1 Z3.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z4.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z4.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 20/50
; update layer progress
M73 L20
M991 S0 P19 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z4
G1 E.4 F1800
; FEATURE: Inner wall
M73 P72 R3
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.4 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z4.4
G1 Z4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z4.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z4.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 4.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 21/50
; update layer progress
M73 L21
M991 S0 P20 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z4.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
M73 P73 R3
G3 Z4.6 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z4.6
G1 Z4.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z4.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z4.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 4.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 22/50
; update layer progress
M73 L22
M991 S0 P21 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z4.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.8 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z4.8
G1 Z4.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
M73 P73 R2
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
M73 P74 R2
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z4.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z4.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z4.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 4.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 23/50
; update layer progress
M73 L23
M991 S0 P22 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z4.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z5
G1 Z4.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z5 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z5 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 4.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 24/50
; update layer progress
M73 L24
M991 S0 P23 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z4.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
M73 P75 R2
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z5.2
G1 Z4.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z5.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z5.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 5
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 25/50
; update layer progress
M73 L25
M991 S0 P24 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z5
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.4 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z5.4
G1 Z5
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
M73 P76 R2
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z5.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z5.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 5.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 26/50
; update layer progress
M73 L26
M991 S0 P25 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z5.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.6 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z5.6
G1 Z5.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z5.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z5.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 5.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 27/50
; update layer progress
M73 L27
M991 S0 P26 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z5.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
M73 P77 R2
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.8 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z5.8
G1 Z5.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z5.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z5.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z5.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 5.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 28/50
; update layer progress
M73 L28
M991 S0 P27 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z5.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z6
G1 Z5.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
M73 P78 R2
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 5.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 29/50
; update layer progress
M73 L29
M991 S0 P28 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z5.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.2 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z6.2
G1 Z5.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z6.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z6.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 30/50
; update layer progress
M73 L30
M991 S0 P29 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
M73 P79 R2
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.4 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z6.4
G1 Z6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z6.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z6.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 6.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 31/50
; update layer progress
M73 L31
M991 S0 P30 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z6.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.6 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z6.6
G1 Z6.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
M73 P80 R2
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z6.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z6.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 6.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 32/50
; update layer progress
M73 L32
M991 S0 P31 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z6.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.8 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z6.8
G1 Z6.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z6.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z6.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z6.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 6.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 33/50
; update layer progress
M73 L33
M991 S0 P32 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z6.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
M73 P81 R2
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z7
G1 Z6.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z7 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z7 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 6.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 34/50
; update layer progress
M73 L34
M991 S0 P33 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z6.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z7.2
G1 Z6.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
M73 P82 R2
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z7.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z7.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 7
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 35/50
; update layer progress
M73 L35
M991 S0 P34 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z7
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.4 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z7.4
G1 Z7
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
M73 P82 R1
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z7.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z7.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 7.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 36/50
; update layer progress
M73 L36
M991 S0 P35 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z7.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
M73 P83 R1
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.6 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z7.6
G1 Z7.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z7.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z7.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 7.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 37/50
; update layer progress
M73 L37
M991 S0 P36 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z7.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4752
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.8 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z7.8
G1 Z7.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
M73 P84 R1
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z7.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z7.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z7.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 7.6
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 38/50
; update layer progress
M73 L38
M991 S0 P37 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z7.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z8
G1 Z7.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
M73 P85 R1
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 7.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 39/50
; update layer progress
M73 L39
M991 S0 P38 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z7.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4754
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.2 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z8.2
G1 Z7.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z8.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z8.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 40/50
; update layer progress
M73 L40
M991 S0 P39 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
M73 P86 R1
G3 Z8.4 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z8.4
G1 Z8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z8.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z8.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 8.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 41/50
; update layer progress
M73 L41
M991 S0 P40 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z8.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4768
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.6 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z8.6
G1 Z8.2
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
M73 P87 R1
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z8.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z8.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 8.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 42/50
; update layer progress
M73 L42
M991 S0 P41 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z8.4
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.8 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z8.8
G1 Z8.4
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z8.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z8.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z8.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 8.6
; LAYER_HEIGHT: 0.200001
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 43/50
; update layer progress
M73 L43
M991 S0 P42 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z8.6
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4768
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4768
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
M73 P88 R1
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9 I1.215 J.062 P1  F60000
G1 X184.05 Y164.674 Z9
G1 Z8.6
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4768
M204 S8000
G1 X184.05 Y166.302 E.05401
G1 X181.302 Y169.05 E.12889
G1 X184.05 Y169.05 E.09114
G1 X165.95 Y150.95 E.84908
G1 X168.698 Y150.95 E.09114
G1 X165.95 Y153.698 E.12889
G1 X165.95 Y155.326 E.05401
M204 S10000
G1 X165.95 Y164.674 F60000
G1 F4768
M204 S8000
G1 X165.95 Y166.302 E.05401
G1 X168.698 Y169.05 E.12889
G1 X165.95 Y169.05 E.09114
G1 X184.05 Y150.95 E.84908
G1 X181.302 Y150.95 E.09114
G1 X184.05 Y153.698 E.12889
G1 X184.05 Y155.326 E.05401
M204 S10000
G1 X171.998 Y169.05 F60000
G1 F4768
M204 S8000
G1 X173.626 Y169.05 E.05401
G1 X184.05 Y158.626 E.48898
G1 X184.05 Y161.374 E.09114
G1 X173.626 Y150.95 E.48898
G1 X176.374 Y150.95 E.09114
G1 X165.95 Y161.374 E.48898
G1 X165.95 Y158.626 E.09114
G1 X176.374 Y169.05 E.48898
G1 X178.002 Y169.05 E.05401
; WIPE_START
G1 F18423.913
G1 X177.002 Y169.05 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z9 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z9 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 8.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 44/50
; update layer progress
M73 L44
M991 S0 P43 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z8.8
G1 E.4 F1800
; FEATURE: Inner wall
G1 F4751
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F4751
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.2 I.143 J-1.209 P1  F60000
G1 X178.002 Y169.05 Z9.2
G1 Z8.8
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F4751
M204 S8000
G1 X176.374 Y169.05 E.05401
G1 X165.95 Y158.626 E.48898
G1 X165.95 Y161.374 E.09114
G1 X176.374 Y150.95 E.48898
G1 X173.626 Y150.95 E.09114
G1 X184.05 Y161.374 E.48898
G1 X184.05 Y158.626 E.09114
G1 X173.626 Y169.05 E.48898
G1 X171.998 Y169.05 E.05401
M204 S10000
G1 X165.95 Y155.326 F60000
G1 F4751
M204 S8000
G1 X165.95 Y153.698 E.05401
G1 X168.698 Y150.95 E.12889
G1 X165.95 Y150.95 E.09114
G1 X184.05 Y169.05 E.84908
G1 X181.302 Y169.05 E.09114
G1 X184.05 Y166.302 E.12889
G1 X184.05 Y164.674 E.05401
M204 S10000
M73 P89 R1
G1 X184.05 Y155.326 F60000
G1 F4751
M204 S8000
G1 X184.05 Y153.698 E.05401
G1 X181.302 Y150.95 E.12889
G1 X184.05 Y150.95 E.09114
G1 X165.95 Y169.05 E.84908
G1 X168.698 Y169.05 E.09114
G1 X165.95 Y166.302 E.12889
G1 X165.95 Y164.674 E.05401
; WIPE_START
G1 F18423.913
G1 X165.95 Y165.674 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z9.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z9.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 9
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 45/50
; update layer progress
M73 L45
M991 S0 P44 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z9
G1 E.4 F1800
; FEATURE: Inner wall
G1 F7198
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5041
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.4 I1.217 J-.031 P1  F60000
G1 X183.673 Y165.051 Z9.4
G1 Z9
G1 E.4 F1800
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F7198
M204 S8000
G1 X183.673 Y166.679 E.05401
G1 X181.679 Y168.673 E.0935
G1 X183.673 Y168.673 E.06612
G1 X166.327 Y151.327 E.81369
G1 X168.321 Y151.327 E.06612
G1 X166.327 Y153.321 E.0935
G1 X166.327 Y154.949 E.05401
M204 S10000
G1 X166.327 Y165.051 F60000
G1 F7198
M204 S8000
G1 X166.327 Y166.679 E.05401
G1 X168.321 Y168.673 E.0935
G1 X166.327 Y168.673 E.06612
G1 X183.673 Y151.327 E.81369
G1 X181.679 Y151.327 E.06612
G1 X183.673 Y153.321 E.0935
G1 X183.673 Y154.949 E.05401
M204 S10000
G1 X184.008 Y150.992 F60000
; Slow Down Start
; FEATURE: Floating vertical shell
; LINE_WIDTH: 0.399323
G1 F3000;_EXTRUDE_SET_SPEED
M204 S8000
G1 X184.035 Y151.124 E.00391
G1 X184.035 Y168.876 E.51556
G1 X184.008 Y169.008 E.00391
G1 X183.876 Y169.035 E.00391
G1 X166.124 Y169.035 E.51556
G1 X165.992 Y169.008 E.00391
G1 X165.965 Y168.876 E.00391
G1 X165.965 Y151.124 E.51556
G1 X165.992 Y150.992 E.00391
G1 X166.124 Y150.965 E.00391
G1 X183.876 Y150.965 E.51556
G1 X183.949 Y150.98 E.00217
; Slow Down End
M204 S10000
G1 X172.375 Y168.673 F60000
; FEATURE: Sparse infill
; LINE_WIDTH: 0.45
G1 F7198
M204 S8000
G1 X174.003 Y168.673 E.05401
G1 X183.673 Y159.003 E.4536
G1 X183.673 Y160.997 E.06611
G1 X174.003 Y151.327 E.4536
G1 X175.997 Y151.327 E.06612
G1 X166.327 Y160.997 E.4536
G1 X166.327 Y159.003 E.06611
G1 X175.997 Y168.673 E.4536
G1 X177.625 Y168.673 E.05401
; WIPE_START
G1 F18423.913
G1 X176.625 Y168.673 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z9.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z9.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

M106 S255
; CHANGE_LAYER
; Z_HEIGHT: 9.2
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 46/50
; update layer progress
M73 L46
M991 S0 P45 ;notify layer change

M106 S255
; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z9.2
G1 E.4 F1800
; FEATURE: Inner wall
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5158
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
M73 P90 R1
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.6 I.887 J-.833 P1  F60000
G1 X183.318 Y169.231 Z9.6
G1 Z9.2
G1 E.4 F1800
; FEATURE: Bridge
; LINE_WIDTH: 0.40124
; LAYER_HEIGHT: 0.4
G1 F3000
M204 S8000
G1 X184.028 Y168.521 E.05171
G1 X184.028 Y167.883 E.03288
G1 X182.883 Y169.028 E.08345
G1 X182.245 Y169.028 E.03288
G1 X184.028 Y167.245 E.12995
G1 X184.028 Y166.606 E.03288
G1 X181.606 Y169.028 E.17644
G1 X180.968 Y169.028 E.03288
G1 X184.028 Y165.968 E.22294
G1 X184.028 Y165.33 E.03288
G1 X180.33 Y169.028 E.26943
G1 X179.692 Y169.028 E.03288
G1 X184.028 Y164.692 E.31592
G1 X184.028 Y164.054 E.03288
G1 X179.054 Y169.028 E.36242
G1 X178.416 Y169.028 E.03288
G1 X184.028 Y163.416 E.40891
G1 X184.028 Y162.778 E.03288
G1 X177.778 Y169.028 E.45541
G1 X177.139 Y169.028 E.03288
G1 X184.028 Y162.139 E.5019
G1 X184.028 Y161.501 E.03288
G1 X176.501 Y169.028 E.54839
G1 X175.863 Y169.028 E.03288
G1 X184.028 Y160.863 E.59489
G1 X184.028 Y160.225 E.03288
G1 X175.225 Y169.028 E.64138
G1 X174.587 Y169.028 E.03288
G1 X184.028 Y159.587 E.68787
G1 X184.028 Y158.949 E.03288
G1 X173.949 Y169.028 E.73437
G1 X173.311 Y169.028 E.03288
G1 X184.028 Y158.311 E.78086
G1 X184.028 Y157.672 E.03288
G1 X172.672 Y169.028 E.82736
G1 X172.034 Y169.028 E.03288
G1 X184.028 Y157.034 E.87385
G1 X184.028 Y156.396 E.03288
G1 X171.396 Y169.028 E.92034
G1 X170.758 Y169.028 E.03288
G1 X184.028 Y155.758 E.96684
G1 X184.028 Y155.12 E.03288
G1 X170.12 Y169.028 E1.01333
G1 X169.482 Y169.028 E.03288
G1 X184.028 Y154.482 E1.05983
G1 X184.028 Y153.843 E.03288
G1 X168.843 Y169.028 E1.10632
G1 X168.205 Y169.028 E.03288
G1 X184.028 Y153.205 E1.15281
G1 X184.028 Y152.567 E.03288
G1 X167.567 Y169.028 E1.19931
G1 X166.929 Y169.028 E.03288
M73 P91 R1
G1 X184.028 Y151.929 E1.2458
G1 X184.028 Y151.291 E.03288
G1 X166.291 Y169.028 E1.29229
G1 X165.972 Y169.028 E.01644
G1 X165.972 Y168.709 E.01644
G1 X183.709 Y150.972 E1.2923
G1 X183.071 Y150.972 E.03288
G1 X165.972 Y168.071 E1.2458
G1 X165.972 Y167.433 E.03288
G1 X182.433 Y150.972 E1.19931
G1 X181.795 Y150.972 E.03288
M73 P91 R0
G1 X165.972 Y166.795 E1.15282
G1 X165.972 Y166.157 E.03288
G1 X181.157 Y150.972 E1.10632
G1 X180.518 Y150.972 E.03288
G1 X165.972 Y165.518 E1.05983
G1 X165.972 Y164.88 E.03288
G1 X179.88 Y150.972 E1.01333
G1 X179.242 Y150.972 E.03288
G1 X165.972 Y164.242 E.96684
G1 X165.972 Y163.604 E.03288
G1 X178.604 Y150.972 E.92035
G1 X177.966 Y150.972 E.03288
G1 X165.972 Y162.966 E.87385
G1 X165.972 Y162.328 E.03288
G1 X177.328 Y150.972 E.82736
G1 X176.69 Y150.972 E.03288
G1 X165.972 Y161.689 E.78086
G1 X165.972 Y161.051 E.03288
G1 X176.051 Y150.972 E.73437
G1 X175.413 Y150.972 E.03288
M73 P92 R0
G1 X165.972 Y160.413 E.68788
G1 X165.972 Y159.775 E.03288
G1 X174.775 Y150.972 E.64138
G1 X174.137 Y150.972 E.03288
G1 X165.972 Y159.137 E.59489
G1 X165.972 Y158.499 E.03288
G1 X173.499 Y150.972 E.54839
G1 X172.861 Y150.972 E.03288
G1 X165.972 Y157.861 E.5019
G1 X165.972 Y157.222 E.03288
G1 X172.222 Y150.972 E.45541
G1 X171.584 Y150.972 E.03288
G1 X165.972 Y156.584 E.40891
G1 X165.972 Y155.946 E.03288
G1 X170.946 Y150.972 E.36242
G1 X170.308 Y150.972 E.03288
G1 X165.972 Y155.308 E.31593
G1 X165.972 Y154.67 E.03288
G1 X169.67 Y150.972 E.26943
G1 X169.032 Y150.972 E.03288
G1 X165.972 Y154.032 E.22294
G1 X165.972 Y153.394 E.03288
G1 X168.394 Y150.972 E.17644
G1 X167.755 Y150.972 E.03288
G1 X165.972 Y152.755 E.12995
G1 X165.972 Y152.117 E.03288
G1 X167.117 Y150.972 E.08346
G1 X166.479 Y150.972 E.03288
G1 X165.769 Y151.682 E.05172
M106 S196.35
; WIPE_START
G1 X166.476 Y150.975 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.6 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z9.6 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z9.6 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 9.4
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 47/50
; update layer progress
M73 L47
M991 S0 P46 ;notify layer change

M106 S201.45
; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z9.4
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5270
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.8 I1.217 J.03 P1  F60000
G1 X184.234 Y151.533 Z9.8
G1 Z9.4
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.42222
G1 F15000
M204 S8000
G1 X183.637 Y150.935 E.02611
G1 X183.101 Y150.935 E.01658
G1 X184.065 Y151.899 E.04214
G1 X184.065 Y152.436 E.01658
G1 X182.564 Y150.935 E.06559
G1 X182.028 Y150.935 E.01658
G1 X184.065 Y152.972 E.08903
G1 X184.065 Y153.509 E.01658
G1 X181.491 Y150.935 E.11248
G1 X180.955 Y150.935 E.01658
G1 X184.065 Y154.045 E.13593
G1 X184.065 Y154.581 E.01658
G1 X180.419 Y150.935 E.15937
G1 X179.882 Y150.935 E.01658
G1 X184.065 Y155.118 E.18282
G1 X184.065 Y155.654 E.01658
G1 X179.346 Y150.935 E.20627
G1 X178.809 Y150.935 E.01658
G1 X184.065 Y156.191 E.22971
G1 X184.065 Y156.727 E.01658
G1 X178.273 Y150.935 E.25316
G1 X177.737 Y150.935 E.01658
G1 X184.065 Y157.263 E.27661
G1 X184.065 Y157.8 E.01658
G1 X177.2 Y150.935 E.30005
G1 X176.664 Y150.935 E.01658
G1 X184.065 Y158.336 E.3235
G1 X184.065 Y158.873 E.01658
G1 X176.127 Y150.935 E.34695
G1 X175.591 Y150.935 E.01658
G1 X184.065 Y159.409 E.3704
G1 X184.065 Y159.946 E.01658
G1 X175.054 Y150.935 E.39384
G1 X174.518 Y150.935 E.01658
G1 X184.065 Y160.482 E.41729
G1 X184.065 Y161.018 E.01658
G1 X173.982 Y150.935 E.44074
G1 X173.445 Y150.935 E.01658
G1 X184.065 Y161.555 E.46418
G1 X184.065 Y162.091 E.01658
G1 X172.909 Y150.935 E.48763
G1 X172.372 Y150.935 E.01658
G1 X184.065 Y162.628 E.51108
G1 X184.065 Y163.164 E.01658
G1 X171.836 Y150.935 E.53452
G1 X171.3 Y150.935 E.01658
G1 X184.065 Y163.7 E.55797
G1 X184.065 Y164.237 E.01658
G1 X170.763 Y150.935 E.58142
G1 X170.227 Y150.935 E.01658
G1 X184.065 Y164.773 E.60486
G1 X184.065 Y165.31 E.01658
G1 X169.69 Y150.935 E.62831
G1 X169.154 Y150.935 E.01658
G1 X184.065 Y165.846 E.65176
G1 X184.065 Y166.382 E.01658
M73 P93 R0
G1 X168.618 Y150.935 E.67521
G1 X168.081 Y150.935 E.01658
G1 X184.065 Y166.919 E.69865
G1 X184.065 Y167.455 E.01658
G1 X167.545 Y150.935 E.7221
G1 X167.008 Y150.935 E.01658
G1 X184.065 Y167.992 E.74555
G1 X184.065 Y168.528 E.01658
G1 X166.472 Y150.935 E.76899
G1 X165.935 Y150.935 E.01658
G1 X184.065 Y169.065 E.79244
G1 X183.528 Y169.065 E.01657
G1 X165.935 Y151.472 E.76901
G1 X165.935 Y152.008 E.01658
G1 X182.992 Y169.065 E.74556
G1 X182.456 Y169.065 E.01658
G1 X165.935 Y152.544 E.72211
G1 X165.935 Y153.081 E.01658
G1 X181.919 Y169.065 E.69867
G1 X181.383 Y169.065 E.01658
G1 X165.935 Y153.617 E.67522
G1 X165.935 Y154.154 E.01658
G1 X180.846 Y169.065 E.65177
G1 X180.31 Y169.065 E.01658
G1 X165.935 Y154.69 E.62833
G1 X165.935 Y155.226 E.01658
G1 X179.774 Y169.065 E.60488
G1 X179.237 Y169.065 E.01658
G1 X165.935 Y155.763 E.58143
G1 X165.935 Y156.299 E.01658
G1 X178.701 Y169.065 E.55799
G1 X178.164 Y169.065 E.01658
G1 X165.935 Y156.836 E.53454
G1 X165.935 Y157.372 E.01658
G1 X177.628 Y169.065 E.51109
G1 X177.092 Y169.065 E.01658
G1 X165.935 Y157.908 E.48765
G1 X165.935 Y158.445 E.01658
G1 X176.555 Y169.065 E.4642
G1 X176.019 Y169.065 E.01658
G1 X165.935 Y158.981 E.44075
G1 X165.935 Y159.518 E.01658
G1 X175.482 Y169.065 E.4173
G1 X174.946 Y169.065 E.01658
G1 X165.935 Y160.054 E.39386
G1 X165.935 Y160.591 E.01658
G1 X174.409 Y169.065 E.37041
G1 X173.873 Y169.065 E.01658
G1 X165.935 Y161.127 E.34696
G1 X165.935 Y161.663 E.01658
G1 X173.337 Y169.065 E.32352
G1 X172.8 Y169.065 E.01658
G1 X165.935 Y162.2 E.30007
G1 X165.935 Y162.736 E.01658
G1 X172.264 Y169.065 E.27662
G1 X171.727 Y169.065 E.01658
G1 X165.935 Y163.273 E.25318
G1 X165.935 Y163.809 E.01658
G1 X171.191 Y169.065 E.22973
G1 X170.655 Y169.065 E.01658
G1 X165.935 Y164.345 E.20628
G1 X165.935 Y164.882 E.01658
G1 X170.118 Y169.065 E.18283
G1 X169.582 Y169.065 E.01658
G1 X165.935 Y165.418 E.15939
G1 X165.935 Y165.955 E.01658
G1 X169.045 Y169.065 E.13594
G1 X168.509 Y169.065 E.01658
G1 X165.935 Y166.491 E.11249
G1 X165.935 Y167.027 E.01658
G1 X167.973 Y169.065 E.08905
G1 X167.436 Y169.065 E.01658
G1 X165.935 Y167.564 E.0656
G1 X165.935 Y168.1 E.01658
G1 X166.9 Y169.065 E.04215
G1 X166.363 Y169.065 E.01658
G1 X165.766 Y168.467 E.02612
; WIPE_START
G1 X166.363 Y169.065 E-.32117
G1 X166.518 Y169.065 E-.05883
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z9.8 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z9.8 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z9.8 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 9.6
; LAYER_HEIGHT: 0.200001
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 48/50
; update layer progress
M73 L48
M991 S0 P47 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z9.6
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5364
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z10 I1.022 J-.661 P1  F60000
G1 X183.467 Y169.234 Z10
G1 Z9.6
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.42222
G1 F15000
M204 S8000
G1 X184.065 Y168.637 E.02611
G1 X184.065 Y168.101 E.01658
G1 X183.101 Y169.065 E.04214
G1 X182.564 Y169.065 E.01658
G1 X184.065 Y167.564 E.06559
G1 X184.065 Y167.028 E.01658
G1 X182.028 Y169.065 E.08903
G1 X181.491 Y169.065 E.01658
G1 X184.065 Y166.491 E.11248
G1 X184.065 Y165.955 E.01658
G1 X180.955 Y169.065 E.13593
G1 X180.419 Y169.065 E.01658
G1 X184.065 Y165.419 E.15937
G1 X184.065 Y164.882 E.01658
G1 X179.882 Y169.065 E.18282
G1 X179.346 Y169.065 E.01658
G1 X184.065 Y164.346 E.20627
G1 X184.065 Y163.809 E.01658
G1 X178.809 Y169.065 E.22971
G1 X178.273 Y169.065 E.01658
G1 X184.065 Y163.273 E.25316
G1 X184.065 Y162.737 E.01658
G1 X177.737 Y169.065 E.27661
G1 X177.2 Y169.065 E.01658
G1 X184.065 Y162.2 E.30006
G1 X184.065 Y161.664 E.01658
G1 X176.664 Y169.065 E.3235
G1 X176.127 Y169.065 E.01658
G1 X184.065 Y161.127 E.34695
G1 X184.065 Y160.591 E.01658
G1 X175.591 Y169.065 E.3704
G1 X175.054 Y169.065 E.01658
G1 X184.065 Y160.054 E.39384
G1 X184.065 Y159.518 E.01658
G1 X174.518 Y169.065 E.41729
G1 X173.982 Y169.065 E.01658
M73 P94 R0
G1 X184.065 Y158.982 E.44074
G1 X184.065 Y158.445 E.01658
G1 X173.445 Y169.065 E.46418
G1 X172.909 Y169.065 E.01658
G1 X184.065 Y157.909 E.48763
G1 X184.065 Y157.372 E.01658
G1 X172.372 Y169.065 E.51108
G1 X171.836 Y169.065 E.01658
G1 X184.065 Y156.836 E.53452
G1 X184.065 Y156.3 E.01658
G1 X171.3 Y169.065 E.55797
G1 X170.763 Y169.065 E.01658
G1 X184.065 Y155.763 E.58142
G1 X184.065 Y155.227 E.01658
G1 X170.227 Y169.065 E.60486
G1 X169.69 Y169.065 E.01658
G1 X184.065 Y154.69 E.62831
G1 X184.065 Y154.154 E.01658
G1 X169.154 Y169.065 E.65176
G1 X168.618 Y169.065 E.01658
G1 X184.065 Y153.618 E.67521
G1 X184.065 Y153.081 E.01658
G1 X168.081 Y169.065 E.69865
G1 X167.545 Y169.065 E.01658
G1 X184.065 Y152.545 E.7221
G1 X184.065 Y152.008 E.01658
G1 X167.008 Y169.065 E.74555
G1 X166.472 Y169.065 E.01658
G1 X184.065 Y151.472 E.76899
G1 X184.065 Y150.935 E.01658
G1 X165.935 Y169.065 E.79244
G1 X165.935 Y168.528 E.01657
G1 X183.528 Y150.935 E.76901
G1 X182.992 Y150.935 E.01658
G1 X165.935 Y167.992 E.74556
G1 X165.935 Y167.456 E.01658
G1 X182.456 Y150.935 E.72211
G1 X181.919 Y150.935 E.01658
G1 X165.935 Y166.919 E.69867
G1 X165.935 Y166.383 E.01658
G1 X181.383 Y150.935 E.67522
G1 X180.846 Y150.935 E.01658
G1 X165.935 Y165.846 E.65177
G1 X165.935 Y165.31 E.01658
G1 X180.31 Y150.935 E.62833
G1 X179.774 Y150.935 E.01658
G1 X165.935 Y164.774 E.60488
G1 X165.935 Y164.237 E.01658
G1 X179.237 Y150.935 E.58143
G1 X178.701 Y150.935 E.01658
G1 X165.935 Y163.701 E.55799
G1 X165.935 Y163.164 E.01658
G1 X178.164 Y150.935 E.53454
G1 X177.628 Y150.935 E.01658
G1 X165.935 Y162.628 E.51109
G1 X165.935 Y162.092 E.01658
G1 X177.092 Y150.935 E.48765
G1 X176.555 Y150.935 E.01658
G1 X165.935 Y161.555 E.4642
G1 X165.935 Y161.019 E.01658
G1 X176.019 Y150.935 E.44075
G1 X175.482 Y150.935 E.01658
G1 X165.935 Y160.482 E.4173
G1 X165.935 Y159.946 E.01658
G1 X174.946 Y150.935 E.39386
G1 X174.409 Y150.935 E.01658
G1 X165.935 Y159.409 E.37041
G1 X165.935 Y158.873 E.01658
G1 X173.873 Y150.935 E.34696
G1 X173.337 Y150.935 E.01658
G1 X165.935 Y158.337 E.32352
G1 X165.935 Y157.8 E.01658
G1 X172.8 Y150.935 E.30007
G1 X172.264 Y150.935 E.01658
G1 X165.935 Y157.264 E.27662
G1 X165.935 Y156.727 E.01658
G1 X171.727 Y150.935 E.25318
G1 X171.191 Y150.935 E.01658
G1 X165.935 Y156.191 E.22973
G1 X165.935 Y155.655 E.01658
G1 X170.655 Y150.935 E.20628
G1 X170.118 Y150.935 E.01658
G1 X165.935 Y155.118 E.18284
G1 X165.935 Y154.582 E.01658
G1 X169.582 Y150.935 E.15939
G1 X169.045 Y150.935 E.01658
G1 X165.935 Y154.045 E.13594
G1 X165.935 Y153.509 E.01658
G1 X168.509 Y150.935 E.11249
G1 X167.973 Y150.935 E.01658
G1 X165.935 Y152.973 E.08905
G1 X165.935 Y152.436 E.01658
G1 X167.436 Y150.935 E.0656
G1 X166.9 Y150.935 E.01658
G1 X165.935 Y151.9 E.04215
G1 X165.935 Y151.363 E.01658
G1 X166.533 Y150.766 E.02612
; WIPE_START
G1 X165.935 Y151.363 E-.32117
G1 X165.935 Y151.518 E-.05883
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z10 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z10 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z10 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 9.8
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 49/50
; update layer progress
M73 L49
M991 S0 P48 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.398 Y169.398 F60000
G1 Z9.8
G1 E.4 F1800
; FEATURE: Inner wall
; LINE_WIDTH: 0.45
G1 F18000
M204 S8000
G1 X165.602 Y169.398 E.62349
G1 X165.602 Y150.602 E.62349
G1 X184.398 Y150.602 E.62349
G1 X184.398 Y169.338 E.6215
; COOLING_NODE: 0
M204 S250
G1 X184.79 Y169.79 F60000
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5435
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
; WIPE_START
G1 F12000
M204 S8000
G1 X183.79 Y169.733 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z10.2 I1.217 J.03 P1  F60000
G1 X184.234 Y151.533 Z10.2
G1 Z9.8
G1 E.4 F1800
; FEATURE: Internal solid infill
; LINE_WIDTH: 0.42222
M73 P95 R0
G1 F15000
M204 S8000
G1 X183.637 Y150.935 E.02611
G1 X183.101 Y150.935 E.01658
G1 X184.065 Y151.899 E.04214
G1 X184.065 Y152.436 E.01658
G1 X182.564 Y150.935 E.06559
G1 X182.028 Y150.935 E.01658
G1 X184.065 Y152.972 E.08903
G1 X184.065 Y153.509 E.01658
G1 X181.491 Y150.935 E.11248
G1 X180.955 Y150.935 E.01658
G1 X184.065 Y154.045 E.13593
G1 X184.065 Y154.581 E.01658
G1 X180.419 Y150.935 E.15937
G1 X179.882 Y150.935 E.01658
G1 X184.065 Y155.118 E.18282
G1 X184.065 Y155.654 E.01658
G1 X179.346 Y150.935 E.20627
G1 X178.809 Y150.935 E.01658
G1 X184.065 Y156.191 E.22971
G1 X184.065 Y156.727 E.01658
G1 X178.273 Y150.935 E.25316
G1 X177.737 Y150.935 E.01658
G1 X184.065 Y157.263 E.27661
G1 X184.065 Y157.8 E.01658
G1 X177.2 Y150.935 E.30005
G1 X176.664 Y150.935 E.01658
G1 X184.065 Y158.336 E.3235
G1 X184.065 Y158.873 E.01658
G1 X176.127 Y150.935 E.34695
G1 X175.591 Y150.935 E.01658
G1 X184.065 Y159.409 E.3704
G1 X184.065 Y159.946 E.01658
G1 X175.054 Y150.935 E.39384
G1 X174.518 Y150.935 E.01658
G1 X184.065 Y160.482 E.41729
G1 X184.065 Y161.018 E.01658
G1 X173.982 Y150.935 E.44074
G1 X173.445 Y150.935 E.01658
G1 X184.065 Y161.555 E.46418
G1 X184.065 Y162.091 E.01658
G1 X172.909 Y150.935 E.48763
G1 X172.372 Y150.935 E.01658
G1 X184.065 Y162.628 E.51108
G1 X184.065 Y163.164 E.01658
G1 X171.836 Y150.935 E.53452
G1 X171.3 Y150.935 E.01658
G1 X184.065 Y163.7 E.55797
G1 X184.065 Y164.237 E.01658
M73 P96 R0
G1 X170.763 Y150.935 E.58142
G1 X170.227 Y150.935 E.01658
G1 X184.065 Y164.773 E.60486
G1 X184.065 Y165.31 E.01658
G1 X169.69 Y150.935 E.62831
G1 X169.154 Y150.935 E.01658
G1 X184.065 Y165.846 E.65176
G1 X184.065 Y166.382 E.01658
G1 X168.618 Y150.935 E.67521
G1 X168.081 Y150.935 E.01658
G1 X184.065 Y166.919 E.69865
G1 X184.065 Y167.455 E.01658
G1 X167.545 Y150.935 E.7221
G1 X167.008 Y150.935 E.01658
G1 X184.065 Y167.992 E.74555
G1 X184.065 Y168.528 E.01658
G1 X166.472 Y150.935 E.76899
G1 X165.935 Y150.935 E.01658
G1 X184.065 Y169.065 E.79244
G1 X183.528 Y169.065 E.01657
G1 X165.935 Y151.472 E.76901
G1 X165.935 Y152.008 E.01658
G1 X182.992 Y169.065 E.74556
G1 X182.456 Y169.065 E.01658
G1 X165.935 Y152.544 E.72211
G1 X165.935 Y153.081 E.01658
G1 X181.919 Y169.065 E.69867
G1 X181.383 Y169.065 E.01658
G1 X165.935 Y153.617 E.67522
G1 X165.935 Y154.154 E.01658
G1 X180.846 Y169.065 E.65177
G1 X180.31 Y169.065 E.01658
G1 X165.935 Y154.69 E.62833
G1 X165.935 Y155.226 E.01658
G1 X179.774 Y169.065 E.60488
G1 X179.237 Y169.065 E.01658
G1 X165.935 Y155.763 E.58143
G1 X165.935 Y156.299 E.01658
G1 X178.701 Y169.065 E.55799
G1 X178.164 Y169.065 E.01658
G1 X165.935 Y156.836 E.53454
G1 X165.935 Y157.372 E.01658
G1 X177.628 Y169.065 E.51109
G1 X177.092 Y169.065 E.01658
G1 X165.935 Y157.908 E.48765
G1 X165.935 Y158.445 E.01658
G1 X176.555 Y169.065 E.4642
G1 X176.019 Y169.065 E.01658
G1 X165.935 Y158.981 E.44075
G1 X165.935 Y159.518 E.01658
G1 X175.482 Y169.065 E.4173
G1 X174.946 Y169.065 E.01658
G1 X165.935 Y160.054 E.39386
G1 X165.935 Y160.591 E.01658
G1 X174.409 Y169.065 E.37041
G1 X173.873 Y169.065 E.01658
G1 X165.935 Y161.127 E.34696
G1 X165.935 Y161.663 E.01658
G1 X173.337 Y169.065 E.32352
G1 X172.8 Y169.065 E.01658
G1 X165.935 Y162.2 E.30007
G1 X165.935 Y162.736 E.01658
G1 X172.264 Y169.065 E.27662
G1 X171.727 Y169.065 E.01658
G1 X165.935 Y163.273 E.25318
G1 X165.935 Y163.809 E.01658
G1 X171.191 Y169.065 E.22973
G1 X170.655 Y169.065 E.01658
G1 X165.935 Y164.345 E.20628
G1 X165.935 Y164.882 E.01658
G1 X170.118 Y169.065 E.18283
G1 X169.582 Y169.065 E.01658
G1 X165.935 Y165.418 E.15939
G1 X165.935 Y165.955 E.01658
G1 X169.045 Y169.065 E.13594
G1 X168.509 Y169.065 E.01658
G1 X165.935 Y166.491 E.11249
G1 X165.935 Y167.027 E.01658
G1 X167.973 Y169.065 E.08905
G1 X167.436 Y169.065 E.01658
G1 X165.935 Y167.564 E.0656
G1 X165.935 Y168.1 E.01658
G1 X166.9 Y169.065 E.04215
G1 X166.363 Y169.065 E.01658
G1 X165.766 Y168.467 E.02612
; WIPE_START
G1 X166.363 Y169.065 E-.32117
G1 X166.518 Y169.065 E-.05883
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z10.2 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z10.2 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z10.2 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; CHANGE_LAYER
; Z_HEIGHT: 10
; LAYER_HEIGHT: 0.2
;======== H2D 20250710 layer_change ========
; layer num/total_layer_count: 50/50
; update layer progress
M73 L50
M991 S0 P49 ;notify layer change

; OBJECT_ID: 132
; COOLING_NODE: 0
G1 X184.79 Y169.79 F60000
G1 Z10
G1 E.4 F1800
; FEATURE: Outer wall
; LINE_WIDTH: 0.42
G1 F5483
M204 S5000
G1 X165.21 Y169.79 E.60164
G1 X165.21 Y150.21 E.60164
G1 X184.79 Y150.21 E.60164
G1 X184.79 Y169.73 E.5998
M204 S10000
G1 X184.583 Y168.815 F60000
; FEATURE: Top surface
G1 F12000
M204 S2000
G1 X183.815 Y169.583 E.03337
G1 X183.281 Y169.583
G1 X184.583 Y168.281 E.05654
G1 X184.583 Y167.748
G1 X182.748 Y169.583 E.07972
G1 X182.215 Y169.583
G1 X184.583 Y167.215 E.10289
G1 X184.583 Y166.682
G1 X181.682 Y169.583 E.12606
G1 X181.148 Y169.583
G1 X184.583 Y166.148 E.14923
G1 X184.583 Y165.615
G1 X180.615 Y169.583 E.17241
G1 X180.082 Y169.583
G1 X184.583 Y165.082 E.19558
G1 X184.583 Y164.549
G1 X179.549 Y169.583 E.21875
G1 X179.015 Y169.583
G1 X184.583 Y164.015 E.24192
G1 X184.583 Y163.482
G1 X178.482 Y169.583 E.2651
G1 X177.949 Y169.583
G1 X184.583 Y162.949 E.28827
G1 X184.583 Y162.416
M73 P97 R0
G1 X177.416 Y169.583 E.31144
G1 X176.882 Y169.583
G1 X184.583 Y161.882 E.33461
G1 X184.583 Y161.349
G1 X176.349 Y169.583 E.35779
G1 X175.816 Y169.583
G1 X184.583 Y160.816 E.38096
G1 X184.583 Y160.283
G1 X175.283 Y169.583 E.40413
G1 X174.749 Y169.583
G1 X184.583 Y159.749 E.42731
G1 X184.583 Y159.216
G1 X174.216 Y169.583 E.45048
G1 X173.683 Y169.583
G1 X184.583 Y158.683 E.47365
G1 X184.583 Y158.15
G1 X173.15 Y169.583 E.49682
G1 X172.616 Y169.583
G1 X184.583 Y157.616 E.52
G1 X184.583 Y157.083
G1 X172.083 Y169.583 E.54317
G1 X171.55 Y169.583
G1 X184.583 Y156.55 E.56634
G1 X184.583 Y156.017
G1 X171.017 Y169.583 E.58951
G1 X170.483 Y169.583
G1 X184.583 Y155.483 E.61269
G1 X184.583 Y154.95
G1 X169.95 Y169.583 E.63586
G1 X169.417 Y169.583
G1 X184.583 Y154.417 E.65903
G1 X184.583 Y153.883
G1 X168.883 Y169.583 E.6822
G1 X168.35 Y169.583
G1 X184.583 Y153.35 E.70538
G1 X184.583 Y152.817
G1 X167.817 Y169.583 E.72855
G1 X167.284 Y169.583
G1 X184.583 Y152.284 E.75172
G1 X184.583 Y151.75
G1 X166.75 Y169.583 E.77489
G1 X166.217 Y169.583
G1 X184.583 Y151.217 E.79807
G1 X184.583 Y150.684
G1 X165.684 Y169.583 E.82124
G1 X165.417 Y169.316
G1 X184.316 Y150.417 E.82123
G1 X183.783 Y150.417
G1 X165.417 Y168.783 E.79806
G1 X165.417 Y168.249
G1 X183.249 Y150.417 E.77489
G1 X182.716 Y150.417
G1 X165.417 Y167.716 E.75172
G1 X165.417 Y167.183
G1 X182.183 Y150.417 E.72854
G1 X181.65 Y150.417
G1 X165.417 Y166.65 E.70537
G1 X165.417 Y166.116
G1 X181.116 Y150.417 E.6822
G1 X180.583 Y150.417
G1 X165.417 Y165.583 E.65903
G1 X165.417 Y165.05
G1 X180.05 Y150.417 E.63585
G1 X179.517 Y150.417
G1 X165.417 Y164.517 E.61268
G1 X165.417 Y163.983
M73 P98 R0
G1 X178.983 Y150.417 E.58951
G1 X178.45 Y150.417
G1 X165.417 Y163.45 E.56634
G1 X165.417 Y162.917
G1 X177.917 Y150.417 E.54316
G1 X177.384 Y150.417
G1 X165.417 Y162.384 E.51999
G1 X165.417 Y161.85
G1 X176.85 Y150.417 E.49682
G1 X176.317 Y150.417
G1 X165.417 Y161.317 E.47364
G1 X165.417 Y160.784
G1 X175.784 Y150.417 E.45047
G1 X175.251 Y150.417
G1 X165.417 Y160.251 E.4273
G1 X165.417 Y159.717
G1 X174.717 Y150.417 E.40413
G1 X174.184 Y150.417
G1 X165.417 Y159.184 E.38095
G1 X165.417 Y158.651
G1 X173.651 Y150.417 E.35778
G1 X173.118 Y150.417
G1 X165.417 Y158.118 E.33461
G1 X165.417 Y157.584
G1 X172.584 Y150.417 E.31144
G1 X172.051 Y150.417
G1 X165.417 Y157.051 E.28826
G1 X165.417 Y156.518
G1 X171.518 Y150.417 E.26509
G1 X170.985 Y150.417
G1 X165.417 Y155.984 E.24192
G1 X165.417 Y155.451
G1 X170.451 Y150.417 E.21875
G1 X169.918 Y150.417
G1 X165.417 Y154.918 E.19557
G1 X165.417 Y154.385
G1 X169.385 Y150.417 E.1724
G1 X168.851 Y150.417
G1 X165.417 Y153.851 E.14923
G1 X165.417 Y153.318
G1 X168.318 Y150.417 E.12606
G1 X167.785 Y150.417
G1 X165.417 Y152.785 E.10288
G1 X165.417 Y152.252
G1 X167.252 Y150.417 E.07971
G1 X166.718 Y150.417
G1 X165.417 Y151.718 E.05654
G1 X165.417 Y151.185
G1 X166.185 Y150.417 E.03336
; WIPE_START
M204 S8000
G1 X165.478 Y151.124 E-.38
; WIPE_END
G1 E-.02 F1800
M204 S10000
G17
G3 Z10.4 I1.217 J0 P1  F60000
;======== H2D 20251104========
; SKIPPABLE_START
; SKIPTYPE: timelapse
M622.1 S1 ; for prev firmware, default turned on

M1002 judge_flag timelapse_record_flag

    M622 J1
    M993 A2 B2 C2
    M993 A0 B0 C0
    
    M622.1 S0 ; for prev firmware, default turn off
    M1002 set_flag smooth_safe_pos_suppoprt_flag=1
    M1002 judge_flag smooth_safe_pos_suppoprt_flag
    
    M622 J0
        

        
            M9711 M0 E0 X177 Y238 Z10.4 S11 C10 O0 T3000
        

        
    M623

    M622 J1
        

        
            M9711 M0 E0 U177 V238 Z10.4 S11 C10 O0 T3000
        

        
    M623

    M993 A3 B3 C3

M623
; SKIPPABLE_END

; close powerlost recovery
M1003 S0
M106 S0
M106 P2 S0
M981 S0 P20000 ; close spaghetti detector
; FEATURE: Custom
; MACHINE_END_GCODE_START
; filament end gcode 
;========== H2D end ==========
;===== date: 2025/12/26 =====

G392 S0 ;turn off nozzle clog detect
M993 A0 B0 C0 ; nozzle cam detection not allowed.

M400 ; wait for buffer to clear
G92 E0 ; zero the extruder
G1 E-0.8 F1800 ; retract
M400
M211 Z1
G1 Z10.4 F900 ; lower z a little

M1002 judge_flag timelapse_record_flag
M622 J1
    G150.3
    M400 ; wait all motion done
    M991 S0 P-1 ;end smooth timelapse at safe pos
    M400 S5 ;wait for last picture to be taken
M623  ;end of "timelapse_record_flag"

G90
G1 Z20 F900 ; lower z a little

G90
M141 S0 ; turn off chamber heating
M140 S0 ; turn off bed
M106 S0 ; turn off fan
M106 P2 S0 ; turn off remote part cooling fan
M106 P3 S0 ; turn off chamber cooling fan
M106 P9 S0 ; turn off ext toodhead cooling fan
; pull back filament to AMS
M620 S65535
T65535
G150.2
M621 S65535

M620 S65279
T65279
G150.2
M621 S65279

G150.3

M104 S0 T0; turn off hotend
M104 S0 T1; turn off hotend

M400 ; wait all motion done
M17 S
M17 Z0.4 ; lower z motor current to reduce impact if there is something in the bottom

    
        G1 Z105 F600
        G1 Z103
    

M400 P100
M17 R ; restore z current

M220 S100  ; Reset feedrate magnitude
M201.2 K1.0 ; Reset acc magnitude
M73.2   R1.0 ;Reset left time magnitude
M1002 set_gcode_claim_speed_level : 0

M1015.4 S0 K0 ;disable air printing detect

;=====printer finish air purification=========
M622.1 S0
M1002 judge_flag print_finish_air_filt_flag

M622 J1
M1002 gcode_claim_action : 66
M145 P1
M106 P6 S255
M400 S180
M106 P6 S0
M623

M622 J2
M1002 gcode_claim_action : 66
M145 P0
M106 P3 S127
M400 S180
M106 P3 S0
M623
;=====printer finish air purification=========


;=====printer finish  sound=========
M17
M400 S1
M1006 S1
M1006 A53 B10 L99 C53 D10 M99 E53 F10 N99 
M1006 A57 B10 L99 C57 D10 M99 E57 F10 N99 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A53 B10 L99 C53 D10 M99 E53 F10 N99 
M1006 A57 B10 L99 C57 D10 M99 E57 F10 N99 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A48 B10 L99 C48 D10 M99 E48 F10 N99 
M1006 A0 B15 L0 C0 D15 M0 E0 F15 N0 
M1006 A60 B10 L99 C60 D10 M99 E60 F10 N99 
M1006 W
;=====printer finish  sound=========
M400
M18

M73 P100 R0
; EXECUTABLE_BLOCK_END

