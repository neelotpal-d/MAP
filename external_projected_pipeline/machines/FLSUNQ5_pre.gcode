; EXECUTABLE_BLOCK_START
M73 P0 R13
M201 X1500 Y1500 Z1500 E3000
M203 X200 Y200 Z200 E60
M204 P1500 R2000 T1500
M205 X5.00 Y5.00 Z5.00 E5.00
;TYPE:Custom
;STARTGCODE
M117 Initializing
; G90 ; use absolute coordinates
M83 ; extruder relative mode
M107
G28 ;Home
M140 S55 ; set bed temp
M190 S55 ; wait for bed temp
M104 S220 ; set extruder temp
M109 S220 ; wait for extruder temp

G92 E0
G1 X-98 Y0 Z0.2 F4000 ; move to arc start
G3 X0 Y-98 I98 Z0.2 E40 F400 ; lay arc stripe 90deg
G0 Z1 
G92 E0.0
; MACHINE_START_GCODE_END
; filament start gcode
;VT0
G90
G21
M83 ; use relative distances for extrusionss
G1 F2000
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Main Print Below ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
