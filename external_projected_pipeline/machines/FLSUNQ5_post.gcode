;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Code Above ;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

G1 E-.045 F1800
M106 S0
;TYPE:Custom
; MACHINE_END_GCODE_START
; filament end gcode 
M107
M104 S0
M140 S0
G92 E1
G1 E-1 F300
G28 X0 Y0
;M84
M18 S180 ;disable motors after 180s
M73 P100 R0
; EXECUTABLE_BLOCK_END