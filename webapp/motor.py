#DIR: CW - Tail to MOTOR; CCW - MOTOR to Tail (Need to set on-device dir for Camera axis to CCW)
#ID: 01 - Camera (X-Axis)
#    02 - FILM (Z-Axis * 2)
#    03 - Focus (Y_Axis * 2)

#   Function Params assume decimal ints/flaots

import serial

### Global literals
GLOBAL_CHECKSUM         = '6B'

FOCUS_ID                = '01'
CAMERA_ID               = '02'
LED_ID                  = '03'

FILM_ID                 = '04'
FILM_ID_2               = '05'

GLOBAL_STEPPING         = 32.0
GLOBAL_ANGLE            = 1.8
### Motor serial command and flag bits ###
CW                      = '00'      # CW - Tail to MOTOR
CCW                     = '01'      # CCW - MOTOR to Tail
FLAG_NO_SYNC            = '00'
FLAG_SYNC               = '01'
FLAG_RELATIVE_POSITION  = '00'
FLAG_ABSOLUTE_POSITION  = '01'
FLAG_RESET              = '9A'
SYNC_PHRASE             = '00FF66' + GLOBAL_CHECKSUM
M_ENABLE                = 'F3AB'
CONTROL_VELOCITY_MODE   = 'F6'
CONTROL_POSITION_MODE   = 'FD'
IMMIDIATE_STOP          = 'FE98'

def init(path = "/dev/ttyTHS1", baud = 115200, time_out = 1):
    global ttl 
    ttl = serial.Serial(path, baudrate = baud, timeout = time_out, bytesize = serial.EIGHTBITS, parity = serial.PARITY_NONE, stopbits = serial.STOPBITS_ONE)

def m_enable(address, enable):
    add = hex(address).lstrip("0x")
    ttl.read(10)

def mm_to_pulse(mm: float, mm_per_rot: int):
    degree = (mm / mm_per_rot ) * 360
    pulse = (degree / GLOBAL_ANGLE) * GLOBAL_STEPPING

    if pulse <= 1: print("ALARM: TRAVEL DISTANCE TOO SHORT")
    return int(pulse)

def move_position_mode(id, direction, velocity, acceleration, num_pulse, sync = False, relative = True, checksum = GLOBAL_CHECKSUM):
    v_str = hex(velocity).lstrip("0x").zfill(4).upper()
    acc_str = hex(acceleration).lstrip("0x").zfill(2).upper()
    pulse_str = hex(num_pulse).lstrip("0x").zfill(8).upper()
    data = id + CONTROL_POSITION_MODE + direction + v_str + acc_str + pulse_str + (FLAG_RELATIVE_POSITION if relative else FLAG_ABSOLUTE_POSITION) + ('01' if sync else '00') + checksum

    ttl.write(bytes.fromhex(data))
    print('move_position_mode sent: ' + data)
    line = ttl.read(20)
    print('move_position_mode received response: ' + line.hex())

#same as move_position mode but takes um as unit of travel instead of number of pulses
def MPM(id, direction, velocity, acceleration, distance, sync = False, relative = True, checksum = GLOBAL_CHECKSUM):
    pulse = mm_to_pulse(distance, 4)
    move_position_mode(id, direction, velocity, acceleration, pulse, sync, relative, checksum)

def z_move(dir):
    if dir == 'up':
        focus_move(CW, 100, 0.01)
        data = bytes.fromhex('03FD0000DF00000000FF00006B')
    elif dir == 'down':
        focus_move(CCW, 100, 0.01)
        data = bytes.fromhex('03FD0100DF00000000FF00006B')


def cam_move(dir, rpm, distance):
    MPM(CAMERA_ID, dir, rpm, 0, distance)

def led_move(dir, rpm, distance):
    #compensate for different mm_per_rot for film axis
    distance_comp = distance * 1
    MPM(LED_ID, dir, rpm, 0, distance_comp)

def focus_move(dir, rpm, distance):
    MPM(FOCUS_ID, dir, rpm, 0, distance)

def film_move_sync(dir, rpm, distance):
    #compensate for different mm_per_rot for film axis
    distance_comp = distance * 1
    MPM(FILM_ID, dir, rpm, 0, distance_comp, sync = True)
    MPM(FILM_ID_2, dir, rpm, 0, distance_comp, sync = True)
    direct_write(SYNC_PHRASE)

def film_move(dir, rpm, distance):
    #compensate for different mm_per_rot for film axis
    distance_comp = distance * 1
    MPM(FILM_ID, dir, rpm, 0, distance_comp)

def reset(id):
    data = id + FLAG_RESET + '02006B'
    ttl.write(bytes.fromhex(data))
    print('reset sent: '+ data)

def direct_write(data):
    ttl.write(bytes.fromhex(data))
    print('DIRECT sent: '+ data)

if __name__ == '__main__':
    ### PUT TESTS HERE ###
    init()
    #film_move(CW, 200, 10)
    cam_move(CCW, 200, 2)
    #led_move(CCW, 800, 2)
    #focus_move(CW, 200, 4)
    #reset(FILM_ID)