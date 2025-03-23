#!/usr/bin/env python3

from serial import Serial, SerialException, EIGHTBITS, PARITY_NONE, STOPBITS_ONE
import time

port = '/dev/ttyACM0'
baud = 9600
serial_timeout = 2


def compute_checksum(data):
    """
    Compute the Modbus RTU CRC-16 checksum.
    """
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    # Return CRC low byte first, then high byte
    return crc & 0xFF, (crc >> 8) & 0xFF


def send_message(address_code, function_code, register_start_address, register_length):
    # Construct the message without checksum
    message = [address_code, function_code] + register_start_address + register_length

    # Compute checksum
    crc_low, crc_high = compute_checksum(message)

    # Append checksum to the message
    full_message = message + [crc_low, crc_high]

    return full_message


def parse_reply(reply):
    if len(reply) < 5:  # Minimum length for a valid Modbus RTU message
        print("Error: Incomplete message received.")
        return
    
    # Convert reply values to integers
    try:
        reply = [int(value) for value in reply]
    except ValueError:
        print("Error: Non-integer value in reply.")
        return

    # Parse the reply message
    address_code = reply[0]
    function_code = reply[1]
    num_valid_bytes = reply[2]
            
    # Extract data areas based on the number of valid bytes
    data_areas = []
    for i in range(num_valid_bytes // 2):  # Each data area is 2 bytes
        start_idx = 3 + i * 2
        data_areas.append(reply[start_idx + 1])
        data_areas.append(reply[start_idx])
            
    # Extract and validate the checksum
    received_crc_low = reply[-2]
    received_crc_high = reply[-1]
            
    # Compute checksum for verification (excluding received CRC bytes)
    computed_crc_low, computed_crc_high = compute_checksum(reply[:-2])
            
    if (received_crc_low, received_crc_high) != (computed_crc_low, computed_crc_high):
        print("Error: Checksum mismatch.")
        return
            
    # Print parsed values
    print("Address Code:", hex(address_code))
    print("Function Code:", hex(function_code))
    print("Number of Valid Bytes:", num_valid_bytes)
    print("Data Areas:", [hex(data) for data in data_areas])
        
    return data_areas


try:
    serial = Serial(port=port,
                    baudrate=baud,
                    bytesize=EIGHTBITS,
                    parity=PARITY_NONE,
                    stopbits=STOPBITS_ONE,
                    timeout=serial_timeout)
    # read the address and baud rate of the device
    #full_message = send_message(0xFF, 0x03, [0x07, 0xD0], [0x00, 0x02])
    #read noise level
    full_message = send_message(0x01, 0x03, [0x00, 0x00], [0x00, 0x01])
    serial.write(full_message)
except SerialException as exp:
      print(str(exp))

try:
    while True:
        if serial.in_waiting > 0:
            reply = serial.read(256)
            data = parse_reply(reply)
            #baud_rate_index = data[1] << 8 | data[0]
            #device_address = data[3] << 8 | data[2]
            #print("Baud Rate Index:", baud_rate_index)
            #print("Device Address:", device_address)
            noise_level = float(data[1] << 8 | data[0]) / 10
            print("Noise Level:", noise_level, "dB")
            break
        time.sleep(0.1)  # Short delay to prevent CPU overuse
except KeyboardInterrupt:
    print("Stopping data collection")
finally:
    serial.close()
