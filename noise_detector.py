#!/usr/bin/env python3

from serial import Serial, SerialException, EIGHTBITS, PARITY_NONE, STOPBITS_ONE
import time
import logging


class NoiseDetector:
    DEFAULT_SERIAL_PORT = '/dev/ttyACM0'
    DEFAULT_BAUD_RATE = 9600
    SERIAL_TIMEOUT = 2
    MAXIMUM_MESSAGE_LENGTH = 256

    def __init__(self, serial_port=DEFAULT_SERIAL_PORT, baud_rate=DEFAULT_BAUD_RATE):
        self.logger = logging.getLogger("NoiseDetector")
        self._serial = Serial(port=serial_port,
                        baudrate=baud_rate,
                        bytesize=EIGHTBITS,
                        parity=PARITY_NONE,
                        stopbits=STOPBITS_ONE,
                        timeout=self.SERIAL_TIMEOUT)

    def __del__(self):
        if self._serial.is_open:
            self._serial.close()
        self.logger.debug("Serial port closed.")

    def _compute_checksum(self, data):
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

    def _get_message(self, address_code, function_code, register_start_address, register_length):
        # Construct the message without checksum
        message = [address_code, function_code] + register_start_address + register_length

        # Compute checksum
        crc_low, crc_high = self._compute_checksum(message)

        # Append checksum to the message
        full_message = message + [crc_low, crc_high]

        return full_message

    def _parse_reply(self, reply):
        if len(reply) < 5:  # Minimum length for a valid Modbus RTU message
            self.logger.error("Incomplete message received.")
            return
        
        # Convert reply values to integers
        try:
            reply = [int(value) for value in reply]
        except ValueError:
            self.logger.error("Non-integer value in reply.")
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
        computed_crc_low, computed_crc_high = self._compute_checksum(reply[:-2])
                
        if (received_crc_low, received_crc_high) != (computed_crc_low, computed_crc_high):
            self.logger.error("Checksum mismatch.")
            return
                
        # Print parsed values
        self.logger.debug(f"Address Code: {hex(address_code)}")
        self.logger.debug(f"Function Code: {hex(function_code)}")
        self.logger.debug(f"Number of Valid Bytes: {num_valid_bytes}")
        self.logger.debug(f"Data Areas: {[hex(data) for data in data_areas]}")
            
        return data_areas
    
    def read_noise_level(self):
        message = self._get_message(0x01, 0x03, [0x00, 0x00], [0x00, 0x01])
        try:
            self._serial.write(message)
            reply = self._serial.read(self.MAXIMUM_MESSAGE_LENGTH)
        except SerialException as exp:
            self.logger.error(str(exp))
            return None
        data = self._parse_reply(reply)
        noise_level = float(data[1] << 8 | data[0]) / 10
        self.logger.debug(f"Noise Level: {noise_level} dB")
        return noise_level

    def read_device_info(self):
        message = self._get_message(0xFF, 0x03, [0x07, 0xD0], [0x00, 0x02])
        try:
            self._serial.write(message)
            reply = self._serial.read(self.MAXIMUM_MESSAGE_LENGTH)
        except SerialException as exp:
            self.logger.error(str(exp))
            return None
        data = self._parse_reply(reply)
        baud_rate_index = data[1] << 8 | data[0]
        device_address = data[3] << 8 | data[2]
        self.logger.debug(f"Baud Rate Index: {baud_rate_index}")
        self.logger.debug(f"Device Address: {device_address}")
        return baud_rate_index, device_address

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    detector = NoiseDetector()
    noise_level = detector.read_noise_level()
    if noise_level is not None:
        print(f"Noise Level: {noise_level} dB")
    device_info = detector.read_device_info()
    if device_info is not None:
        print(f"Device Info: {device_info}")
