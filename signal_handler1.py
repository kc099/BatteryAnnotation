import threading
import time
import traceback
import json

# Import centralized debug system
from debug_utils import debug_print

# Try importing pyserial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    debug_print("pyserial library not found - 24V signal detection will not be available", "errors")

class SignalHandler:
    """Handler for external 24V signal detection and communication for battery lid quality control"""
    
    def __init__(self, signal_callback=None):
        """
        Initialize signal handler
        
        Args:
            signal_callback: Function to call when 24V signal is detected
        """
        self.signal_callback = signal_callback
        self.thread = None
        self.stop_flag = False
        self.is_running = False
        self.serial_port = None
        self.settings = self._load_settings()
        self.is_processing_signal = False  # Flag to track signal processing state
        self.last_callback_time = 0 
        self.min_callback_interval = 2.0      # Track last callback time globally
        
    def _load_settings(self):
        """Load settings from settings.json"""
        from config_manager import ConfigManager
        config = ConfigManager()
        return {
            "com_port": config.settings.get("com_port"),
            "baud_rate": config.settings.get("baud_rate", 19200),
            "modbus_slave_id": config.settings.get("modbus_slave_id", 1)
        }
        
    def start_detection(self):
        """Start 24V signal detection thread"""
        if not SERIAL_AVAILABLE:
            debug_print("24V signal detection not available (pyserial not installed)", "signal")
            return False
            
        if self.is_running:
            debug_print("Signal detection already running", "signal")
            return True
            
        # Check if we have a valid COM port
        com_port = self.settings.get("com_port")
        
        # If no COM port is selected, try to auto-detect
        if not com_port:
            try:
                available_ports = [port.device for port in serial.tools.list_ports.comports()]
                if available_ports:
                    com_port = available_ports[0]  # Use the first available port
                    debug_print(f"Auto-detected COM port: {com_port}", "signal")
                    # Update settings with the detected port
                    self.settings["com_port"] = com_port
                else:
                    debug_print("No COM ports available", "errors")
                    return False
            except Exception as e:
                debug_print(f"Error checking available ports: {e}", "errors")
                return False
            
        # Verify port exists
        try:
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                debug_print(f"Configured COM port {com_port} not found. Available ports: {available_ports}", "errors")
                return False
        except Exception as e:
            debug_print(f"Error checking available ports: {e}", "errors")
            return False
            
        self.stop_flag = False
        self.thread = threading.Thread(target=self._detect_signal_thread, daemon=True)
        self.thread.start()
        self.is_running = True
        return True
        
    def stop_detection(self):
        """Stop 24V signal detection thread"""
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.is_running = False
        self.is_processing_signal = False  # Reset processing state
        
    def reset_processing_state(self):
        """Manually reset the processing state (useful for debugging)"""
        self.is_processing_signal = False
        debug_print("Processing state reset manually", "signal")
        
    def _detect_signal_thread(self):
        """Thread for monitoring 24V signal with frame buffering and cooldown"""
        if not SERIAL_AVAILABLE:
            return
            
        try:
            # Get COM port and baud rate from settings
            com_port = self.settings.get("com_port")
            baud_rate = self.settings.get("baud_rate", 19200)
            slave_id = self.settings.get("modbus_slave_id", 1)
            debug_print(f"Starting signal detection with slave ID: {slave_id}", "signal")
            
            if not com_port:
                debug_print("No COM port selected in settings", "errors")
                return
                
            # Verify port exists
            available_ports = [port.device for port in serial.tools.list_ports.comports()]
            if com_port not in available_ports:
                debug_print(f"Selected COM port {com_port} not found", "errors")
                return
            
            # Open the serial port with improved settings
            self.serial_port = serial.Serial(
                port=com_port,
                baudrate=baud_rate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,        # Shorter timeout for non-blocking behavior
                write_timeout=0.3,  # Add write timeout
                inter_byte_timeout=None  # Let data accumulate in buffer
            )
            debug_print(f"Connected to {com_port} at {baud_rate} baud for 24V signal detection", "signal")
            
            # Clear any existing data in the buffer
            self.serial_port.reset_input_buffer()
            self.serial_port.reset_output_buffer()
            debug_print("Serial buffers cleared", "signal")
            
            # Initialize frame buffering variables
            data_buffer = bytearray()
            frame_size = 8  # Expected Modbus frame size
            
            debug_print("Modbus frame detection started - waiting for signals...", "signal")
            
            # Main detection loop with proper frame buffering
            while not self.stop_flag:
                try:
                    # Check if there's data available
                    bytes_waiting = self.serial_port.in_waiting
                    if bytes_waiting > 0:
                        # Read all available bytes
                        new_data = self.serial_port.read(bytes_waiting)
                        if len(new_data) > 0:
                            data_buffer.extend(new_data)
                            debug_print(f"Received {len(new_data)} bytes: {new_data.hex().upper()}", "signal")
                            debug_print(f"Buffer now contains {len(data_buffer)} bytes total", "signal")
                            
                            # Process complete frames
                            while len(data_buffer) >= frame_size:
                                # Extract potential frame
                                potential_frame = data_buffer[:frame_size]
                                debug_print(f"Processing potential 8-byte frame: {potential_frame.hex().upper()}", "signal")
                                
                                # Process the frame
                                if self._process_modbus_data(potential_frame, slave_id):
                                    # Valid frame found, remove it from buffer
                                    data_buffer = data_buffer[frame_size:]
                                    debug_print(f"Valid frame processed, {len(data_buffer)} bytes remaining in buffer", "signal")
                                    break  # Process one frame at a time to avoid callback flooding
                                else:
                                    # No valid frame at start of buffer, shift by 1 byte and try again
                                    discarded_byte = data_buffer[0]
                                    data_buffer = data_buffer[1:]
                                    debug_print(f"Invalid frame start, discarded byte: 0x{discarded_byte:02X}, {len(data_buffer)} bytes remaining", "signal")
                            
                            # Clear buffer if it gets too large (prevent memory issues)
                            if len(data_buffer) > 100:
                                debug_print(f"Buffer too large ({len(data_buffer)} bytes), clearing...", "signal")
                                data_buffer.clear()
                    
                    # Small delay to prevent CPU hogging
                    time.sleep(0.01)
                    
                except Exception as e:
                    debug_print(f"Error in signal detection loop: {e}", "errors")
                    time.sleep(1.0)  # Wait before retry
                    
        except Exception as e:
            debug_print(f"Critical error in signal detection thread: {e}", "errors")
            traceback.print_exc()
        finally:
            # Cleanup serial port
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.close()
                    debug_print("Serial port closed", "signal")
                except Exception as e:
                    debug_print(f"Error closing serial port: {e}", "errors")
            self.serial_port = None
            self.is_running = False
    
    def _process_modbus_data(self, data, expected_slave_id):
        """Process received data to find valid Modbus frames
        
        Args:
            data: Received data bytes (should be exactly 8 bytes)
            expected_slave_id: Expected slave ID from settings
            
        Returns:
            bool: True if valid Modbus frame was found and processed, False otherwise
        """
        try:
            # Ensure we have exactly 8 bytes
            if len(data) != 8:
                debug_print(f"Invalid frame length: {len(data)} bytes (expected 8)", "signal")
                return False
            
            # Extract slave ID and function code from first two bytes
            slave_id = data[0]
            function_code = data[1]
            
            debug_print(f"Frame analysis: Slave ID: {slave_id} (expected: {expected_slave_id}), Function: 0x{function_code:02X}", "signal")
            
            # Check for read holding registers request (0x03) to configured slave ID
            if slave_id == expected_slave_id and function_code == 0x03:
                debug_print(f"VALID MODBUS FRAME DETECTED - Slave ID: {slave_id}, Function: 0x03", "signal")
                debug_print(f"  Complete frame: {data.hex().upper()}", "signal")
                
                # Debounce logic: only allow callback if enough time has passed
                now = time.time()
                if now - self.last_callback_time < self.min_callback_interval:
                    debug_print(f"Signal ignored due to debounce (last: {self.last_callback_time}, now: {now})", "signal")
                    return True
                self.last_callback_time = now
                # Trigger the callback function if it exists
                if self.signal_callback:
                    self.signal_callback(signal_type="MODBUS_FRAME")
                return True
            
            # Check for other common Modbus function codes to the correct slave
            elif slave_id == expected_slave_id and function_code in [0x01, 0x02, 0x04, 0x05, 0x06, 0x0F, 0x10]:
                debug_print(f"Modbus frame for correct slave but different function - Slave: {slave_id}, Function: 0x{function_code:02X}", "signal")
                # Optionally trigger callback for any valid Modbus frame to correct slave
                # if self.signal_callback:
                #     self.signal_callback(signal_type="MODBUS_FRAME")
                return True  # Valid frame structure, even if not the expected function
            
            # Wrong slave ID or invalid function code
            else:
                if slave_id != expected_slave_id:
                    debug_print(f"Wrong slave ID: {slave_id} (expected: {expected_slave_id})", "signal")
                else:
                    debug_print(f"Unsupported function code: 0x{function_code:02X}", "signal")
                return False
            
        except Exception as e:
            debug_print(f"Error processing Modbus data: {e}", "errors")
            traceback.print_exc()
            return False
    
    def send_battery_quality_result(self, quality_result):
        """Send battery lid quality result via simple Modbus frame
        
        Frame structure (3 bytes total):
        - Slave ID (1 byte): From settings
        - Function code (1 byte): 0x03
        - Quality result (1 byte): 1 for GOOD, 0 for BAD
        
        Args:
            quality_result: Boolean or int - True/1 for GOOD, False/0 for BAD
            
        Returns:
            bool: True if data was sent successfully, False otherwise
        """
        if not SERIAL_AVAILABLE:
            debug_print("Cannot send quality result - pyserial not installed", "modbus")
            return False
        
        # Check if we have an open serial connection
        if self.serial_port is None or not self.serial_port.is_open:
            debug_print("Cannot send quality result - serial port not open", "modbus")
            debug_print("The signal detection system must be running to send data", "modbus")
            return False
            
        try:
            # Get Modbus parameters
            SLAVE_ADDRESS = self.settings.get("modbus_slave_id", 1)
            FUNCTION_CODE = 0x03  # Read Holding Registers
            
            # Convert quality result to integer
            result_value = 1 if quality_result else 0
            
            # Create the simple 3-byte frame
            frame = bytes([
                SLAVE_ADDRESS,    # Byte 0: Slave ID
                FUNCTION_CODE,    # Byte 1: Function code (0x03)
                result_value      # Byte 2: Quality result (1=GOOD, 0=BAD)
            ])
            
            # Send the frame
            self.serial_port.write(frame)
            
            # Print information about the sent data
            result_text = "GOOD" if result_value == 1 else "BAD"
            debug_print(f"Sent battery quality result via Modbus:", "modbus")
            debug_print(f"  Quality: {result_text} ({result_value})", "modbus")
            debug_print(f"  Slave ID: {SLAVE_ADDRESS}", "modbus")
            debug_print(f"  Frame size: {len(frame)} bytes", "modbus")
            
            # Print hexadecimal representation for debugging
            hex_repr = ' '.join([f'{b:02X}' for b in frame])
            debug_print(f"  Frame (hex): {hex_repr}", "modbus")
            
            return True
            
        except Exception as e:
            debug_print(f"Error sending battery quality result: {e}", "errors")
            traceback.print_exc()
            return False
            
    
    