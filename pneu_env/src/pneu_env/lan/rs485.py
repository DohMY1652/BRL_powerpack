import serial

# Define the serial port and settings
port = '/dev/tty.usbserial'  # Change this to your specific port
baudrate = 9600
timeout = 1  # 1 second timeout for reading

# Initialize the serial connection
ser = serial.Serial(port, baudrate, timeout=timeout)

def send_data(data):
    """Send data over RS485."""
    if ser.is_open:
        ser.write(data.encode())
        print(f"Sent: {data}")
    else:
        print("Serial port not open")

def receive_data():
    """Receive data over RS485."""
    if ser.is_open:
        data = ser.readline().decode().strip()
        print(f"Received: {data}")
        return data
    else:
        print("Serial port not open")
        return None

def main():
    # Example usage
    send_data("Hello, RS485!")
    response = receive_data()
    print(f"Response: {response}")

if __name__ == "__main__":
    try:
        main()
    finally:
        if ser.is_open:
            ser.close()