"""
Microplastic Detection - Live Detection with IoT Sensors
==========================================================
Real-time microplastic detection using Raspberry Pi Camera with integrated
hardware sensors for comprehensive water quality monitoring.

Sensors:
    - DS18B20 Digital Temperature Sensor (1-Wire)
    - Turbidity Sensor (Analog via MCP3008 ADC, Channel 0)
    - TDS Sensor (Analog via MCP3008 ADC, Channel 1)

Hardware:
    - Raspberry Pi 4
    - Pi Camera Module / USB Microscope Camera
    - MCP3008 ADC (SPI interface)
    - DS18B20 Temperature Probe

Usage:
    python src/live_detect.py
    
Controls:
    Press 's' → Scan current frame + read sensors + log to CSV
    Press 'q' → Quit and save session
"""

import csv
import os
import time

import cv2
import geocoder
import spidev
from ultralytics import YOLO
from w1thermsensor import W1ThermSensor


# --- PATH & LOCATION SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
CSV_FILE = os.path.join(PROJECT_DIR, "microplastics_data.csv")
MODEL_PATH = os.path.join(PROJECT_DIR, "results", "weights", "best.pt")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_location():
    """Auto-detect location using IP-based geolocation."""
    print("Syncing with network location...")
    try:
        g = geocoder.ip("me")
        location = f"{g.city}, {g.state}" if g.city else "Pune Area (Estimated)"
        coordinates = f"{g.lat}, {g.lng}"
    except Exception:
        location = "Unknown Location"
        coordinates = "0.0, 0.0"
    print(f"Location Locked: {location}")
    return location, coordinates


def setup_spi():
    """Initialize SPI interface for MCP3008 ADC."""
    spi = spidev.SpiDev()
    spi.open(0, 0)
    spi.max_speed_hz = 1350000
    return spi


def read_analog_voltage(spi, channel):
    """
    Read analog voltage from MCP3008 ADC.

    Args:
        spi: SPI device instance
        channel: MCP3008 channel (0-7)

    Returns:
        Voltage reading (0 to 3.3V), rounded to 2 decimal places
    """
    if channel < 0 or channel > 7:
        return -1
    adc = spi.xfer2([1, (8 + channel) << 4, 0])
    data = ((adc[1] & 3) << 8) + adc[2]
    voltage = (data * 3.3) / 1024.0
    return round(voltage, 2)


def setup_temperature_sensor():
    """Initialize DS18B20 digital temperature sensor."""
    try:
        sensor = W1ThermSensor()
        print("✓ Temperature Sensor Found!")
        return sensor
    except Exception:
        print("✖ Temp Sensor NOT found! (Check wiring and 1-Wire settings)")
        return None


def init_csv(csv_file):
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(csv_file):
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Date_Time",
                "Location",
                "Coordinates",
                "Image_Name",
                "Microplastic_Count",
                "Temp_C",
                "Turbidity_V",
                "TDS_V",
            ])


def scan_and_log(frame, model, spi, temp_sensor, location, coordinates, csv_file):
    """
    Run YOLO detection + read sensors + log results to CSV.

    Args:
        frame: Current camera frame
        model: Loaded YOLO model
        spi: SPI device for ADC readings
        temp_sensor: W1ThermSensor instance (or None)
        location: String location name
        coordinates: String GPS coordinates
        csv_file: Path to output CSV
    
    Returns:
        Annotated frame with detection boxes
    """
    print("\nScanning Water Sample...")

    # 1. Run AI Model
    results = model(frame, conf=0.25)
    count = len(results[0].boxes)
    res_plotted = results[0].plot()
    cv2.imshow("LATEST_DETECTION", res_plotted)

    # 2. File info
    timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
    filename = f'scan_{time.strftime("%H%M%S")}.jpg'
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), res_plotted)

    # 3. Read Hardware Sensors
    current_temp = (
        round(temp_sensor.get_temperature(), 2) if temp_sensor else "Error"
    )
    turbidity_volts = read_analog_voltage(spi, 0)  # Channel 0
    tds_volts = read_analog_voltage(spi, 1)  # Channel 1

    # 4. Save to CSV
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp_str,
            location,
            coordinates,
            filename,
            count,
            current_temp,
            turbidity_volts,
            tds_volts,
        ])
        f.flush()

    print(
        f"✓ LOGGED: {count} particles | "
        f"Temp: {current_temp}°C | "
        f"Turbidity: {turbidity_volts}V | "
        f"TDS: {tds_volts}V"
    )

    return res_plotted


def main():
    """Main entry point for live detection with IoT sensors."""
    print("=" * 60)
    print("  🔬 Microplastic Detection - Live Scan Mode")
    print("  📡 With IoT Water Quality Sensors")
    print("=" * 60)

    # Setup
    location, coordinates = get_location()
    model = YOLO(MODEL_PATH)

    print("\nInitializing Water Sensors...")
    spi = setup_spi()
    temp_sensor = setup_temperature_sensor()
    init_csv(CSV_FILE)

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nMicroscope & Sensors Ready!")
    print("Press 's' to Scan & Log | Press 'q' to Quit\n")

    scan_count = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("LIVE_VIEW", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s"):
                scan_and_log(
                    frame, model, spi, temp_sensor, location, coordinates, CSV_FILE
                )
                scan_count += 1

            elif key == ord("q"):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        spi.close()

        print("\n" + "=" * 60)
        print("  📊 Session Summary")
        print("=" * 60)
        print(f"  Total scans:      {scan_count}")
        print(f"  Data saved to:    {CSV_FILE}")
        print(f"  Images saved to:  {OUTPUT_DIR}")
        print("=" * 60)
        print("Session ended. All data safely saved.")


if __name__ == "__main__":
    main()
