# System Architecture — Technical Description

> This document provides a detailed technical breakdown of the IoT-Integrated AI System for Microplastic Detection in Water. The system operates across four distinct phases, spanning hardware initialization through to real-time data visualization.

---

## Phase 1: Pre-Deployment & Initialization (Warming Up)

### Offline Model Training (on PC)
Before deployment, the YOLOv5-Tiny model is trained on a labeled microplastic dataset using a GPU-enabled workstation. The workflow:
1. **Annotate Dataset** — Label microplastic particles in water sample images using MakeSense.AI
2. **Train YOLOv5** — Fine-tune the pre-trained YOLOv5 Nano model on the custom dataset
3. **Export `best.pt` weights** — Extract the optimized model weights for edge deployment

### Hardware Layer (Raspberry Pi OS)
- **System Boot** → Load Debian OS & activate Python virtual environment (`venv`)
- **Protocol Boot** → Enable SPI (for MCP3008 ADC) & 1-Wire (for DS18B20 temperature sensor)
- **Load Model Weights** → Load `best.pt` (Tiny YOLOv5) weights into memory
- **Hardware Check** → Verify sensor connectivity; print error to terminal if sensors not found

---

## Phase 2: Signal Digitization & Protocol Bridging

Microprocessors like the Raspberry Pi operate purely on digital logic and lack native hardware for reading raw continuous voltages. This layer bridges that physical-to-digital gap.

### MCP3008 ADC (Analog-to-Digital Converter)
- **10-bit integrated circuit** that samples continuous analog voltages from Turbidity and TDS sensors
- Quantizes readings into discrete digital integers (ranging from 0 to 1023)

### The SPI Bus
- The MCP3008 transmits digital integers to the Pi using the **Serial Peripheral Interface (SPI)**
- High-speed, synchronous protocol utilizing four dedicated hardware lines: **MOSI, MISO, SCLK, and CE0**
- Enables microsecond-accurate data transfer

---

## Phase 3: The Edge Compute Architecture

The central nervous system of the project, designed specifically to eliminate cloud dependency, thereby removing network latency and preserving data privacy.

### Compute Node (Raspberry Pi 4 Model B)
- Powered by a **Quad-core ARM Cortex-A72** processor running Debian Linux
- Isolates all software dependencies within a Python Virtual Environment (`venv`)
- Ensures kernel-level stability while simultaneously handling hardware I/O operations and heavy matrix multiplications for the AI

---

## Phase 4: Machine Learning & Vision Pipeline

The inference engine responsible for identifying and quantifying microscopic anomalies within the spatial data.

### YOLOv5-Tiny Architecture (PyTorch)
- **YOLO (You Only Look Once)** — state-of-the-art, single-shot object detection model
- The "Tiny" variant was explicitly chosen because its reduced parameter count (fewer hidden layers) allows for rapid **CPU-bound inference** without requiring a dedicated discrete GPU

### OpenCV (cv2) Preprocessing
- Intercepts the raw camera frame
- Scales the pixel matrices to the exact dimensions expected by the YOLO tensor
- Normalizes the color channels

### Bounding Box Regression & NMS
- As the model detects particles, it draws localized bounding boxes
- To prevent counting the same particle twice, it applies **Non-Maximum Suppression (NMS)**

---

## Phase 5: The Execution Trigger & Parallel Processing

When the user presses the `'s'` key, three parallel pipelines activate:

### Thread A: Vision Pipeline
1. **Optical Capture (V4L2)** → Trigger camera module
2. **Preprocessing (OpenCV)** → Resize, normalize tensors
3. **SPI Bus Read** → MCP3008 digitizes Analog TDS & Turbidity
4. **1-Wire Bus Read** → Parse digital temp from `/sys/bus/w1/`
5. **Feature Extraction** → Generate particle count

### Thread B: Sensor Pipeline
1. **Polling Triggered** → Initiate hardware bus read
2. **SPI Bus Read (MCP3008)** → Digitize Analog TDS & Turbidity
3. **YOLOv5 Inference** → CPU-bound matrix ops
4. **Confidence Threshold** → `conf > 0.25`
5. **Telemetry Extraction** → Generate Temp_C, TDS_V values

### Thread C: Geospatial Pipeline
1. **Network Request (Geocoder)** → Ping router IP
2. **Spatial Extraction** → Generate location string (e.g., "Pune, Maharashtra")

---

## Phase 6: Data Unification & Serialization

After all three threads complete:
1. **Data Unification** → Merge: Timestamp, Count, Temp, Turbidity, TDS, Location
2. **Data Validation (Null Check)** → If any sensor fails, append `'Error'` string and proceed
3. **Serialization** → Open CSV in Append Mode (`'a'`)
4. **Local Storage** → Append row to `microplastics_data.csv`

---

## Phase 7: ETL & Visualization (Asynchronous)

The presentation layer runs independently on a separate machine:

1. **SAMBA (SMB) Network Bridge** → Broadcasts CSV directory over local WLAN
2. **Zero-Latency Read Access** → Power BI connects to the shared CSV
3. **Query Trigger (Power Query)** → Power BI executes refresh request
4. **ETL Pipeline** → Extract raw CSV data, transform strings to decimals, load into data model
5. **UI Render** → Update KPI cards, recalculate pie charts, redraw geo mapping
6. **Real-Time End-User Dashboard Updated**

---

## System Flowchart

> See `assets/Flowchart.png` for the complete visual system architecture diagram covering all phases from offline model training through real-time dashboard visualization.

![System Flowchart](../assets/Flowchart.png)
