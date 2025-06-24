# AI-Based Network Traffic Anomaly Detection Using Mininet

🚀Project for detecting anomalous network traffic using simulated data.

## 💻 Tools & Tech
- Mininet (v2.3.0) in Ubuntu VM
- Python 3.12
- PyShark + tcpdump
- Scikit-learn: Isolation Forest & Random Forest
- Matplotlib for visualization
- Wireshark for packet inspection

## 📁 Folder Structure
- `extract_features.py`: Extracts features from .pcap files
- `train_model.py`: Trains models and generates metrics + graph
- `rf_model.pkl`: Saved Random Forest model (not committed)
- `network_traffic.csv`: Dataset (not committed)
- `requirements.txt`: List of required Python libraries

## 🧠 ML Models Used
- **Isolation Forest** (Unsupervised): Detects anomalies
- **Random Forest Classifier** (Supervised): High-accuracy prediction of attack/normal traffic

## ⚙️ Usage

```bash
# Activate environment
source netenv/bin/activate

# Extract features
python extract_features.py

# Train models
python train_model.py
