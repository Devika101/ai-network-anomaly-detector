import pyshark
import pandas as pd

def extract_features(pcap_file, label, limit=5000):
    print(f" Reading {pcap_file} (label={label})...")
    cap = pyshark.FileCapture(pcap_file)  
    data = []

    count = 0
    for pkt in cap:
        if count >= limit:
            break
        try:
            
            time = float(pkt.sniff_time.timestamp())  
            protocol = pkt.highest_layer
            length = int(pkt.length)

            data.append([time, protocol, length, label])
            count += 1

            if count % 1000 == 0:
                print(f" {count} packets processed from {pcap_file}")
        except Exception:
            continue  
    cap.close()
    print(f" Finished {pcap_file} ({count} packets)\n")
    return data


normal_data = extract_features('normal.pcap', label=0, limit=5000)
attack_data = extract_features('attack_clean.pcap', label=1, limit=5000)


print(" Combining datasets...")
df = pd.DataFrame(normal_data + attack_data, columns=["time", "protocol", "length", "label"])


df.to_csv("network_traffic.csv", index=False)
print(" Saved as network_traffic.csv")
print(f" Total rows: {len(df)}")
print(" Done!")

