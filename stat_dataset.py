import os
from torchtitan.datasets.alfred_dataset import ALFREDDataset, AlfredDataLoader
from torchtitan.datasets.infini_thor_dataset import InfiniTHORDataset
from torchtitan.datasets.alfred_dataset_validact_pred import ALFREDDataset, AlfredValidActDataLoader
from torchtitan.datasets.processor import build_hf_processor
import json
from collections import defaultdict

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"

processor = build_hf_processor(model_name)

traj_data_dir = '/data/alfred/train_validact/traj_part1'
img_data_dir = '/data/alfred/train_validact/img_tar'

valact_stat = defaultdict(int)
bestact_stat = defaultdict(int)
for filename in os.listdir(traj_data_dir):
    file_path = os.path.join(traj_data_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        traj = json.loads(f.read())

    for pair, valact in traj['validact_pair'].items():
        act_list = [x['action'] for x in valact]
        act_list.sort()
        _valact_key = tuple(act_list)
        valact_stat[_valact_key] += 1
    for best_act in traj['generated_actions']:
        bestact_stat[best_act['action']] += 1

from pprint import pprint

# Print valact statistics
print("=" * 50)
print("VALID ACTION STATISTICS")
print("=" * 50)
pprint(valact_stat)

# Calculate total count for valact
total_count = sum(valact_stat.values())

# Print statistics with percentages
print(f"\nTotal count: {total_count}\n")
for key, count in sorted(valact_stat.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_count) * 100
    print(f"{key}: {count} ({percentage:.2f}%)")

# Print bestact statistics
print("\n" + "=" * 50)
print("BEST ACTION STATISTICS")
print("=" * 50)
pprint(bestact_stat)

# Calculate total count for bestact
total_bestact_count = sum(bestact_stat.values())

# Print statistics with percentages
print(f"\nTotal count: {total_bestact_count}\n")
for key, count in sorted(bestact_stat.items(), key=lambda x: x[1], reverse=True):
    percentage = (count / total_bestact_count) * 100
    print(f"{key}: {count} ({percentage:.2f}%)")

# Create donut plot for best actions
import matplotlib.pyplot as plt

# Sort by count for better visualization
sorted_bestact = sorted(bestact_stat.items(), key=lambda x: x[1], reverse=True)
labels = [item[0] for item in sorted_bestact]
sizes = [item[1] for item in sorted_bestact]

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Create donut chart
colors = plt.cm.Set3(range(len(labels)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                    startangle=90, colors=colors,
                                    pctdistance=0.85)

# Create pie chart
colors = plt.cm.Set3(range(len(labels)))
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                    startangle=90, colors=colors,
                                    pctdistance=0.85)

# Styling
plt.setp(autotexts, size=9, weight="bold")
plt.setp(texts, size=10)

ax.set_title('Best Action Distribution for Navigation', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()

# Save the plot
plt.savefig('best_action_pie_plot.png', dpi=300, bbox_inches='tight')
print(f"\nDonut plot saved as 'best_action_donut_plot.png'")
plt.show()
