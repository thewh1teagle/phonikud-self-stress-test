import json
import matplotlib.pyplot as plt

# Read report.json
with open('report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

# Extract data
summary = report['summary']
models = list(summary['total_errors'].keys())
wer_values = [summary['average_wer'][model] for model in models]
error_counts = [summary['total_errors'][model] for model in models]

# Create bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(models, wer_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])

# Add text on top of bars
for i, (bar, wer, errors) in enumerate(zip(bars, wer_values, error_counts)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'WER: {wer:.3f}\nErrors: {errors}', 
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Word Error Rate (WER) by Model', fontsize=14, fontweight='bold')
plt.ylabel('WER', fontsize=12)
plt.xlabel('Models', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('wer_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Chart saved as 'wer_comparison.png'") 