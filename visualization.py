import pandas as pd
import matplotlib.pyplot as plt

results_path = 'runs/detect/train/results.csv'

results = pd.read_csv(results_path)

results.columns = results.columns.str.strip()
print("CSV 파일의 컬럼 목록:", results.columns)

plt.figure(figsize=(10, 6))
plt.plot(results['epoch'], results['metrics/precision(B)'], label='Precision')
plt.plot(results['epoch'], results['metrics/recall(B)'], label='Recall')

plt.xlabel('Epochs')
plt.ylabel('Score')
plt.title('Model Performance Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
