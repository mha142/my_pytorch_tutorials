import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files
data1 = pd.read_csv('cascaded.csv')
data2 = pd.read_csv('linear2.csv')

# Plot the data using 'Step' as the X-axis and 'Value' as the Y-axis
plt.figure(figsize=(8, 5))
plt.plot(data1['Step'], data1['Value'], label='Dataset 1', marker='o')
plt.plot(data2['Step'], data2['Value'], label='Dataset 2', marker='x')

# Customize the plot
plt.title('Comparison of Two CSV Files')
plt.xlabel('Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

