import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Set display format
pd.options.display.float_format = '{:,.2f}'.format

# Register matplotlib converters
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Read the dataset
data = pd.read_csv('cost_revenue_dirty.csv')

# Create a bubble graph with advanced features
plt.figure(figsize=(14, 10))
sns.set_theme(style="whitegrid")

# Scatter plot with size based on revenue, color based on cost, and marker style based on profit
sns.scatterplot(x='Cost', y='Revenue', size='Revenue', hue='Cost', style='Profit', data=data,
                palette='viridis', sizes=(50, 500), alpha=0.7, markers=True)

# Regression line
regression_model = LinearRegression()
regression_model.fit(data[['Cost']], data['Revenue'])
plt.plot(data['Cost'], regression_model.predict(data[['Cost']]), color='red', linewidth=2, linestyle='dashed',
         label='Regression Line')

# Customize the plot
plt.title('Advanced Bubble Graph: Cost vs Revenue', fontsize=18)
plt.xlabel('Cost', fontsize=14)
plt.ylabel('Revenue', fontsize=14)
plt.legend(title='Legend', bbox_to_anchor=(1, 1), loc='upper left', fontsize=12)

# Add annotations for top-performing points
top_performers = data.nlargest(5, 'Revenue')
for index, row in top_performers.iterrows():
    plt.annotate(f'{row["Product"]}\n${row["Revenue"]:.2f}M', (row['Cost'], row['Revenue']),
                 textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='green')

# Add a colorbar
colorbar = plt.colorbar()
colorbar.set_label('Cost', rotation=270, labelpad=15, fontsize=12)

# Display grid with a different linestyle
plt.grid(True, linestyle='--', alpha=0.5)

# Save the plot as an image
plt.savefig('advanced_bubble_graph.png')

# Show the plot
plt.show()
