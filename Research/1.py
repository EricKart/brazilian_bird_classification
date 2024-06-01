import matplotlib.pyplot as plt
import numpy as np

# # # Define the model types and their accuracies at different periods
models = ["SVM", "CNN", "RNN"]
accuracy_2010 = [60, 70, 65]  # Hypothetical accuracies in 2010
accuracy_2015 = [65, 85, 80]  # Hypothetical accuracies in 2015
accuracy_2020 = [70, 90, 88]  # Hypothetical accuracies in 2020

# # # Set the positions and width for the bars
pos = np.arange(len(models))
bar_width = 0.2

# Plotting the bars
fig, ax = plt.subplots(figsize=(10, 6))

plt.bar(pos - bar_width, accuracy_2010, bar_width, label="2010", color="skyblue")
plt.bar(pos, accuracy_2015, bar_width, label="2015", color="orange")
plt.bar(pos + bar_width, accuracy_2020, bar_width, label="2020", color="green")

# # # Add some final touches to the plot
ax.set_ylabel("Accuracy (%)")
ax.set_title("Accuracy Improvements of ML Models for Bird Classification Over Time")
ax.set_xticks(pos)
ax.set_xticklabels(models)
ax.legend()

# Display the plot
plt.tight_layout()
plt.show()


import pandas as pd
from tabulate import tabulate

# # # Define the data for the table
data = {
    "Approach": [
        "Manual Identification",
        "Acoustic Signal Processing",
        "Machine Learning Classification",
    ],
    "Key Characteristics": [
        "- Relies on visual and auditory cues.\n- Dependent on expert knowledge.\n- Utilizes field guides for reference.",
        "- Uses signal processing techniques on audio recordings.\n- Focuses on features like frequency, duration, and pitch.\n- Early computational approach to bird identification.",
        "- Employs supervised and unsupervised learning.\n- Analyzes audio and/or visual data.\n- Leverages complex algorithms like CNNs and RNNs.",
    ],
    "Advantages": [
        "- High accuracy with experienced observers.\n- Rich in contextual understanding.",
        "- Automates part of the identification process.\n- Can handle larger datasets than manual methods.",
        "- Highly scalable for big data applications.\n- Continuously improves with more data.",
    ],
    "Limitations": [
        "- Time-consuming and labor-intensive.\n- Limited by human perceptual abilities.\n- Not scalable for large datasets.",
        "- Requires detailed pre-processing of data.\n- Limited by the quality of recordings.\n- May struggle with complex or overlapping calls.",
        "- Requires large, labeled datasets for training.\n- Model interpretability can be challenging.\n- Initial setup and training are resource-intensive.",
    ],
}

# # # Create a DataFrame
df = pd.DataFrame(data)

# Use tabulate to print the table
print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))
import matplotlib.pyplot as plt
import pandas as pd

# # Data preparation
data = {
    "Methodology": ["Traditional", "Machine Learning", "Hybrid Approaches"],
    "Accuracy (%)": [60, 80, 95],
    "Computational Efficiency (1-10)": [2, 6, 8],
}
df = pd.DataFrame(data)

# # # Plotting
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
df.plot(
    kind="bar", x="Methodology", y="Accuracy (%)", ax=ax1, color="skyblue", position=1
)
df.plot(
    kind="bar",
    x="Methodology",
    y="Computational Efficiency (1-10)",
    ax=ax2,
    color="orange",
    position=0,
)

ax1.set_ylabel("Accuracy (%)", color="skyblue")
ax2.set_ylabel("Computational Efficiency (1-10)", color="orange")
plt.title("Comparative Analysis of Bird Detection Methodologies")
plt.show()
import matplotlib.pyplot as plt

# Simulated accuracies
methods = ["Traditional Methods", "Machine Learning Methods", "Hybrid Approaches"]
accuracies = [0.6, 0.75, 0.9]  # Example accuracies

plt.figure(figsize=(10, 6))
plt.bar(methods, accuracies, color=["blue", "green", "red"])
plt.xlabel("Methodology")
plt.ylabel("Accuracy")
plt.title("Comparative Analysis of Bird Detection Methodologies")
plt.ylim(0, 1)
plt.show()
# Simulated model footprint sizes (in MB)
footprints = [0, 50, 150]  # Example footprints

plt.figure(figsize=(10, 6))
plt.bar(methods, footprints, color=["blue", "green", "red"])
plt.xlabel("Methodology")
plt.ylabel("Model Footprint Size (MB)")
plt.title("Model Footprint Size by Methodology")
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# # Sample data representing years and classification accuracy before and after feature engineering
years = np.array([2010, 2012, 2014, 2016, 2018, 2020, 2022])
accuracy_before_fe = np.array([70, 72, 75, 78, 82, 85, 87])
accuracy_after_fe = np.array([70, 73, 77, 81, 85, 88, 90])

# Creating the plot
plt.figure(figsize=(12, 7))
plt.plot(
    years,
    accuracy_before_fe,
    label="Before Feature Engineering",
    marker="o",
    linestyle="--",
    color="red",
)
plt.plot(
    years,
    accuracy_after_fe,
    label="After Feature Engineering",
    marker="o",
    linestyle="-",
    color="green",
)

plt.xlabel("Year")
plt.ylabel("Accuracy (%)")
plt.title("Impact of Feature Engineering on Bird Classification Accuracy Over Time")
plt.legend()
plt.grid(True)
plt.ylim(65, 95)

# Displaying the plot
plt.show()
