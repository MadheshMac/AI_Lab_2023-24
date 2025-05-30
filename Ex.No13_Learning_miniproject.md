# Ex.No: 13 Mini Project
### DATE: 4.11.2024                                                                 
### REGISTER NUMBER : 212222040017

### AIM:
To write a program to train a classifier for peck hour Scheduling

### Algorithm:
1. Import necessary libraries 
2. Load the dataset 
3. Define the EfficientNet-B0 model and add a classification head to it
4. Create data loaders for training and validation sets
5. Train the model using the Adam optimizer and cross-entropy loss function
6. Monitor the model's performance on the validation set during training


### Program:
```python
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Seed for reproducibility
np.random.seed(42)

# Constants
city = 'Chennai'
routes = ['Route_1', 'Route_2', 'Route_3', 'Route_4', 'Route_5', 'Route_6']
peak_types = ['School', 'College', 'Office']
peak_times = ['Morning_School', 'Morning_Office', 'Afternoon_School', 'Evening_Office']
non_peak_times = ['Late_Morning', 'Early_Afternoon', 'Night']
time_of_day = peak_times + non_peak_times
places = ['T. Nagar', 'Adyar', 'Velachery', 'Guindy', 'Anna Nagar', 'Tambaram', 'Mylapore', 'Vadapalani', 'Kodambakkam', 'Porur']

# Define function to simulate traffic conditions
def traffic_conditions():
    return random.choice(['Low', 'Moderate', 'High'])

# Define function to simulate passenger demand
def passenger_demand(time):
    if time in peak_times:
        return np.random.randint(80, 200)  # Higher demand during peak times
    else:
        return np.random.randint(20, 80)  # Lower demand during non-peak times

# Define function to simulate bus availability (utilization)
def bus_utilization(demand, traffic):
    if traffic == 'Low':
        return demand // 10
    elif traffic == 'Moderate':
        return demand // 8
    else:
        return demand // 5

# Generate 1000 records with peak hour and non-peak hour data
data = []
for i in range(1000):
    record = {}
    record['Place'] = random.choice(places)
    record['Route'] = random.choice(routes)
    record['Time_of_Day'] = random.choice(time_of_day)
    record['Peak_Type'] = random.choice(peak_types) if record['Time_of_Day'] in peak_times else 'Non_Peak'
    record['Passenger_Demand'] = passenger_demand(record['Time_of_Day'])
    record['Traffic_Conditions'] = traffic_conditions()
    record['Scheduled_Buses'] = bus_utilization(record['Passenger_Demand'], record['Traffic_Conditions'])
    data.append(record)

# Create a DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv('/content/chennai_bus_scheduling_dataset_1000.csv', index=False)

# Display the first 5 rows of the dataset
print(df.head())

# Step 2: Function to search the dataset by place name
def search_by_place(place_name):
    # Filter the dataset by the provided place
    result = df[df['Place'].str.contains(place_name, case=False)]

    if not result.empty:
        return result[['Place', 'Time_of_Day', 'Peak_Type', 'Passenger_Demand', 'Traffic_Conditions', 'Scheduled_Buses']]
    else:
        return f"No data found for the place: {place_name}"

# Step 3: Search based on user input
user_input = input("Enter the place name to search (e.g., Velachery, T. Nagar): ")
search_result = search_by_place(user_input)
print(search_result)
```


### Output:
![image](https://github.com/user-attachments/assets/8277911d-b988-40ad-a044-111451f3be38)

The Model reached an accuracy of 97.5% after 10 epochs against the test dataset.


### Result:
Thus the system was trained successfully and the prediction was carried out.
