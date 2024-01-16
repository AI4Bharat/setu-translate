import pandas as pd
from joblib import Parallel, delayed
 
# initialise data dictionary.
data_dict = {'CustomerID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              
             'Gender': ["Male", "Female", "Female", "Male",
                        "Male", "Female", "Male", "Male",
                        "Female", "Male"],
              
             'Age': [20, 21, 19, 18, 25, 26, 32, 41, 20, 19],
              
             'Annual Income(k$)': [10, 20, 30, 10, 25, 60, 70,
                                   15, 21, 22],
              
             'Spending Score': [30, 50, 48, 84, 90, 65, 32, 46,
                                12, 56]}
 
# Create DataFrame
data = pd.DataFrame(data_dict)
 
# Write to CSV file
data.to_csv("tmp/Customers.csv")
 
# Print the output.
print(data)


import pandas as pd
 
# read DataFrame
data = pd.read_csv("tmp/Customers.csv")

data_grouped = data.groupby(['Gender', 'Annual Income(k$)'])
 
for (Gender, Income), group in data.groupby(['Gender', 'Annual Income(k$)']):
    group.to_csv(f'{Gender} {Income}.csv', index=False)