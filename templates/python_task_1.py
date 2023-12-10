import pandas as pd

def generate_car_matrix(df):
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Assuming df has columns 'id_1', 'id_2', and 'car'
    
    # Pivot the DataFrame to create the initial matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car')

    # Fill NaN values with 0
    car_matrix = car_matrix.fillna(0)

    # Replace diagonal values with 0
    for i in range(min(car_matrix.shape)):
        car_matrix.iloc[i, i] = 0.0

    return car_matrix

# Load the dataset from CSV
file_path = 'dataset-1.csv'
dataset_df = pd.read_csv(file_path)

# Generate the car matrix
result_matrix = generate_car_matrix(dataset_df)

# Display or use the resulting matrix
print(result_matrix)


def get_type_count(df):
    """
    Returns the count of occurrences for each car_type category.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: Dictionary containing counts for 'low', 'medium', and 'high' car_type categories.
    """
    # Add a new categorical column 'car_type' based on values of the column 'car'
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each 'car_type' category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    type_counts = dict(sorted(type_counts.items()))

    return type_counts

# Load the dataset from CSV
file_path = 'dataset-1.csv'
dataset_df = pd.read_csv(file_path)

# Get the count of occurrences for each car_type category
result_counts = get_type_count(dataset_df)

# Display or use the resulting dictionary
print(result_counts)

def filter_routes(df):
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame): DataFrame containing route information.

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Calculate the average 'truck' values for each route
    average_truck_values = df.groupby('route')['truck'].mean()

    # Filter routes with average 'truck' values greater than 7
    selected_routes = average_truck_values[average_truck_values > 7].index.tolist()

    return selected_routes

def filter_routes(dataframe):
    """
    Filters routes based on the average of the 'truck' column.

    Args:
        dataframe (pandas.DataFrame): Input DataFrame.

    Returns:
        list: Sorted list of values of the 'route' column.
    """
    # Calculate the average of the 'truck' column for each route
    avg_truck_per_route = dataframe.groupby('route')['truck'].mean()

    # Filter routes where the average truck value is greater than 7
    selected_routes = avg_truck_per_route[avg_truck_per_route > 7].index.tolist()

    # Sort the list of selected routes
    sorted_routes = sorted(selected_routes)

    return sorted_routes

# Example usage:
if __name__ == "__main__":
    # Assuming dataset-1.csv is in the current working directory
    file_path = "dataset-1.csv"

    # Read the CSV file into a DataFrame
    dataset = pd.read_csv(file_path)

    # Call the filter_routes function
    result = filter_routes(dataset)

    # Print the sorted list of routes
    print(result)

def multiply_matrix(matrix):
    """
    Modifies matrix values based on custom conditions.

    Args:
        matrix (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Apply custom conditions to modify values
    modified_matrix = matrix.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage:
if __name__ == "__main__":
    # Assuming the result DataFrame from Question 1 is stored in the variable 'result'
    # Modify the result DataFrame using multiply_matrix function
    modified_result = multiply_matrix(result)

    # Print the modified DataFrame
    print(modified_result)
