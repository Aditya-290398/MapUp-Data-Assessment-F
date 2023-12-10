import pandas as pd

def calculate_distance_matrix(df):
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: Distance matrix.
    """
    # Extract unique toll locations
    unique_tolls = df[['toll_booth_A', 'toll_booth_B']].stack().unique()

    # Create a dictionary to store distances between toll locations
    distances = {toll: {toll: 0.0 for toll in unique_tolls} for toll in unique_tolls}

    # Populate the distances dictionary with known distances
    for index, row in df.iterrows():
        distances[row['toll_booth_A']][row['toll_booth_B']] = row['distance']
        distances[row['toll_booth_B']][row['toll_booth_A']] = row['distance']

    # Apply Floyd-Warshall algorithm to find the shortest paths
    for k in unique_tolls:
        for i in unique_tolls:
            for j in unique_tolls:
                if distances[i][k] + distances[k][j] < distances[i][j]:
                    distances[i][j] = distances[i][k] + distances[k][j]

    # Create a DataFrame from the distances dictionary
    distance_matrix = pd.DataFrame.from_dict(distances, orient='index', columns=unique_tolls)

    return distance_matrix

# Example usage:
if __name__ == "__main__":
    # Assuming dataset-3.csv is in the current working directory
    file_path = "dataset-3.csv"

    # Read the CSV file into a DataFrame
    dataset = pd.read_csv(file_path)

    # Call the calculate_distance_matrix function
    result_matrix = calculate_distance_matrix(dataset)

    # Print the resulting distance matrix
    print(result_matrix)


def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): Distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to store the unrolled data
    unrolled_data = []

    # Iterate through columns and rows of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip the case where id_start and id_end are the same
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df

# Example usage:
if __name__ == "__main__":
    # Assuming result_matrix is the distance matrix obtained from the previous question
    # Call the unroll_distance_matrix function
    unrolled_result = unroll_distance_matrix(result_matrix)

    # Print the resulting unrolled DataFrame
    print(unrolled_result)

def unroll_distance_matrix(distance_matrix):
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        distance_matrix (pandas.DataFrame): Distance matrix.

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Initialize an empty list to store the unrolled data
    unrolled_data = []

    # Iterate through columns and rows of the distance matrix
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Skip the case where id_start and id_end are the same
            if id_start != id_end:
                distance = distance_matrix.loc[id_start, id_end]
                unrolled_data.append([id_start, id_end, distance])

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df

def find_ids_within_ten_percentage_threshold(df, reference_id):
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame): Input DataFrame.
        reference_id (int): Reference ID for calculating the average distance.

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Use unroll_distance_matrix to get the unrolled DataFrame
    unrolled_df = unroll_distance_matrix(df)

    # Filter DataFrame for the reference_id
    reference_data = unrolled_df[unrolled_df['id_start'] == reference_id]

    # Calculate the average distance for the reference_id
    reference_avg_distance = reference_data['distance'].mean()

    # Calculate the lower and upper bounds for the threshold (10%)
    lower_bound = reference_avg_distance - 0.1 * reference_avg_distance
    upper_bound = reference_avg_distance + 0.1 * reference_avg_distance

    # Filter DataFrame based on the threshold
    filtered_df = unrolled_df[(unrolled_df['id_start'] != reference_id) & (unrolled_df['distance'] >= lower_bound) & (unrolled_df['distance'] <= upper_bound)]

    # Get unique values from the 'id_start' column and sort them
    result_ids = sorted(filtered_df['id_start'].unique())

    # Create a DataFrame with the result_ids
    result_df = pd.DataFrame({'id_start': result_ids})

    return result_df

# Example usage:
if __name__ == "__main__":
    # Assuming result_matrix is the distance matrix obtained from Question 1
    # Call the find_ids_within_ten_percentage_threshold function
    result_within_threshold = find_ids_within_ten_percentage_threshold(result_matrix, reference_id=1001400)

    # Print the resulting DataFrame
    print(result_within_threshold)




import pandas as pd

def calculate_toll_rate(df):
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with toll rates for each vehicle type.
    """
    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Iterate through rate_coefficients and create new columns in the DataFrame
    for vehicle_type, coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * coefficient

    return df

# Example usage:
if __name__ == "__main__":
    # Assuming unrolled_result is the DataFrame obtained from the previous question
    # Call the calculate_toll_rate function
    result_with_toll_rate = calculate_toll_rate(unrolled_result)

    # Print the resulting DataFrame
    print(result_with_toll_rate)



import pandas as pd
from datetime import time

def calculate_time_based_toll_rates(df):
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame): Input DataFrame.

    Returns:
        pandas.DataFrame: DataFrame with time-based toll rates.
    """
    # Define discount factors for different time intervals and days
    discount_factors = {
        'weekday': {
            '00:00:00-10:00:00': 0.8,
            '10:00:00-18:00:00': 1.2,
            '18:00:00-23:59:59': 0.8
        },
        'weekend': {
            '00:00:00-23:59:59': 0.7
        }
    }

    # Iterate through discount factors and create new columns in the DataFrame
    for day_type in ['weekday', 'weekend']:
        for time_range, discount_factor in discount_factors[day_type].items():
            start_time_str, end_time_str = time_range.split('-')
            start_time = time.fromisoformat(start_time_str)
            end_time = time.fromisoformat(end_time_str)

            # Filter DataFrame based on the day type and time range
            filtered_df = df[(df['start_day_type'] == day_type) & (df['start_time'] >= start_time) & (df['end_time'] <= end_time)]

            # Apply the discount factor to vehicle columns
            for vehicle_type in ['moto', 'car', 'rv', 'bus', 'truck']:
                df.loc[filtered_df.index, vehicle_type] *= discount_factor

    return df

# Example usage:
if __name__ == "__main__":
    # Assuming result_with_toll_rate is the DataFrame obtained from the previous question
    # Add columns for start_day_type, start_time, end_day_type, and end_time
    result_with_toll_rate['start_day_type'] = 'weekday'
    result_with_toll_rate['end_day_type'] = 'weekend'
    result_with_toll_rate['start_time'] = time.fromisoformat('00:00:00')
    result_with_toll_rate['end_time'] = time.fromisoformat('23:59:59')

    # Call the calculate_time_based_toll_rates function
    result_with_time_based_toll_rates = calculate_time_based_toll_rates(result_with_toll_rate)

    # Print the resulting DataFrame
    print(result_with_time_based_toll_rates)

