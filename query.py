import requests
import pandas as pd
import time
from io import StringIO
from datetime import datetime, timedelta

"""
Script that runs Space-Track queries using the websites API.

- Downloads historical TLE data based on input parameters.
- Filters downloaded data to include only most recent TLE for each NORAD_CAT_ID.

* Modify the USERNAME and PASSWORD variables in run_query() for this to run *
"""

##########################################################################################
######################################    MODIFY    ######################################
##########################################################################################

# Define the number of periods and step size (in days)
num_periods = 5
# Time range for each query, in days --- DO NOT GO ABOVE 2.0
step_size = 2.0
# Define the time of interest (time transient was observed)
start_time = 60697.36836711
# Name output file
output_file = "tle_data.csv"

##########################################################################################

def mjd_to_datetime(mjd):
    # MJD epoch: midnight on November 17, 1858
    mjd_epoch = datetime(1858, 11, 17)
    # Convert MJD to datetime
    return mjd_epoch + timedelta(days=mjd)

start_time = mjd_to_datetime(start_time)

def get_days_since():
    # Compute the number of days since start_time for querrying purposes
    end_time = datetime.now()
    d = end_time-start_time

    return d.days + d.seconds / (24 * 3600)

def parse_tle_epoch(tle_line1):
    """
    Parse the epoch from the first line of a TLE and convert it to a UTC datetime object.
    """
    year = int(tle_line1[18:20])
    year = 2000 + year if year < 57 else 1900 + year  # Handle Y2K
    day_of_year = float(tle_line1[20:32])
    epoch = datetime(year, 1, 1) + timedelta(days=day_of_year - 1)
    return epoch

def run_query():
    # Space-Track credentials
    USERNAME = 'INSERT_USERNAME_HERE'
    PASSWORD = 'INSERT_PASSWORD_HERE'

    # Create a session and authenticate
    session = requests.Session()
    login_url = 'https://www.space-track.org/ajaxauth/login'
    login_payload = {
        'identity': USERNAME,
        'password': PASSWORD
    }

    response = session.post(login_url, data=login_payload)
    if response.status_code != 200:
        print(f"Failed to log in: HTTP {response.status_code}")
        print(response.text)  # Debugging info
        exit()
    else:
        print("Successfully logged in!")

    # Base URL for the Space-Track API
    BASE_URL = 'https://www.space-track.org/basicspacedata/query/class/gp_history'

    # Storage for results
    all_data = []

    # Query variable for epoch 
    days_since = get_days_since()

    # Loop to query consecutive periods
    for i in range(num_periods):
        # Calculate the start and end for this period
        start = days_since + i * step_size
        end = start + step_size

        # Construct the query for this period
        query = f"{BASE_URL}/OBJECT_TYPE/PAYLOAD/EPOCH/%3Enow-{end}%2C%3Cnow-{start}/orderby/NORAD_CAT_ID%20asc/format/csv/emptyresult/show"
        print(f"Fetching data for period: now-{end} to now-{start}")

        # Send the request using the authenticated session
        response = session.get(query)

        # Check if the request was successful
        if response.status_code == 200:
            # Load the data into a DataFrame
            if response.text.strip():  # Ensure data is not empty
                period_data = pd.read_csv(StringIO(response.text), low_memory=False)
                all_data.append(period_data)
                print(f"Successfully fetched data for period: now-{start} to now-{end}")
                print(f"Periods completed: {i+1}")
            else:
                print(f"No data for period: now-{start} to now-{end}")
        else:
            print(f"Failed to fetch data for period: now-{start} to now-{end}")
            print(f"HTTP Status Code: {response.status_code}")
            print(response.text)  # Debugging info
        
        # Sleep briefly to avoid hitting rate limits
        time.sleep(2)

    print("All data successfully downloaded")

    return all_data

def process_queries(all_data):
    # Combine all data into a single DataFrame
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=True)

        # Parse TLE_EPOCH from TLE_LINE1
        combined_data['TLE_EPOCH'] = combined_data['TLE_LINE1'].apply(parse_tle_epoch)

        # Filter the data to retain rows with the highest FILE number for each NORAD_CAT_ID
        filtered_data = (
            combined_data.loc[combined_data.groupby('NORAD_CAT_ID')['FILE'].idxmax()]
            .reset_index(drop=True)
        )

        # Calculate the age of each TLE relative to end_time in days
        filtered_data['TLE_AGE'] = (start_time - filtered_data['TLE_EPOCH']).dt.total_seconds() / (24 * 3600)

        # Save the filtered data to a new CSV file
        filtered_data.to_csv(output_file, index=False)
        print("Filtered data with TLE age saved to " + output_file)
    else:
        print("No data fetched across all periods.")

# Start data download from Space-Track
all_data = run_query()
# Combine and process data from each query
process_queries(all_data)

