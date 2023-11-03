import pandas as pd
import requests
import urllib.parse

def get_elevation(lat, long):
    """
      Fetch the elevation (in meters) for a given latitude and longitude using the Open-Meteo API.

      Args:
          lat (float): Latitude of the location for which elevation is required.
          long (float): Longitude of the location for which elevation is required.

      Returns:
          float: Elevation in meters for the given latitude and longitude based on the Copernicus DEM 2021 release GLO-90 with 90 meters resolution.
    """
    # Base URL for the Open-Meteo historical weather API
    ELEVATION_API_BASE_URL = "https://api.open-meteo.com/v1/elevation?"

    # Construct the URL parameters for the API request
    ELEVATION_API_URL_PARAMS = {
        "latitude": lat,
        "longitude": long
    }

    # Create the full URL for the API request
    url = ELEVATION_API_BASE_URL + urllib.parse.urlencode(ELEVATION_API_URL_PARAMS, doseq=True)

    # Make the API request
    response = requests.get(url, timeout=5)

    # Get the real elevation
    ELEVATION = response.json()

    return ELEVATION['elevation']


def get_met_data(lat_list, long_list, start_date, end_date):
    """
    Fetch historical weather data from the Open-Meteo API for given locations and date ranges.

    Parameters:
    - lat_list (list of str): List of latitudes for desired locations.
    - long_list (list of str): List of longitudes for desired locations.
    - start_date (list of str): List of start dates in 'YYYY-MM-DD' format.
    - end_date (list of str): List of end dates in 'YYYY-MM-DD' format.

    Returns:
    - DataFrame: Weather data for the given locations and date ranges.

    For available variables, refer to:
    https://open-meteo.com/en/docs/historical-weather-api#models=best_match
    """

    WEATHER_API_BASE_URL = "https://archive-api.open-meteo.com/v1/archive?"
    VARIABLES = ["precipitation_sum", "et0_fao_evapotranspiration", "weathercode"]
    met_data = pd.DataFrame()

    for lat, long, start, end in zip(lat_list, long_list, start_date, end_date):
        WEATHER_API_URL_PARAMS = {
            "latitude": lat,
            "longitude": long,
            "start_date": start,
            "end_date": end,
            "daily": VARIABLES,
            "timezone": "GMT"
        }

        url = WEATHER_API_BASE_URL + urllib.parse.urlencode(WEATHER_API_URL_PARAMS, doseq=True)
        response = requests.get(url)
        data = response.json()
        current_data = pd.DataFrame.from_dict(data["daily"])
        current_data["lat"] = lat
        current_data["long"] = long
        met_data = pd.concat([met_data, current_data], ignore_index=True)

    return met_data


if __name__ == "__main__":
    lat_list = ["37.71008935", "52.1250655"]
    long_list = ["-122.196217", "4.7563068"]
    start_date = ["2023-04-01", "2023-04-01"]
    end_date = ["2023-04-02", "2023-04-02"]
    data = get_met_data(lat_list, long_list, start_date, end_date)
    print(data)