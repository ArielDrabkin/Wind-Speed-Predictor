import pandas as pd
import requests
import json


def get_met_data(token, url):
    """
    Fetches meteorological data from the given URL.

    Parameters:
    - token (str): Authorization token to access the API.
    - url (str): The URL endpoint where the data resides.

    Returns:
    - dict: The parsed JSON data returned from the API.
    """
    headers = {
        'Authorization': token
    }

    response = requests.request("GET", url, headers=headers)
    data = json.loads(response.text.encode('utf8'))
    return data


def data_to_df(data):
    """
    Converts the meteorological data format into a DataFrame where each 
    'name' from the 'channels' becomes a column.

    Parameters:
    - data (dict): The meteorological data with 'datetime' and 'channels'.

    Returns:
    - DataFrame: A reshaped DataFrame where each 'name' becomes a column.
    """
    # Convert channels to dictionary {name: value}
    reshaped_data = []
    for entry in data['data']:
        datetime = entry['datetime']
        temp_dict = {'datetime': datetime}
        for channel in entry['channels']:
            temp_dict[channel['name']] = channel['value']
        reshaped_data.append(temp_dict)

    # Convert reshaped data to DataFrame
    df = pd.DataFrame(reshaped_data)

    # Drop the original 'channels' column which is now redundant
    df.drop(columns=['channels'], inplace=True, errors='ignore')
    return df


if __name__ == "__main__":
    with open('Token.json', 'r') as file:
        TOKEN = json.load(file)
    STATION = 28
    START_DATE = '2022/10/13'
    END_DATE = '2023/10/14'
    url = f"https://api.ims.gov.il/v1/envista/stations/{STATION}/data?from={START_DATE}&to={END_DATE}"
    data = get_met_data(TOKEN, url)
    df = data_to_df(data)
    df.to_csv('IMS_data.csv', index=False)
    print(df.head())
