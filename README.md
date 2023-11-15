# Wind Speed Forecasting Using Deep Learning 
![](https://www.windsystemsmag.com/wp-content/uploads/2019/10/1019-CW-I1.jpg)
## Introduction
This project aims to predict wind speed for the next 24 hours using meteorological data, contributing to the efficiency of wind energy production. 
The data is sourced from the Israeli Meteorological Service (IMS) through a REST API. 
The project employs a Convolutional Neural Network-Long Short Term Memory (CNN-LSTM) model to forecast wind speed with improved accuracy.

## Data Source
The meteorological data for this project was a 10min WS (m/s) records obtained from the Israeli Meteorological Service (IMS) via a REST API. 
Access to this API requires a token, which can be requested on the IMS website.
link: https://ims.gov.il/en/ObservationDataAPI

## Models
I used two models in this project:

1. Baseline Model (Moving Average): A simple model to establish a baseline for comparison.
2. Main Model (CNN-LSTM):  
A deep learning model combining Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks.

## Repository Structure
**Data:** Contains one year of data from station 28, used as an example.
**Models:** Includes the specific CNN-LSTM model with its weights and architecture.
**Notebooks:**
1. EDA Notebook: For Exploratory Data Analysis.
2. Training Notebook: To train the models.  

**Scripts:**
1. IMS_API_data: Retrieves meteorological data from IMS.
2. WS: Contains all utility functions for the project.
3. main: The main script to run the project end-to-end.
## Adjustability
The project's code is designed to be flexible and can be easily adjusted to predict other meteorological variables besides wind speed.

## Installation and Usage
To set up this project:  
1. Clone the repository.
2. Install required dependencies (list any major libraries/frameworks).
3. Place the IMS API token file in the repository folder.
4. Run the main script to execute the project.

## Contributing
Contributions to the project are welcome. Please feel free to fork the repository and submit pull requests.

## Acknowledgments
Special thanks to the Israeli Meteorological Service for providing the data necessary for this project.