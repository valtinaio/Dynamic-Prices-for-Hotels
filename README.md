# Dynamic-Prices-for-Hotels
Optimization Algorithm to maximize hotel revenues based on the current demand level using a Random Forest.

## Requirements
Python Version 3.10 or newer.  
csv Database  
See requirements.txt

## Usage of hotel_po
This library can be used to optimize prices of hotels with three different room classes for arrival days in the future. The main scope is to predict the demand of a hotel room with a Random Forest Regressor based on different features (including the price) whereby for a certain date in the future a revenue-maximizing price can be found for each room class. For doing so 5 different classes were created with a pipeline order:

**Data**: A class that imports and checks the data.  
**Overview**: A class that creates different overview possibilities of the data.  
**Features**: A class that creates optimized features for a Random Forest Regressor.  
**ModelRF**: A class that creates an optimized Random Forest Regressor to predict demand.  
**PriceOptimization**: A class that optimizes prices for a given future date maximizing the revenue.

-> Each step will require an instance of one of the classes above to work. Otherwise useful error-messages will arise.  
-> For each class- and instance-method short informative docstrings are provided.  
-> A main() function was implemented to run the complete basic pipeline for a certain data-base and future date automatically. A summary-result will be printed directly. Returns None though.  
-> A demonstration of the package in use can be found in demo_of_project.html

## Why hotel_po is useful
This package is a powerful tool for hotels to maximize their revenue by dynamic prices. Based on three room types the package allows the hotel owner to predict their demand and more importantly to optimize the prices for a revenue-maximation. This way especially in low-season the price can be used as an instrument to stimulate demand, whereas in high season one can avoid to be too cheap not using the full potential of the high season.

## Limitations
There are two main limitations:  
1. Since the model learns from old data, the hotel must have been using dynamic prices already before using this package to create enough variance within the prices.  
2. The hotel must have exactly 3 room classes and the exact features needed. Although those features should be standard in any hotel data base (date, price etc.)

## Recommended basic Workflow to Optimize Prices using hotel_po
Create an Overview() object and use the available methods to get to know your data.  
Create a Features() object and apply the get_all_final_features() method.  
Create a ModelRF() object and apply the get_all_optimized_hyperparaeters() method.  
Based on that same ModelRF() object apply next the get_all_final_models() method to fit all final models.  
Create a PriceOptimazation() object for a date in the future and apply the get_all_optimized_prices() and get_all_comparisons() methods.  
Review results which are saved as instance-attributes within the PriceOptimazation() object.  

An application of this workflow can be found in demo_of_project.html

## Code-Example:
import sys  
from pathlib import Path  
import os  

current_path = Path.cwd().resolve()  
sys.path.append(str(current_path))  

from hotel_po import price_optimization as po  

features = po.Features("synthetic_hotel_data.csv")  
features.get_all_final_features()  

model = po.ModelRF(features)  
model.get_all_optimized_hyperparameters()  
model.get_all_final_models()  

price = po.PriceOptimization(model, "2025-02-15")  
price.get_all_optimized_prices()  
price.get_all_comparisons()  
