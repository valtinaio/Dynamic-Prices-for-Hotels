# ===========================
# Libraries
# ===========================
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import holidays

# ===========================
# Decorators for iterations
# ===========================
def _iteration_decorator(func):
    # All methods using this decorator must define all args before the kwargs!
    # arg = name-less => Saved in a tuple ()
    # kwargs = variables => Saved as an key in a dict {}
    # No kwargs needed at the moment, but maybe later => Keep it
    def wrapper(*args, **kwargs):
        print("Starting Iteration through all room classes.")
        result = func(*args, **kwargs)
        print("Finished Iteration successfully.")
        return result
    return wrapper


# =====================================
# The Data Class -> Importing the Data
# =====================================

# -> No silent exception handling with try catch method -> raising errors is wanted!

# -------------------------------------
# Functions for the Data class setters
# -------------------------------------

# Is the path a string?
def _path_string_check(path):
    if isinstance(path, str):
        return True
    else:
        raise TypeError("The path must be a string.")

# Does the path exist? 
def _path_exist_check(string_check_result, path):
    if string_check_result == True:
        target_path = Path(path)
        if target_path.exists() and target_path.is_file():
            data = pd.read_csv(target_path)
            return data
        else:
            raise FileNotFoundError("This path does not exist, or the filename is not in the CWD. Give another path or filename.")

# Are the correct columns present?
def _columns_check(exist_check_result):
    data = exist_check_result
    if list(data.columns.values) == ["arrival_date", "room_class", "price_per_night", "lead_time", "is_holiday", "is_weekend"]:
        pass
    else:
        raise ValueError("The columns must have the following names: \"arrival_date\", \"room_class\", \"price_per_night\", \"lead_time\", \"is_holiday\", \"is_weekend\"")

# Are the needed columns numeric?
def _numeric_check(exist_check_result):
    data = exist_check_result
    columns_for_check = range(1, 6) # Only choosing relevant columns
    #  List Comprehension +  Single-line if else
    checks = [True if data.iloc[:, i].dtype in [int, float] else False for i in columns_for_check]
    
    if sum(checks) == 5: 
        pass
    else:
        raise TypeError("Columns must be numeric (float or integer).")

# ------------------
# Data class itself 
# ------------------
class Data():
    """A class that imports and checks the data.
    
    Attributes
    ----------
        data (str): The path were the raw data is stored. If file is in CWD => Only the filename works as well.
        data_agg (pd.DataFrame): The raw data aggregated by arrival date and room class.
        data_1 (pd.DataFrame): The aggregated data filtered by room class 1.
        data_2 (pd.DataFrame): The aggregated data filtered by room class 2.
        data_3 (pd.DataFrame): The aggregated data filtered by room class 3."""
    def __init__(self, path:str):
        self.data = path    # The path will only be the input of the setter!
        self.data_agg = None
        agg_results = self.aggregate()
        self.data_1 = agg_results["data_1"]
        self.data_2 = agg_results["data_2"]
        self.data_3 = agg_results["data_3"]

    # --- Getters & Setters ---
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, path):
        
        # Using the helper-functions
        check_string = _path_string_check(path)
        check_exist = _path_exist_check(check_string, path)
        _columns_check(check_exist)
        _numeric_check(check_exist)
        self._data = check_exist

    # Method 1 - Data Aggregation per arrival day and room class & add time based features
    def aggregate(self):
        """Aggregate the data by arriaval data and room class."""
        data_agg_comp = self.data.groupby(["arrival_date", "room_class"]).agg(
            is_holiday = ("is_holiday", "mean"),    # Mean of constant = constant
            is_weekend = ("is_weekend", "mean"),
            total_demand = ("arrival_date", "count"),
            mean_price = ("price_per_night", "mean"),
            mean_lead_time = ("lead_time", "mean")
        ).reset_index()

        # Transforming floats back to integers where needed for better readability
        data_agg_comp["room_class"] = data_agg_comp["room_class"].astype(int)
        data_agg_comp["is_holiday"] = data_agg_comp["is_holiday"].astype(int)
        data_agg_comp["is_weekend"] = data_agg_comp["is_weekend"].astype(int)

        # Transforming arrival_date to pd.datetime
        data_agg_comp["arrival_date"] = pd.to_datetime(data_agg_comp["arrival_date"])

        # Adding Time Based Features
        data_agg_comp.insert(2, "month", data_agg_comp["arrival_date"].dt.month)
        data_agg_comp.insert(3, "week", data_agg_comp["arrival_date"].dt.isocalendar().week)
        data_agg_comp.insert(4, "day", data_agg_comp["arrival_date"].dt.day)

        # Finally, keeping only data where "mean_price" is > 0.
        self.data_agg = data_agg_comp[data_agg_comp["mean_price"] > 0]

        # Creating a pd.DataFrame per room_class and saving it as instance attribute.
        results = {}
        for i in [1, 2, 3]:
            results[f"data_{i}"] = self.data_agg[self.data_agg["room_class"] == i].set_index("arrival_date", drop=False)
        return results

    # Enabeling len() function on a Data object - Returns the length of the complete aggregated data
    def __len__(self):
        return len(self.data_agg)


# ===========================================
# The Overview Class -> Overview of the Data
# ===========================================
class Overview(Data):
    """A class that creates different overview possibilities of the data.
    
    Attributes
    ----------
        path (str): The path were the raw data is stored."""
    def __init__(self, path:str):
        super().__init__(path)      # Inherits the __init__ from the superclass (= Data) => User can start workflow from here directly

    # Method 1 - Visualization: Line Plot
    def vis_line(self, room_class, column):
        """Create a line-plot of a chosen room_class and column."""
        data = getattr(self, f"data_{room_class}")

        def create_line_plot(room_class):
            # Checking if the selected column is a string and if it is present in the given data
            if isinstance(column, str) and column in data.columns:
                plt.plot(data["arrival_date"], data[column])
                plt.title(f"Line Plot of {column} of Room Class {room_class}")
                plt.xlabel("Time")
                plt.ylabel(f"{column}")
            else:
                raise ValueError("The column must be a string and must exist.")
        return create_line_plot(room_class)
    
    # Method 2 - Visualization: Histogram
    def vis_hist(self, room_class, column):
        """Create a histogram of a chosen room_class and column."""
        data = getattr(self, f"data_{room_class}")

        def create_hist_plot(room_class):
            # Checking if the selected column is a string and if it is present in the given data
            if isinstance(column, str) and column in data.columns:
                plt.hist(data[column])
                plt.title(f"Histogram of {column} of Room Class {room_class}")
                plt.xlabel(f"{column}")
                plt.ylabel("Absolute Frequency")
            else:
                raise ValueError("The column must be a string and must exist.")
        return create_hist_plot(room_class)

    # Method 3 - Basic Statistical Overview
    def stat_overview(self, room_class):
        """Create a statistical overview of the aggregated data of a chosen room_class."""
        data = getattr(self, f"data_{room_class}")
        return data.describe()


# =======================================================================
# The Features Class -> Splitting Target vs. Features + Feature Analysis
# =======================================================================
class Features(Data):
    """A class that creates optimized features for a Random Forest Regressor.
    
    Attributes
    ----------
        path (str): The path were the raw data is stored.
        splitted_data (dict): Dictionary containing all splitted data.
        feature_ranking_i (pd.Index): The features for room class i which maximizes R^2 using RFECV.
        final_Y_train_i (pd.DataFrame): Target training data used for the Model Optimization of room_class i.
        final_Y_test_i (pd.DataFrame): Target test data used for the Model Optimization of room_class i.
        final_X_train_i (pd.DataFrame): Feature training data used for the Model Optimization of room_class i.
        final_X_test_i (pd.DataFrame): Feature training data used for the Model Optimization of room_class i.
    
    Key Methods
    -----------
        get_all_final_features (): Extract the final features of all room classes at once."""
    # --- Class Attributes & Getters and Setters ---
    _test_size = 0.2

    @classmethod
    def get_test_size(cls):
        return cls._test_size
    
    @classmethod
    def set_test_size(cls, new_test_size):
        if isinstance(new_test_size, float) and 1 >= new_test_size > 0:
            cls._test_size = new_test_size
        else:
            raise Exception("The test_size must be numeric and between 0 and 1.")

    def __init__(self, path:str):
        super().__init__(path)          # Inherits the __init__ from the superclass (= Data) => User can start workflow from here directly
        self.splitted_data = None
        self.feature_ranking_1 = None
        self.feature_ranking_2 = None
        self.feature_ranking_3 = None
        self.final_Y_train_1 = None
        self.final_Y_train_2 = None
        self.final_Y_train_3 = None
        self.final_Y_test_1 = None
        self.final_Y_test_2 = None
        self.final_Y_test_3 = None
        self.final_X_train_1 = None
        self.final_X_train_2 = None
        self.final_X_train_3 = None
        self.final_X_test_1 = None
        self.final_X_test_2 = None
        self.final_X_test_3 = None
        self.train_test_split()

    # Method 1 - Training/Test Split
    def train_test_split(self):
        """Split the aggregated data of a chosen room class into training and test data."""
        splitted_data = {}
        for room_class in [1, 2, 3]:
            data = getattr(self, f"data_{room_class}") # Getting the Data for Splitting
            split_point = int(round((len(data) * (1 - Features._test_size)), ndigits=0)) # time-based split
            
            # --- Defining X_train, X_test, y_train, y_test ---
            features = data[["week", "day", "is_holiday", "is_weekend", "mean_price", "mean_lead_time"]]
            target = data[["total_demand"]]
            X_train = features.iloc[0:split_point, :]
            X_test = features.iloc[split_point:, :]
            final_y_train = target.iloc[0:split_point, :]
            final_y_test = target.iloc[split_point:, :]
            splitted_data_room_class = {
                f"X_train_{room_class}": X_train,
                f"X_test_{room_class}": X_test,
                f"final_y_train_{room_class}": final_y_train,
                f"final_y_test_{room_class}": final_y_test
            }
            # --- Summarizing ---
            setattr(self, f"final_Y_train_{room_class}", splitted_data_room_class[f"final_y_train_{room_class}"])
            setattr(self, f"final_Y_test_{room_class}", splitted_data_room_class[f"final_y_test_{room_class}"])
            splitted_data[f"data_{room_class}_splitted"] = splitted_data_room_class
        self.splitted_data = splitted_data

    # Method 2 - Feature Correlation: As a plot or raw-correlation matrix
    def correlation(self, room_class, plot=True):
        """Analyze the correlation of the training features of a chosen room_class."""
        data_for_corr = self.splitted_data[f"data_{room_class}_splitted"][f"X_train_{room_class}"]
        
        # --- Generating the plot if plot=True ---
        if plot == True:
            corr_matrix = data_for_corr.corr()
            sns.heatmap(corr_matrix, 
                annot=True,      # showing numbers in the plot
                cmap='coolwarm', # color theme
                fmt='.2f',       # Amount of decimals
                center=0)
            plt.title("Heatmap Feature Correlation")
        else:
            # --- Genrating the correlation matrix if plot=False ---
            return data_for_corr.corr()
    
    # Method 3 - Feature Ranking with Recursive Feature Elimination with Cross-Validation (RFECV)
    def feature_ranking_rfecv(self, room_class):
        """Create a feature ranking with RFECV based on the features of a chosen room_class."""
        data_for_rank = self.splitted_data[f"data_{room_class}_splitted"]
        
        # 1. Defining the model and the cross validation
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # 2. Defining the selector
        selector = RFECV(estimator=rf, step=1, cv=cv, scoring='r2')

        # 3. Fitting of the different models
        selector = selector.fit(data_for_rank[f"X_train_{room_class}"], data_for_rank[f"final_y_train_{room_class}"])

        # 4. Defining the best features
        feature_ranking_rfecv = data_for_rank[f"X_train_{room_class}"].columns[selector.support_]

        # 5. Adding Result to Feature_Results
        setattr(self, f"feature_ranking_{room_class}", feature_ranking_rfecv)
        return feature_ranking_rfecv

    # Method 4 - Extracting final features
    def final_features(self, room_class):
        """Filter only the best features of a chosen room_class based on the feature ranking."""
        # --- Filtering best Features according to the feature ranking ---
        best_features = list(self.feature_ranking_rfecv(room_class))
        final_X_train = self.splitted_data[f"data_{room_class}_splitted"][f"X_train_{room_class}"][best_features]
        final_X_test = self.splitted_data[f"data_{room_class}_splitted"][f"X_test_{room_class}"][best_features]
        setattr(self, f"final_X_train_{room_class}", final_X_train)
        setattr(self, f"final_X_test_{room_class}", final_X_test)
        return final_X_train, final_X_test
    
    # Method 5 - Extracting all final features
    @_iteration_decorator
    def get_all_final_features(self):
        """Extract the final features of all room classes at once."""
        for i in [1, 2, 3]:
            self.final_features(i)


# ================================================================================
# The ModelRF Class -> Optimizing Hyperparameters based on Training and Test Data
# ================================================================================
class ModelRF():
    """A class that creates an optimized Random Forest Regressor to predict demand.
    
    Attributes
    ----------
        features_object (Features): Contains a Features object with all the data needed for this class.
        optimal_model_i (dict): Results of the model with optimized hyperparameters using only the training data for class i.
        final_model_i (dict): Results of the model with optimized hyperparameters using all data (but only best features) for class i.
    
    Key Methods
    -----------
        get_all_optimized_hyperparameters(): Optimize hyperparameters of a Random Forest Regressor using GridSearchCV for all room classes at once.
        get_all_final_models(): Create final Random Forest Regressor with the optimal hyperparameters for all room classes at once."""
    def __init__(self, features_object:Features):
        self.features_object = features_object
        self.optimal_model_1 = None
        self.optimal_model_2 = None
        self.optimal_model_3 = None
        self.final_model_1 = None
        self.final_model_2 = None
        self.final_model_3 = None
    
    # --- Instance-Attributes Getters & Setters ---
    @property
    def features_object(self):
        return self._features_object
    
    @features_object.setter
    def features_object(self, feature_object):
        if isinstance(feature_object, Features):
            self._features_object = feature_object
        else:
            raise TypeError("The feature_object must be from the class \"Features\".")

    # Method 1 - Hyperparameter Optimazation with GridSearchCV
    def optimize_hyperparameter(self, room_class):
        """Optimize hyperparameters of a Random Forest Regressor using GridSearchCV for chosen room_class and optional individual param_grid."""
        if self.features_object.final_X_test_1 is not None:
            # 1. Defining Parameter Grid
            param_grid = {
                "random_state": [42],
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
                "min_samples_split": [2, 5, 10],
                "max_features": ['sqrt', 1.0, 0.5]
            }

            # 2. Defining the GridSearchCV
            grid_search = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid=param_grid,
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )

            # 3. Execution of the GridSearchCV only with the best features
            grid_search.fit(getattr(self.features_object, f"final_X_train_{room_class}"), getattr(self.features_object, f"final_Y_train_{room_class}"))

            optimal_pred = grid_search.best_estimator_.predict(getattr(self.features_object, f"final_X_test_{room_class}"))

            # 4. Summarizing Results
            results = {
                "optimal_model": grid_search.best_estimator_,
                "optimal_hyperparams": grid_search.best_params_,
                "optimal_pred": optimal_pred,
                "optimal_r2": grid_search.best_score_,
                "optimal_mae": mean_absolute_error(getattr(self.features_object, f"final_Y_test_{room_class}"), optimal_pred),
                "optimal_mse": mean_squared_error(getattr(self.features_object, f"final_Y_test_{room_class}"), optimal_pred)
            }
            # Adding the optimal hyperparameter results as instance attributes.
            setattr(self, f"optimal_model_{room_class}", results)
            return results
        else:
            print("There are no final training features available. Did you create them within the features_objet?")
    
    # Method 2 - Optimizing all Models at once (Takes approx. 15 Min with provided synthetic dataset)
    @_iteration_decorator
    def get_all_optimized_hyperparameters(self):
        """Optimize hyperparameters of a Random Forest Regressor using GridSearchCV for all room classes at once."""
        for i in [1, 2, 3]:
            self.optimize_hyperparameter(i)

    # Method 2 - Optimal Model Result Visualization
    def optimal_model_vis(self, room_class):
        """Visualize the optimal model's prediction vs. the real test data."""
        if getattr(self, f"optimal_model_{room_class}") is not None:  # Checking if optimal model was created already
            data = getattr(self, f"optimal_model_{room_class}")
            optimal_results_df = pd.DataFrame({                       # Creating a pd.DataFrame for easier visualization
                "prediction": data["optimal_pred"],
                "real": getattr(self.features_object, f"final_Y_test_{room_class}").iloc[:, 0]
            })
            optimal_results_df.plot()
            plt.title(f"Random Forest Prediction vs. Reality - Room Class {room_class}")
        else:
            raise RuntimeError("You first must do the Hyperparameter-Optimazation with the get_all_optimized_hyperparameter() method before you are able to visualize the optimal model results.")

    # Method 3 - Final Model
    def final_model(self, room_class):
        """Create final Random Forest Regressor with the optimal hyperparameters for a chosen room_class."""
        # --- Creating model with optimal hyperparameters and ALL final features---
        if getattr(self, f"optimal_model_{room_class}") is not None:
            final_features = getattr(self.features_object, f"data_{room_class}")[getattr(self.features_object, f"final_X_train_{room_class}").columns]
            final_target = getattr(self.features_object, f"data_{room_class}")["total_demand"]
            final_model = RandomForestRegressor(
                **getattr(self, f"optimal_model_{room_class}")["optimal_hyperparams"]
            )
            final_model.fit(final_features, final_target)
            
            # --- Creating evaluation metrics (no more testing data!) & adding results as instance attribute ---
            final_y_pred = final_model.predict(final_features)
            final_model_results = {
                "final_model": final_model,
                "r2_on_train": final_model.score(final_features, final_target),
                "mae_on_train": mean_absolute_error(final_target, final_y_pred),
                "mse_on_train": mean_squared_error(final_target, final_y_pred)
            }
            setattr(self, f"final_model_{room_class}", final_model_results)
            return final_model_results
        else:
            raise RuntimeError("You first must do the Hyperparameter-Optimazation with the get_all_optimized_hyperparameter() method.")

    # Method 4 - Extracting all final models at once
    @_iteration_decorator
    def get_all_final_models(self):
        """Create final Random Forest Regressor with the optimal hyperparameters for all room classes at once."""
        for i in [1, 2, 3]:
            self.final_model(i)
    
    # Special Method for Indexing with []
    def __getitem__(self, room_class):
        try:
            return getattr(self, f"final_model_{room_class}")["final_model"]
        except KeyError:
            print("The Index must be one of the three room classes: 1, 2, 3. Slicing is not possible.")


# =======================================================================
# The ProceOptimazation Class -> Optimizing Prices based on a given date
# =======================================================================
class PriceOptimization():
    """A class that optimizes prices for a given future date maximizing the revenue.
    
    Attributes
    ----------
        modelrf_object (ModelRF): Contains a ModelRF object with all the data needed for this class.
        date (pd.Timestamp): A future date for which the price should be optimized.
        optimal_price_i (pd.DataFrame): Ordered (top down by revenue) tabel with recommended prices to maximize revenue for room class i.
        comparison_i (pd.DataFrame): Ordered (top down by revenue) training data from the same time-period as the future date the price is optimized for.
    
    Key Methods
    -----------
        get_all_optimized_prices(): Optimize the price which maximizes the revenue for all room classes at once.
        get_all_comparisons(): Get the training data from the same time period as the future date for comparison for all room classes at once."""
    def __init__(self,
                 modelrf_object:ModelRF, date:str):
        self.modelrf_object = modelrf_object
        self.date = date
        self.optimal_price_1 = None
        self.optimal_price_2 = None
        self.optimal_price_3 = None
        self.comparison_1 = None
        self.comparison_2 = None
        self.comparison_3 = None

    # --- Getters & Setters ---
    @property
    def date(self):
        return self._date
    
    @property
    def modelrf_object(self):
        return self._modelrf_object

    # The PriceOptimization class will only be functional if the models within the ModelRF 
    # class have already been created. This is (also) checked in the setter:
    @date.setter
    def date(self, date):
        if isinstance(date, str) and pd.to_datetime(date) > max(getattr(self.modelrf_object, "features_object").data_agg["arrival_date"]):
            self._date = pd.to_datetime(date)
        else:
            raise TypeError("\"date\" must be a String and be in the future.")
    
    @modelrf_object.setter
    def modelrf_object(self, modelrf_object):
        if isinstance(modelrf_object, ModelRF):
            self._modelrf_object = modelrf_object
        else:
            raise TypeError("\"modelrf_object\" must be from the class \"ModelRF\"")

    # Method 1 - Price Optimization
    def optimize_price(self, room_class):
        """Optimize the price for a chosen room_class which maximizes the revenue."""
        # 1. Getting the optimized Model of the class and its feature names and italian holidays
        model = getattr(self.modelrf_object, f"final_model_{room_class}")["final_model"]
        feature_names = getattr(self.modelrf_object.features_object, f"final_X_train_{room_class}").columns
        holidays_it = holidays.Italy(subdiv="BZ")

        # 2. Defining prices to optimize based on all seen prices in the training data
        min_p = int(round(min(getattr(self.modelrf_object.features_object, f"data_{room_class}")["mean_price"][getattr(self.modelrf_object.features_object, f"data_{room_class}")["week"] == self.date.isocalendar().week])))
        max_p = int(round(max(getattr(self.modelrf_object.features_object, f"data_{room_class}")["mean_price"][getattr(self.modelrf_object.features_object, f"data_{room_class}")["week"] == self.date.isocalendar().week])))
        price_list = list(range(min_p, max_p, 10))

        # 3. Creating the Features-DataFrame -> Since all of them (except the price) are constants => Broadcasting
        optimal_features = pd.DataFrame({"mean_price": price_list})
        for feature in feature_names:
            match feature:
                case "mean_price":
                    continue                    # Already given
                case "week":
                    optimal_features[feature] = self.date.isocalendar().week
                case "day":
                    optimal_features[feature] = self.date.day
                case "is_holiday":
                    optimal_features[feature] = int(self.date in holidays_it)
                case "is_weekend":
                    optimal_features[feature] = int(self.date.day_of_week in [5, 6])
                case "mean_lead_time":
                    optimal_features[feature] = (self.date - max(getattr(self.modelrf_object.features_object, f"data_{room_class}").index)).days
        optimal_features = optimal_features[feature_names]

        # 4. Predicting the demand for all prices
        predicted_demand = model.predict(optimal_features)

        # 5. Saving the results
        optimal_features["demand"] = predicted_demand
        optimal_features["revenue"] = optimal_features["demand"] * optimal_features["mean_price"]

        # 6. Summarzing
        results = optimal_features[["demand", "revenue", "mean_price", "week", "mean_lead_time"]].sort_values(by="revenue", ascending=False).head(10)
        setattr(self, f"optimal_price_{room_class}", results)
        return results
    
    # Method 2 - Get comparison
    def get_comparison(self, room_class):
        """Get the training data from the same time period as the future date for comparison for a chosen room_class."""
        # --- Getting Features and Target with the same week as the week we are optimizing the price for ---
        data_for_comp =  getattr(self.modelrf_object.features_object, f"data_{room_class}")[getattr(self.modelrf_object.features_object, f"data_{room_class}")["week"] == self.date.isocalendar().week]
        revenue = pd.DataFrame({"revenue": data_for_comp["mean_price"] * data_for_comp["total_demand"]})
        comparison = pd.concat([revenue, data_for_comp], axis=1)
        comparison = comparison[["total_demand", "revenue", "mean_price", "week", "mean_lead_time"]].sort_values(by="revenue", ascending=False).head(20)
        setattr(self, f"comparison_{room_class}", comparison)
        return comparison

    # Method 3 - Optimize all prices at once
    @_iteration_decorator
    def get_all_optimized_prices(self):
        """Optimize the price which maximizes the revenue for all room classes at once."""
        for i in [1, 2, 3]:
            self.optimize_price(i)
    
    # Method 4 - Get all comparisons
    @_iteration_decorator
    def get_all_comparisons(self):
        """Get the training data from the same time period as the future date for comparison for all room classes at once."""
        for i in [1, 2, 3]:
            self.get_comparison(i)
    
    # --- Enabeling print() function on PriceOptimization class ---
    def __str__(self):
        str_output = f"--- Optimized Prices for {self.date.strftime('%Y-%m-%d')} ---\n"
        str_output += f"Recommended Price for Room Class 1: {self.optimal_price_1.iloc[0, 2]}\n"
        str_output += f"Recommended Price for Room Class 2: {self.optimal_price_2.iloc[0, 2]}\n"
        str_output += f"Recommended Price for Room Class 3: {self.optimal_price_3.iloc[0, 2]}"
        return str_output
    
    # --- Enabeling slicing function on PriceOptimization class ---
    def __getitem__(self, room_class):
        try:
            return getattr(self, f"optimal_price_{room_class}").iloc[0, 2]
        except KeyError:
            print("The Index must be one of the three room classes: 1, 2, 3. Slicing is not possible.")


# =============================
# main() Function
# =============================
# This following function is meant to be used if the program is not used as a
# package, but within a user-interface where the hole process should be done
# automated:
def main():
    """Execute the complete price optimization pipeline automatically."""
    # --- Import of the data ---
    while True:
        user_path = input("Please provide the path where the data is located in. If the data is in the CWD only the filename is also accepted. Write \"exit\" to exit the program.")
        
        # 1. Exit Check
        if user_path.lower() == "exit":
            print("Program closed.")
            return # Exit the hole function
        try:
            # 2. Checking for correct path properties
            features_object = Features(user_path)
        except (TypeError, FileNotFoundError, ValueError) as e:
            if isinstance(e, TypeError):
                print(f"{e}: The path must be a string. And all columns must be numeric (float or integer).")
            elif isinstance(e, FileNotFoundError):
                print(f"{e}: This path does not exist, or the filename is not in the CWD. Give another path or filename.")
            else:
                print(f"{e}: The columns must have the following names: \"arrival_date\", \"room_class\", \"price_per_night\", \"lead_time\", \"is_holiday\", \"is_weekend\"")
        else:
            # 3. Exit the while loop if everything is OK
            print("Data loaded successfully!\n")
            break
    
    # --- Defining the date for the optimization process ---
    while True:
        user_date = input("Please provide a date in the future (YYYY-MM-DD) you want to optimize the prices for (or 'exit').")
        
        # 1. Exit Check
        if user_date.lower() == "exit":
            print("Program closed.")
            return # Exit the hole function
        
        # 2. Format Check & Converting: format='%Y-%m-%d' if not => ValueError.
        try:
            converted_date = pd.to_datetime(user_date, format='%Y-%m-%d')
        except ValueError:
            print("Incorrect format! Please use YYYY-MM-DD (e.g., 2025-05-20).")
        else:
            # 3. Check if date is in the future (compared to the complete training data)
            max_date = max(features_object.data_agg["arrival_date"])
            if converted_date > max_date:
                print(f"Date accepted: {converted_date.date()}\n")
                break # Exit while loop
            else:
                print(f"Date must be in the future (after {max_date.date()}).")
                
    # --- Execution of the Feature Ranking and final Extraction ---
    print("Starting with the Feature Extraction.")
    try:
        features_object.get_all_final_features()
    except Exception as e:
        return f"There was an {e.__class__.__name__} in the Feature extraction. Contact our service."
    else:
        print("Extracted all Features successfully.\n")
    
    # --- Execution of the Model Creation ---
    print("Starting with the Model Creation.")
    try:
        model_object = ModelRF(features_object)
        model_object.get_all_optimized_hyperparameters()
        model_object.get_all_final_models()
    except Exception as e:
        return f"There was an {e.__class__.__name__} in the Model creation. Contact our service."
    else:
        print("The Model was created successfully!\n")

    # --- Execution of the Price Optimization ---
    print("Starting with the Price Optimization")
    try:
        price_opimization_object = PriceOptimization(model_object, user_date)
        price_opimization_object.get_all_optimized_prices()
        price_opimization_object.get_all_comparisons()
        print("Finished Price Optimization successfully.\n")
        print(price_opimization_object)
    except Exception as e:
        return f"There was an {e.__class__.__name__} in the Price Optimization. Contact our service."