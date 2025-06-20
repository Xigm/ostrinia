from tsl.datasets.prototypes import DatetimeDataset
import pandas as pd
import os
import numpy as np
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree


class Ostrinia(DatetimeDataset):

    similarity_options = {'distance'}

    def __init__(self, root: str = "datasets", input_zeros: bool = True, freq=None, target="nb_ostrinia", smooth: bool = False):
        
        self.root = root
        self.extra_data = None
        self.target = target
        self.smooth = smooth

        df, dist, mask = self.load(input_zeros)

        super().__init__(target=df,
                         mask=mask,
                         freq=freq,
                         similarity_score="distance",
                         temporal_aggregation="nearest",
                         name="Ostrinia")

        self.add_covariate('dist', dist, pattern='n n')

    def load_raw(self):

        self.maybe_build()
        # load insect data
        insect_path = os.path.join(self.root_dir, 'thesis_dataset_after_transformation.csv')
        df = pd.read_csv(insect_path)

        # add missing values (index is sorted)
        # date_range = pd.date_range(df.index[0], df.index[-1], freq='5T')
        # df = df.reindex(index=date_range)

        # load distance matrix
        path = os.path.join(self.root_dir, 'ostrinia_distance_matrix.npy')
        dist = np.load(path)

        # convert distance matrix to edge list and edge weights

        # load coordinates
        coords_path = os.path.join(self.root_dir, 'ostrinia_coordinate_pairs.npy')
        coords = np.load(coords_path)

        return df, dist, coords

    def load(self, impute_zeros=True):

        df, dist, coords = self.load_raw()

        # change Latitude and Longitude vars to its closes cluster (which are coords)
        df = map_to_nearest_coords(df, coords)        

        # reshape the df to be date as index and node_index as columns
        df = df.set_index('Date')
        df_clean = df.pivot(columns='node_index', values=self.target)

        # if smooth is True, apply a rolling mean with a window of 7
        if self.smooth:
            df_clean = df_clean.rolling(window=7, min_periods=7).mean()

        # compute mask for nan values
        mask = ((~np.isnan(df_clean.values)).astype('uint8'))
        if impute_zeros:
            # replace NaN with 0
            df_clean = df_clean.fillna(0)
        else:
            print("Imputing zeros is disabled, NaN values will remain in the dataset.")

        # Check for duplicate combinations of Date and node_index
        duplicates = df.reset_index().duplicated(subset=['Date', 'node_index'], keep=False)
        if duplicates.any():
            print("Duplicate combinations of Date and node_index found:")
            print(df.reset_index()[duplicates][['node_index']])
        # do the same with columns trap_elevation, corn_size, incrementing_ostrinia, nb_ostrinia_new, Weather Station Changins, TempAv, TempMin, TempMax, TempGrasMin, TempSol-5, TempSoil-10, RHmoy, Prec, PrecMax, Insol, Ray, EvapTranspi, WindSpeedAb, WindSpeedmax, WaterDef, Somt, Soms
        # put them in self.extra_data which will be a dict of DataFrames
        list_of_extra_columns = ['nb_ostrinia', 'trap_elevation', 'corn_size', 'incrementing_ostrinia', 'nb_ostrinia_new', 'Weather Station Changins', 'TempAv', 'TempMin', 'TempMax', 'TempGrasMin', 'TempSol-5', 'TempSoil-10', 'RHmoy', 'Prec', 'PrecMax', 'Insol', 'Ray', 'EvapTranspi', 'WindSpeedAb', 'WindSpeedmax', 'WaterDef', 'Somt', 'Soms']
        extra_data = {}
        for element in list_of_extra_columns:
            if element != self.target:
                extra_data[element] = df.pivot(columns='node_index', values=element)
                if element in ['nb_ostrinia', 'incrementing_ostrinia', 'nb_ostrinia_new']:
                    # if the column is one of these, we want to impute zeros
                    extra_data[element] = extra_data[element].rolling(window=7, min_periods=1).mean()

        # add as covariate the day of the year. index is a string with format YYYY-MM-DD. extract the day of the year and create an enconding 
        index = df.index
        # encode it as a number from 0 to 365
        df['day_of_year'] = pd.to_datetime(index, format='%Y-%m-%d').dayofyear
        extra_data['day_of_year'] = df.pivot(columns='node_index', values="day_of_year")

        # drop Weather Station Changins
        if 'Weather Station Changins' in extra_data:
            extra_data.pop('Weather Station Changins')

        self.extra_data = extra_data

        return df_clean, dist, mask


    def get_connectivity(self, layout, **kwargs):
        """
        Get the connectivity matrix for the dataset.
        
        Returns:
            np.ndarray: The connectivity matrix.
        """
        if layout == 'edge_index':
            from tsl.ops.connectivity import adj_to_edge_index
            return adj_to_edge_index(self._covariates['dist']['value'])
        elif layout == 'distance':
            dist = self._covariates['dist']['value']
        return dist

    # def set_connectivity(self, connectivity):
    #     """
    #     Set the connectivity matrix for the dataset.
        
    #     Parameters:
    #         connectivity (np.ndarray): The new connectivity matrix.
    #     """
    #     self.edge_index, self.edge_weight = self._parse_connectivity(
    #         connectivity, 'edge_index')
        

def map_to_nearest_coords(df, coord_list):
    """
    Map DataFrame coordinates to nearest coordinates in coord_list
    
    Parameters:
    df: DataFrame with 'Latitude_x' and 'Longitude_x' columns
    coord_list: List of coordinate pairs [(lat1, lon1), (lat2, lon2), ...]
    
    Returns:
    DataFrame with added columns for nearest coordinates and their indices
    """
    
    # Convert coord_list to numpy array
    coord_array = np.array(coord_list)
    
    # Extract df coordinates
    df_coords = df[['Latitude_x', 'Longitude_x']].values
    
    # Convert to radians for haversine distance
    df_coords_rad = np.radians(df_coords)
    coord_array_rad = np.radians(coord_array)
    
    # Use BallTree for efficient nearest neighbor search with haversine metric
    tree = BallTree(coord_array_rad, metric='haversine')
    
    # Find nearest neighbor for each point in df
    distances, indices = tree.query(df_coords_rad, k=1)
    
    # Convert distances from radians to kilometers (Earth's radius ≈ 6371 km)
    distances_km = distances.flatten() * 6371
    
    # Add results to dataframe
    df['node_index'] = indices.flatten()
    df['error_distance'] = distances_km

    return df

