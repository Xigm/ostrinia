# open datasets/thesis_dataset_after_transformation.csv
import numpy as np
import pandas as pd
import os
import folium


df = pd.read_csv("datasets/thesis_dataset_after_transformation.csv")

# get positiion data, columns Latitude_x and Longitude_x
pos_data = df[['Latitude_x', 'Longitude_x']].values

# check unique values in Latitude_x and Longitude_x
unique_latitudes = np.unique(pos_data[:, 0])
unique_longitudes = np.unique(pos_data[:, 1])
print(f"Unique latitudes: {unique_latitudes}")
print(f"Unique longitudes: {unique_longitudes}")

# get unique lat long pairs
unique_positions = np.unique(pos_data, axis=0)
print(f"Unique positions: {unique_positions.shape[0]}")

# dont use to compute the clusters positions 46.424419     6.283014 and 46.424484     6.283247
excluded_positions = np.array([[46.424419, 6.283014], [46.424484, 6.283247]])
unique_positions = np.array([pos for pos in unique_positions if not np.any(np.all(pos == excluded_positions, axis=1))])

# create a folium map centered at the mean position
mean_position = np.mean(unique_positions, axis=0)
m = folium.Map(location=mean_position, zoom_start=10)
# add markers for each unique position
for lat, lon in unique_positions:
    folium.Marker(location=[lat, lon]).add_to(m)
# save the map to an HTML file
m.save("unique_positions_map.html")

# cluster them in 9 different clusters BUT dont use 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=11, random_state=0).fit(unique_positions)
# get the cluster centers
cluster_centers = kmeans.cluster_centers_
print(f"Cluster centers: {cluster_centers}")

# add back the excluded positions to the cluster centers
cluster_centers = np.vstack((cluster_centers, excluded_positions))

# create a folium map centered at the mean position
m_clusters = folium.Map(location=mean_position, zoom_start=10)
# add markers for each cluster center
for lat, lon in cluster_centers:
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='red')).add_to(m_clusters)
# save the map to an HTML file
m_clusters.save("cluster_centers_map.html")

# check that there are no datapoints with same date and position
# change lat and lon to cluster index
df['node_index'] = kmeans.predict(pos_data)
duplicates = df[df.duplicated(subset=['Date', 'node_index'], keep=False)]
if not duplicates.empty:
    print("Duplicate datapoints found:")
    print(duplicates)

# plot duplicates on a map centered at the mean position of duplicates SO YOU HAVE TO COMPUTE AGAIN THE MEAN POSITION
mean_position = np.mean(duplicates[['Latitude_x', 'Longitude_x']].values, axis=0)
import folium
m_duplicates = folium.Map(location=mean_position, zoom_start=10)
# add markers for each duplicate position
for lat, lon in duplicates[['Latitude_x', 'Longitude_x']].values:
    folium.Marker(location=[lat, lon], icon=folium.Icon(color='blue')).add_to(m_duplicates)
# save the map to an HTML file
m_duplicates.save("duplicates_map.html")

import numpy as np
import math

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Earth's radius in kilometers
    R = 6371.0
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    distance = R * c
    return distance

def calculate_distances_vector(coordinates):
    """
    Calculate distances between consecutive pairs of coordinates.
    
    Args:
        coordinates: numpy array of shape (N, 2) where N is even
                    Each row is [latitude, longitude]
    
    Returns:
        numpy array of distances between consecutive pairs
    """
    coordinates = np.array(coordinates)
    
    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (N, 2)")
    
    if coordinates.shape[0] % 2 != 0:
        raise ValueError("Number of coordinates must be even (pairs)")
    
    distances = []
    
    # Process coordinates in pairs
    for i in range(0, len(coordinates), 2):
        lat1, lon1 = coordinates[i]
        lat2, lon2 = coordinates[i + 1]
        
        distance = calculate_distance(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    return np.array(distances)

def calculate_distance_matrix(coordinates):
    """
    Calculate all pairwise distances between coordinates.
    
    Args:
        coordinates: numpy array of shape (N, 2)
                    Each row is [latitude, longitude]
    
    Returns:
        numpy array of shape (N, N) with pairwise distances
    """
    coordinates = np.array(coordinates)
    
    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (N, 2)")
    
    n = len(coordinates)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = coordinates[i]
            lat2, lon2 = coordinates[j]
            
            distance = calculate_distance(lat1, lon1, lat2, lon2)
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # Symmetric matrix
    
    return distance_matrix

def calculate_distances_vectorized(coordinates):
    """
    Vectorized version using numpy for better performance with large datasets.
    Calculate distances between consecutive pairs.
    """
    coordinates = np.array(coordinates)
    
    if coordinates.shape[1] != 2:
        raise ValueError("Coordinates must have shape (N, 2)")
    
    if coordinates.shape[0] % 2 != 0:
        raise ValueError("Number of coordinates must be even (pairs)")
    
    # Reshape to pairs
    pairs = coordinates.reshape(-1, 2, 2)
    
    # Convert to radians
    coords_rad = np.radians(pairs)
    
    # Extract coordinates
    lat1, lon1 = coords_rad[:, 0, 0], coords_rad[:, 0, 1]
    lat2, lon2 = coords_rad[:, 1, 0], coords_rad[:, 1, 1]
    
    # Haversine formula vectorized
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Earth's radius in kilometers
    R = 6371.0
    distances = R * c
    
    return distances

# Example coordinates: [latitude, longitude]
coordinate_pairs = cluster_centers

    # Calculate NxN distance matrix
distance_matrix = calculate_distance_matrix(coordinate_pairs)

print("Distance Matrix:")
print(distance_matrix)

# Save the distance matrix to a file, with each coordinate ordered in the same order
np.save("datasets/ostrinia_distance_matrix.npy", distance_matrix)
# Save the coordinate pairs to a file
np.save("datasets/ostrinia_coordinate_pairs.npy", coordinate_pairs)