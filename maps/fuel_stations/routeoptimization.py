import os
import dask.dataframe as dd
import sqlite3
import pandas as pd
import folium
import requests
import json
from geopy.distance import geodesic
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
import concurrent.futures
from datetime import datetime
from sklearn.linear_model import LinearRegression
from functools import lru_cache
from scipy.spatial import cKDTree
import numpy as np
from .models import FuelStation

# Default configuration values
DEFAULT_TANK_CAPACITY = 50  # gallons
DEFAULT_FUEL_EFFICIENCY = 10  # miles per gallon
DEFAULT_MAX_DETOUR = 15  # miles
DEFAULT_MIN_FUEL_STOP = 5  # gallons

class FuelStop:
    def __init__(self, location: Tuple[float, float], price: float, distance: float):
        self.location = location
        self.price = price
        self.distance = distance
        self.cost_to_reach = float('inf')
        self.previous_stop = None

def preprocess_and_save_to_sqlite(csv_file, sqlite_db):
    if not os.path.exists(csv_file):
        print(f"Error: The file {csv_file} does not exist.")
        return

    # Load entire CSV with Dask
    dask_df = dd.read_csv(csv_file)

    # Convert to Pandas
    pandas_df = dask_df.compute()

    # Save to SQLite
    conn = sqlite3.connect(sqlite_db)
    pandas_df.to_sql('fuel_stations', conn, if_exists='replace', index=False)
    conn.close()
    print("All data saved to SQLite.")

def get_coordinates(location, api_key):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={location}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['results']:
        lat = data['results'][0]['geometry']['lat']
        lng = data['results'][0]['geometry']['lng']
        return lat, lng
    else:
        print(f"Warning: Could not get coordinates for location: {location}")
        return None

def calculate_distance(point1, point2):
    """Calculate distance between two points using Haversine formula"""
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Convert to radians
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Earth's radius in miles
    r = 3958.8
    return c * r

@lru_cache(maxsize=1024)
def calculate_distance_cached(point1, point2):
    """Cached version of distance calculation"""
    return calculate_distance(point1, point2)

def get_route_data(start_coords, end_coords, api_key):
    url = (
        f"https://graphhopper.com/api/1/route?point={start_coords[0]},{start_coords[1]}"
        f"&point={end_coords[0]},{end_coords[1]}&vehicle=car&points_encoded=false&key={api_key}"
    )
    response = requests.get(url)
    data = response.json()

    if 'paths' in data:
        total_distance = data['paths'][0]['distance'] / 1000  # Convert to km
        estimated_time = data['paths'][0]['time'] / 1000 / 60  # Convert ms to minutes
        route_points = [
            [point[1], point[0]]
            for point in data['paths'][0]['points']['coordinates']
        ]
        return {
            'total_distance_miles': total_distance * 0.621371,  # Convert to miles
            'estimated_time_hours': estimated_time / 60,  # Convert to hours
            'route_points': route_points
        }
    else:
        raise ValueError("Could not get route data: " + str(data))

def load_fuel_stations(db_path=None):
    """Load fuel stations from Django database"""
    stations = []
    for station in FuelStation.objects.all():
        stations.append({
            'Truckstop Name': station.truckstop_name,
            'Address': station.address,
            'Retail Price': float(station.retail_price),
            'Latitude': station.latitude,
            'Longitude': station.longitude
        })
    return stations

def generate_map_and_route_data(start_coords, end_coords, fuel_stops, route_geometry):
    # Create a map centered on the starting location
    m = folium.Map(location=start_coords, zoom_start=6)

    # Add a marker for the starting point
    folium.Marker(
        start_coords,
        popup="Start",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # Add a marker for the ending point
    folium.Marker(
        end_coords,
        popup="End",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Plot each refueling stop with information if available
    for stop in fuel_stops:
        if stop['station']:  # Check if a station is found
            station = stop['station']
            folium.Marker(
                (station['lat'], station['lng']),
                popup=(f"{station['name']}<br>Cost: ${station['cost']:.2f}<br>"
                       f"Price/Gallon: ${station['price_per_gallon']:.2f}<br>"
                       f"Gallons Purchased: {station['gallons_purchased']:.2f}<br>"
                       f"Address: {station['address']}"),
                icon=folium.Icon(color="blue", icon="cloud")
            ).add_to(m)
        else:
            # If no station was found, mark the location with a different icon
            folium.Marker(
                stop['position'],
                popup="No station found within 5 miles",
                icon=folium.Icon(color="gray", icon="info-sign")
            ).add_to(m)

    # Draw a line along the route
    folium.PolyLine([start_coords] + [stop['position'] for stop in fuel_stops] + [end_coords], color="blue").add_to(m)

    # Add route corridor
    for point in route_geometry:
        folium.Circle(
            location=point,
            radius=DEFAULT_MAX_DETOUR * 1609.34,  # 15 miles in meters
            color="gray",
            fill=True,
            opacity=0.1
        ).add_to(m)

    return m

def calculate_detour_distance(point, route_points):
    """Calculate minimum detour distance from a point to the route"""
    min_detour = float('inf')
    for i in range(len(route_points) - 1):
        point1 = (route_points[i][0], route_points[i][1])
        point2 = (route_points[i+1][0], route_points[i+1][1])
        station_point = (point[0], point[1])
        
        # Calculate distances
        d1 = calculate_distance(point1, station_point)
        d2 = calculate_distance(station_point, point2)
        direct = calculate_distance(point1, point2)
        
        detour = d1 + d2 - direct
        min_detour = min(min_detour, detour)
    
    return min_detour

def get_nearest_route_point(point, route_points, max_distance=50):
    """Find the nearest point on route within max_distance miles"""
    point_lat, point_lng = point
    min_dist = float('inf')
    nearest_point = None
    
    for route_point in route_points:
        dist = calculate_distance((point_lat, point_lng), (route_point[0], route_point[1]))
        if dist < min_dist and dist <= max_distance:
            min_dist = dist
            nearest_point = route_point
    
    return nearest_point, min_dist

def calculate_distance_matrix(points1, points2):
    """Vectorized distance calculation for multiple points"""
    lat1, lon1 = np.radians(points1[:, 0]), np.radians(points1[:, 1])
    lat2, lon2 = np.radians(points2[:, 0]), np.radians(points2[:, 1])
    
    dlat = lat2.reshape(-1, 1) - lat1
    dlon = lon2.reshape(-1, 1) - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2.reshape(-1, 1)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3958.8 * c  # Earth's radius in miles

def filter_stations_in_corridor(fuel_stations, route_points, corridor_width=25):
    """Optimized station filtering"""
    if not fuel_stations:
        return []
    
    # Convert to numpy arrays
    stations = np.array([[s['Latitude'], s['Longitude']] for s in fuel_stations])
    route = np.array(route_points)
    
    # Track stations within corridor
    station_distances = np.full(len(fuel_stations), np.inf)
    
    # Process route in chunks
    chunk_size = 50
    for i in range(0, len(route)-1, chunk_size):
        end_idx = min(i + chunk_size, len(route))
        route_chunk = route[i:end_idx]
        
        # Calculate distances for this chunk
        distances = np.zeros((len(stations), len(route_chunk)))
        for j, point in enumerate(route_chunk):
            dlat = np.radians(stations[:, 0] - point[0])
            dlon = np.radians(stations[:, 1] - point[1])
            lat1 = np.radians(stations[:, 0])
            lat2 = np.radians(point[0])
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distances[:, j] = 3958.8 * c  # Earth's radius in miles
        
        # Update minimum distances
        chunk_min = np.min(distances, axis=1)
        station_distances = np.minimum(station_distances, chunk_min)
    
    # Filter stations within corridor
    filtered_stations = []
    mask = station_distances <= corridor_width
    
    for idx, station in enumerate(fuel_stations):
        if mask[idx]:
            station_copy = station.copy()
            station_copy['route_distance'] = float(station_distances[idx])
            filtered_stations.append(station_copy)
    
    # Remove duplicates
    unique_stations = {}
    for station in filtered_stations:
        name = station['Truckstop Name']
        if name not in unique_stations or station['route_distance'] < unique_stations[name]['route_distance']:
            unique_stations[name] = station
    
    return list(unique_stations.values())

def analyze_price_zones(filtered_stations, total_distance):
    """Analyze price zones along the route"""
    if not filtered_stations:
        return [{'start': 0, 'end': total_distance, 'avg_price': 999999, 'min_price': 999999, 'max_price': 999999}]
    
    zones = []
    window_size = 200  # miles
    
    for i in range(0, int(total_distance), int(window_size/2)):
        zone_stations = [
            s for s in filtered_stations
            if i <= s['Distance'] < i + window_size
        ]
        if zone_stations:
            zones.append({
                'start': i,
                'end': i + window_size,
                'avg_price': sum(s['Retail Price'] for s in zone_stations) / len(zone_stations),
                'min_price': min(s['Retail Price'] for s in zone_stations),
                'max_price': max(s['Retail Price'] for s in zone_stations)
            })
    
    # Ensure at least one zone exists
    if not zones:
        zones = [{'start': 0, 'end': total_distance, 'avg_price': 999999, 'min_price': 999999, 'max_price': 999999}]
    
    return zones

def calculate_station_score(station, current_price_zone, next_price_zone, fuel_in_tank, distance_to_station, fuel_efficiency):
    """Calculate a weighted score for station selection"""
    if not current_price_zone:
        return station['Retail Price']  # Fallback to simple price comparison
    
    PRICE_WEIGHT = 0.4
    DISTANCE_WEIGHT = 0.3
    DETOUR_WEIGHT = 0.2
    ZONE_WEIGHT = 0.1
    
    # Price score (lower is better)
    price_score = station['Retail Price'] / current_price_zone['avg_price']
    
    # Distance score (prefer stations when tank is getting low)
    remaining_range = fuel_in_tank * fuel_efficiency
    distance_score = 1 - (remaining_range - distance_to_station) / remaining_range
    
    # Detour score (lower is better)
    detour_score = station['route_distance'] / 15  # Normalized to max allowed detour
    
    # Zone transition score
    zone_score = 1.0
    if next_price_zone and next_price_zone['min_price'] > current_price_zone['min_price']:
        zone_score = 0.7  # Bonus for filling up before more expensive zone
    
    final_score = (
        PRICE_WEIGHT * price_score +
        DISTANCE_WEIGHT * distance_score +
        DETOUR_WEIGHT * detour_score +
        ZONE_WEIGHT * zone_score
    )
    
    return final_score

def optimize_route_dp(stations: List[Dict], total_distance: float, 
                     fuel_efficiency: float, tank_capacity: float) -> List[Dict]:
    """Dynamic programming approach for optimal fuel stops"""
    stations.sort(key=lambda x: x['Distance'])
    n = len(stations)
    
    # dp[i] represents minimum cost to reach destination from station i
    dp = [float('inf')] * (n + 1)
    dp[n] = 0  # Base case: cost at destination is 0
    next_stop = [None] * (n + 1)
    
    for i in range(n-1, -1, -1):
        current = stations[i]
        max_reach = current['Distance'] + tank_capacity * fuel_efficiency
        
        for j in range(i+1, n+1):
            next_dist = total_distance if j == n else stations[j]['Distance']
            if next_dist - current['Distance'] <= tank_capacity * fuel_efficiency:
                fuel_needed = (next_dist - current['Distance']) / fuel_efficiency
                cost = fuel_needed * current['Retail Price'] + dp[j]
                
                if cost < dp[i]:
                    dp[i] = cost
                    next_stop[i] = j

    # Reconstruct path
    path = []
    current_idx = 0
    while current_idx is not None and current_idx < n:
        path.append(stations[current_idx])
        current_idx = next_stop[current_idx]
    
    return path

def parallel_station_filtering(stations: List[Dict], route_points: List[List[float]], 
                             max_distance: float) -> List[Dict]:
    """Parallel processing for station filtering"""
    def process_chunk(chunk):
        return [
            station for station in chunk
            if min(calculate_distance((station['Latitude'], station['Longitude']), 
                                   (point[0], point[1])) 
                  for point in route_points) <= max_distance
        ]
    
    # Split stations into chunks for parallel processing
    chunk_size = len(stations) // (os.cpu_count() or 4)
    chunks = [stations[i:i + chunk_size] for i in range(0, len(stations), chunk_size)]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        filtered_chunks = list(executor.map(process_chunk, chunks))
    
    return [station for chunk in filtered_chunks for station in chunk]

def predict_fuel_prices(stations: List[Dict], historical_data: Dict) -> Dict[str, float]:
    """Predict fuel prices using historical trends"""
    predictions = {}
    model = LinearRegression()
    
    for station in stations:
        if station['Truckstop Name'] in historical_data:
            hist = historical_data[station['Truckstop Name']]
            X = [[d.timestamp()] for d in hist['dates']]
            y = hist['prices']
            model.fit(X, y)
            
            # Predict price for next 24 hours
            future_time = datetime.now().timestamp() + 86400
            predictions[station['Truckstop Name']] = model.predict([[future_time]])[0]
    
    return predictions

def calculate_refueling_min_cost_enhanced(total_distance: float, max_range: float,
                                        fuel_efficiency: float, route_geometry: List[List[float]], 
                                        fuel_stations: List[Dict]) -> Tuple[List[Dict], float]:
    """Enhanced version of refueling optimization"""
    # Filter stations using parallel processing
    filtered_stations = parallel_station_filtering(fuel_stations, route_geometry, 50)
    
    # Calculate tank capacity from max_range and fuel_efficiency
    tank_capacity = max_range / fuel_efficiency
    
    # Get optimal stops using dynamic programming
    optimal_stops = optimize_route_dp(filtered_stations, total_distance, 
                                    fuel_efficiency, tank_capacity)
    
    # Calculate total cost and format results
    total_cost = sum(stop['Retail Price'] * ((next_stop['Distance'] - stop['Distance']) 
                    / fuel_efficiency if next_stop else 
                    (total_distance - stop['Distance']) / fuel_efficiency)
                    for stop, next_stop in zip(optimal_stops, optimal_stops[1:] + [None]))
    
    return optimal_stops, total_cost

def filter_stations_along_route(fuel_stations, route_points, max_distance=15):
    """Filter stations using KD-Tree for faster spatial queries"""
    # Convert stations and route points to numpy arrays
    station_coords = np.array([[s['Latitude'], s['Longitude']] for s in fuel_stations])
    route_coords = np.array(route_points)
    
    # Build KD-Tree for route points
    tree = cKDTree(route_coords)
    
    # Find stations within max_distance of route
    distances, indices = tree.query(station_coords, distance_upper_bound=max_distance/69.0)  # Convert miles to degrees
    
    # Create dictionary for station deduplication
    stations_by_location = {}
    
    for idx, station in enumerate(fuel_stations):
        if distances[idx] <= max_distance:
            location_key = f"{station['Latitude']:.4f}_{station['Longitude']:.4f}"
            if (location_key not in stations_by_location or 
                station['Retail Price'] < stations_by_location[location_key]['Retail Price']):
                station_copy = station.copy()
                station_copy['route_distance'] = distances[idx]
                # Calculate distance along route
                station_copy['Distance'] = calculate_distance(
                    (route_coords[0][0], route_coords[0][1]),  # Start point
                    (station['Latitude'], station['Longitude'])  # Station location
                )
                stations_by_location[location_key] = station_copy
    
    # Sort stations by distance along route
    stations_list = list(stations_by_location.values())
    stations_list.sort(key=lambda x: x['Distance'])
    
    return stations_list

def optimize_fuel_stops(filtered_stations, total_distance, fuel_efficiency, tank_capacity):
    """Optimize fuel stops using dynamic programming"""
    stations = sorted(filtered_stations, key=lambda x: x['Distance'])
    n = len(stations)
    
    # Minimum fuel needed at start
    initial_stations = [s for s in stations if s['Distance'] <= 100]
    if not initial_stations:
        raise ValueError("No stations found within first 100 miles")
    
    first_station = min(initial_stations, key=lambda x: x['Retail Price'])
    initial_fuel = (first_station['Distance'] / fuel_efficiency) + 1  # Add 1 gallon buffer
    
    # dp[i] represents minimum cost to reach destination from station i
    dp = [float('inf')] * (n + 1)
    dp[n] = 0
    prev = [None] * (n + 1)
    min_fuel = [0] * (n + 1)  # Track minimum fuel needed
    
    for i in range(n - 1, -1, -1):
        current = stations[i]
        max_reach = current['Distance'] + (tank_capacity * fuel_efficiency)
        
        # Try to reach next stations or destination
        for j in range(i + 1, n + 1):
            next_dist = total_distance if j == n else stations[j]['Distance']
            if next_dist - current['Distance'] <= tank_capacity * fuel_efficiency:
                fuel_needed = (next_dist - current['Distance']) / fuel_efficiency
                
                # Ensure minimum 5 gallons purchase unless it's the last stop
                if j < n and fuel_needed < 5:
                    continue
                    
                cost = fuel_needed * current['Retail Price']
                total_cost = cost + dp[j]
                
                if total_cost < dp[i]:
                    dp[i] = total_cost
                    prev[i] = j
                    min_fuel[i] = fuel_needed
    
    # Reconstruct the path
    path = []
    current = 0
    while current is not None and current < n:
        station = stations[current].copy()
        fuel_needed = min_fuel[current]
        
        if fuel_needed >= 5 or current == n-1:  # Only add stops with significant fuel purchases
            path.append({
                'station': {
                    'name': station['Truckstop Name'],
                    'address': station['Address'],
                    'price_per_gallon': station['Retail Price'],
                    'gallons_purchased': fuel_needed,
                    'cost': fuel_needed * station['Retail Price'],
                    'lat': station['Latitude'],
                    'lng': station['Longitude'],
                    'detour_distance': station['route_distance']
                },
                'position': (station['Latitude'], station['Longitude']),
                'distance_traveled': station['Distance']
            })
        
        current = prev[current]
    
    return path, dp[0], initial_fuel

def calculate_refueling_min_cost(total_distance, max_range, fuel_efficiency, route_geometry, fuel_stations):
    """Main function to calculate optimal refueling stops"""
    try:
        filtered_stations = filter_stations_along_route(fuel_stations, route_geometry)
        
        if not filtered_stations:
            raise ValueError("No fuel stations found along the route")
        
        # Calculate tank capacity from max_range and fuel_efficiency
        tank_capacity = max_range / fuel_efficiency
        
        stops, total_cost, initial_fuel = optimize_fuel_stops(
            filtered_stations,
            total_distance,
            fuel_efficiency,
            tank_capacity
        )
        
        return stops, total_cost, initial_fuel
    except Exception as e:
        print(f"Error in calculate_refueling_min_cost: {str(e)}")
        raise

# Keep only this line
__all__ = [
    'get_coordinates', 'get_route_data', 'calculate_distance', 'calculate_refueling_min_cost'
]




