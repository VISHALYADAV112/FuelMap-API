# FuelMap-API


A Django-based API for calculating optimized routes across the USA with cost-effective fuel stops along the way. The API includes functionality to estimate fuel expenses, plan efficient stops based on fuel prices, and generate a map of the route.

## Table of Contents
- [Project Setup](#project-setup)
- [API Endpoints](#api-endpoints)
  - [Get All Fuel Stations](#get-all-fuel-stations)
  - [Route Optimization Documentation](#route-optimization-documentation)
  - [Calculate Route with Fuel Stops](#calculate-route-with-fuel-stops)
  - [Get Route Map](#get-route-map)
- [Usage Examples](#usage-examples)
- [Error Handling](#error-handling)
- [API Documentation Interface](#api-documentation-interface)
- [Notes](#notes)

---

## Project Setup

To get started, set up the Django environment and load the fuel station data:

1. Migrate the database:
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   ```

2. Load initial fuel station data:
   ```bash
   python manage.py loaddata fuel_stations.json
   ```

## API Endpoints

### 1. Get All Fuel Stations
- **Endpoint**: `GET /api/stations/`
- **Description**: Retrieves a list of all fuel stations.
- **Response**:
    ```json
    [
        {
            "id": 1,
            "truckstop_name": "Sample Station",
            "address": "123 Main St",
            "retail_price": "3.50",
            "latitude": 34.0522,
            "longitude": -118.2437
        },
        ...
    ]
    ```

### 2. Route Optimization Documentation
- **Endpoint**: `GET /api/optimize/optimize/`
- **Description**: Provides instructions on using the route optimization API.
- **Response**:
    ```json
    {
        "message": "Please use POST request with the following format",
        "sample_request": {
            "start_location": "Atlanta, GA",
            "end_location": "Las Vegas, NV",
            "tank_capacity": 50,
            "fuel_efficiency": 10
        }
    }
    ```

### 3. Calculate Route with Fuel Stops
- **Endpoint**: `POST /api/optimize/optimize/`
- **Description**: Calculates the optimal route with suggested fuel stops.
- **Request Body**:
    ```json
    {
        "start_location": "Atlanta, GA",
        "end_location": "Las Vegas, NV",
        "tank_capacity": 50,
        "fuel_efficiency": 10
    }
    ```
- **Response**:
    ```json
    {
        "success": true,
        "start_location": "Atlanta, GA",
        "end_location": "Las Vegas, NV",
        "initial_fuel": 25.5,
        "total_cost": 450.75,
        "total_distance": 1500.25,
        "estimated_time": 24.5,
        "stops": [
            {
                "station": {
                    "name": "Truck Stop A",
                    "address": "456 Highway Rd",
                    "price_per_gallon": 3.45,
                    "gallons_purchased": 45.2,
                    "cost": 155.94,
                    "lat": 35.1234,
                    "lng": -115.5678,
                    "detour_distance": 0.5
                },
                "position": [35.1234, -115.5678],
                "distance_traveled": 500.25
            },
            ...
        ],
        "map_html": "<html>...</html>"
    }
    ```

### 4. Get Route Map
- **Endpoint**: `GET /api/optimize/get_map/`
- **Description**: Returns an HTML map for a specific route with fuel stops marked.
- **Query Parameters**:
    - `start`: Starting location (required)
    - `end`: Ending location (required)
    - `tank_capacity`: Tank capacity in gallons (optional, default: 50)
    - `fuel_efficiency`: Fuel efficiency in miles per gallon (optional, default: 10)
- **Example**: `/api/optimize/get_map/?start=Atlanta,GA&end=Las Vegas,NV&tank_capacity=50&fuel_efficiency=10`

## Usage Examples

Using `curl`:

1. **Get all stations**:
    ```bash
    curl -X GET http://localhost:8000/api/stations/
    ```

2. **Get optimization documentation**:
    ```bash
    curl -X GET http://localhost:8000/api/optimize/optimize/
    ```

3. **Calculate optimal route**:
    ```bash
    curl -X POST http://localhost:8000/api/optimize/optimize/ \
    -H "Content-Type: application/json" \
    -d '{
        "start_location": "Atlanta, GA",
        "end_location": "Las Vegas, NV",
        "tank_capacity": 50,
        "fuel_efficiency": 10
    }'
    ```

4. **Get route map**:
    ```bash
    curl -X GET "http://localhost:8000/api/optimize/get_map/?start=Atlanta,GA&end=Las Vegas,NV"
    ```

## Error Handling

The API uses standard HTTP status codes:
- **200**: Success
- **400**: Bad Request (e.g., invalid parameters)
- **500**: Server Error

### Example Error Response
```json
{
    "success": false,
    "error": "Error description"
}
```

## API Documentation Interface

A detailed, interactive documentation interface is available at `/docs/`. It includes endpoint descriptions, request/response formats, and allows API testing.

## Notes

- **Distance** is measured in miles.
- **Fuel capacity** is in gallons.
- **Fuel efficiency** is in miles per gallon.
- **Prices** are in USD.
- **Coordinates** are in decimal degrees.
- The API integrates with **OpenCage** for geocoding and **GraphHopper** for routing, requiring API keys in the project settings.

--- 

Feel free to customize further! This structure should provide users with a comprehensive and easy-to-navigate overview.
