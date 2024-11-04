from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from .models import FuelStation, Route
from .serializers import FuelStationSerializer, RouteSerializer, RouteOptimizationRequestSerializer
from .routeoptimization import (
    calculate_refueling_min_cost, get_route_data, get_coordinates, 
    generate_map_and_route_data
)
from django.conf import settings
from django.http import HttpResponse
import folium

class FuelStationViewSet(viewsets.ModelViewSet):
    queryset = FuelStation.objects.all()
    serializer_class = FuelStationSerializer
    http_method_names = ['get']  # Only allow GET requests

class RouteOptimizationViewSet(viewsets.ViewSet):
    @action(detail=False, methods=['get', 'post'])
    def optimize(self, request):
        if request.method == 'GET':
            # Handle GET request - return sample response or documentation
            return Response({
                "message": "Please use POST request with the following format",
                "sample_request": {
                    "start_location": "Atlanta, GA",
                    "end_location": "Las Vegas, NV",
                    "tank_capacity": 50,
                    "fuel_efficiency": 10
                }
            })
        
        # Handle POST request
        try:
            data = request.data
            
            # Get coordinates for locations
            start_coords = get_coordinates(data['start_location'], settings.OPENCAGE_API_KEY)
            end_coords = get_coordinates(data['end_location'], settings.OPENCAGE_API_KEY)
            
            if not start_coords or not end_coords:
                return Response({
                    'success': False,
                    'error': 'Could not find coordinates for given locations'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Get route data
            route_data = get_route_data(start_coords, end_coords, settings.GRAPHHOPPER_API_KEY)

            # Get fuel stations
            fuel_stations = list(FuelStation.objects.values(
                'truckstop_name', 'address', 'retail_price', 'latitude', 'longitude'
            ))

            # Convert QuerySet to list of dicts with proper keys
            stations_list = [
                {
                    'Truckstop Name': station['truckstop_name'],
                    'Address': station['address'],
                    'Retail Price': float(station['retail_price']),
                    'Latitude': station['latitude'],
                    'Longitude': station['longitude']
                }
                for station in fuel_stations
            ]

            # Calculate optimal route
            tank_capacity = float(data.get('tank_capacity', 50))  # default 50 gallons
            fuel_efficiency = float(data.get('fuel_efficiency', 10))  # default 10 mpg
            
            stops, total_cost, initial_fuel = calculate_refueling_min_cost(
                route_data['total_distance_miles'],
                tank_capacity * fuel_efficiency,
                fuel_efficiency,
                route_data['route_points'],
                stations_list
            )

            # Generate map
            map_obj = generate_map_and_route_data(
                start_coords, 
                end_coords, 
                stops, 
                route_data['route_points']
            )
            
            # Save map to HTML
            map_html = map_obj._repr_html_()

            # Save route to database
            Route.objects.create(
                start_location=data['start_location'],
                end_location=data['end_location'],
                total_distance=route_data['total_distance_miles'],
                total_cost=total_cost
            )

            response_data = {
                'success': True,
                'start_location': data['start_location'],
                'end_location': data['end_location'],
                'initial_fuel': initial_fuel,
                'total_cost': total_cost,
                'total_distance': route_data['total_distance_miles'],
                'estimated_time': route_data['estimated_time_hours'],
                'stops': stops,
                'map_html': map_html  # Include the map HTML in response
            }

            return Response(response_data)

        except Exception as e:
            return Response({
                'success': False,
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    @action(detail=False, methods=['get'])
    def get_map(self, request):
        """Return just the map for a specific route"""
        try:
            start_location = request.query_params.get('start')
            end_location = request.query_params.get('end')
            tank_capacity = float(request.query_params.get('tank_capacity', 50))
            fuel_efficiency = float(request.query_params.get('fuel_efficiency', 10))
            
            if not start_location or not end_location:
                return Response({
                    'error': 'Start and end locations are required'
                }, status=status.HTTP_400_BAD_REQUEST)

            # Get coordinates
            start_coords = get_coordinates(start_location, settings.OPENCAGE_API_KEY)
            end_coords = get_coordinates(end_location, settings.OPENCAGE_API_KEY)
            
            # Get route data
            route_data = get_route_data(start_coords, end_coords, settings.GRAPHHOPPER_API_KEY)

            # Get fuel stations and calculate optimal route
            fuel_stations = list(FuelStation.objects.values(
                'truckstop_name', 'address', 'retail_price', 'latitude', 'longitude'
            ))

            stations_list = [
                {
                    'Truckstop Name': station['truckstop_name'],
                    'Address': station['address'],
                    'Retail Price': float(station['retail_price']),
                    'Latitude': station['latitude'],
                    'Longitude': station['longitude']
                }
                for station in fuel_stations
            ]

            # Calculate optimal route
            stops, total_cost, initial_fuel = calculate_refueling_min_cost(
                route_data['total_distance_miles'],
                tank_capacity * fuel_efficiency,
                fuel_efficiency,
                route_data['route_points'],
                stations_list
            )

            # Generate map using the full function
            map_obj = generate_map_and_route_data(
                start_coords,
                end_coords,
                stops,
                route_data['route_points']
            )

            # Return the map as HTML
            response = HttpResponse(map_obj._repr_html_())
            response['Content-Type'] = 'text/html'
            return response

        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)