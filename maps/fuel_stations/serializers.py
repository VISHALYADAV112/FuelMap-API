from rest_framework import serializers
from .models import FuelStation, Route

class FuelStationSerializer(serializers.ModelSerializer):
    class Meta:
        model = FuelStation
        fields = '__all__'

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = '__all__'

class RouteOptimizationRequestSerializer(serializers.Serializer):
    start_location = serializers.CharField()
    end_location = serializers.CharField()
    tank_capacity = serializers.FloatField(default=50)
    fuel_efficiency = serializers.FloatField(default=10)
