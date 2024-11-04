from django.db import models

class FuelStation(models.Model):
    truckstop_name = models.CharField(max_length=200)
    address = models.CharField(max_length=500)
    retail_price = models.DecimalField(max_digits=5, decimal_places=2)
    latitude = models.FloatField()
    longitude = models.FloatField()

    class Meta:
        db_table = 'fuel_stations'

    def __str__(self):
        return self.truckstop_name

class Route(models.Model):
    start_location = models.CharField(max_length=200)
    end_location = models.CharField(max_length=200)
    total_distance = models.FloatField()
    total_cost = models.DecimalField(max_digits=10, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
