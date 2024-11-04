from django.core.management.base import BaseCommand
from fuel_stations.models import FuelStation
import csv

class Command(BaseCommand):
    help = 'Load fuel stations data from CSV'

    def handle(self, *args, **options):
        # Clear existing data
        FuelStation.objects.all().delete()

        # Path to your CSV file
        csv_file = 'C:\\Users\\aatis\\Desktop\\TRUCK\\maps\\fuel_stations\\fuel-prices-cleaned.csv'
        
        try:
            with open(csv_file, 'r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    FuelStation.objects.create(
                        truckstop_name=row['Truckstop Name'],
                        address=row['Address'],
                        retail_price=float(row['Retail Price']),
                        latitude=float(row['Latitude']),
                        longitude=float(row['Longitude'])
                    )
            
            self.stdout.write(self.style.SUCCESS('Successfully loaded fuel stations data'))
            
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Error loading data: {str(e)}'))
