from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import FuelStationViewSet, RouteOptimizationViewSet

router = DefaultRouter()
router.register(r'stations', FuelStationViewSet)
router.register(r'optimize', RouteOptimizationViewSet, basename='optimize')

urlpatterns = [
    path('', include(router.urls)),
]
