from django.urls import path
from . import views

urlpatterns = [
    # ── Auth ──────────────────────────────
    path('register/', views.registerPage, name='register'),
    path('login/',    views.loginPage,    name='login'),
    path('logout/',   views.logoutUser,   name='logout'),

    # ── Main ──────────────────────────────
    path('',          views.home,         name='home'),

    # ── Prediction ────────────────────────
    path('predict/',  views.predictImage, name='predictImage'),

    path('home2/',  views.home2, name='home2'),
    path('predict1/',  views.predictImage1, name='predictImage1'),

]