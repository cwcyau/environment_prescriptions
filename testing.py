import numpy as np
import matplotlib.pyplot as plt

a = ["sigma", "Intercept", "hydro_rain_values", "met_rain_values", "met_tmax_values", "aqrean_carbon_monoxide_values",
 "aqrean_daqi_overall_values", "aqrean_nitrogen_monoxide_values", "aqrean_nitrogen_dioxide_values",
 "aqrean_daqi_nitrogen_dioxide_values", "aqrean_nox_expressed_as_nitrogen_dioxide_values", "aqrean_ozone_values",
 "aqrean_daqi_ozone_values", "aqrean_pm2p5_values", "aqrean_pm10_values", "aqrean_daqi_pm10_values", "aqrean_sulfur_dioxide_values",
 "aqrean_daqi_sulfur_dioxide_values",
 "hydro_rain_values:met_rain_values", "met_rain_values:met_tmax_values", "hydro_rain_values:met_tmax_values",
 "aqrean_nitrogen_dioxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
 "aqrean_nitrogen_monoxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
 "aqrean_daqi_overall_values:aqrean_daqi_nitrogen_dioxide_values", "aqrean_daqi_overall_values:aqrean_daqi_ozone_values",
 "aqrean_daqi_overall_values:aqrean_daqi_pm10_values", "aqrean_daqi_overall_values:aqrean_daqi_sulfur_dioxide_values",
 "aqrean_daqi_nitrogen_dioxide_values:aqrean_nitrogen_dioxide_values", "aqrean_daqi_ozone_values:aqrean_ozone_values",
 "aqrean_daqi_pm10_values:aqrean_pm10_values", "aqrean_daqi_sulfur_dioxide_values:aqrean_sulfur_dioxide_values",
 "C(month)", "1|practice_id_sigma", "1|practice_id_offset", "date_code|practice_id_sigma", "date_code|practice_id_offset"]

b = ["sigma", "Intercept", "C(flood)", "hydro_rain_values", "met_rain_values", "met_tmax_values", "met_tmin_values",
     "aqrean_carbon_monoxide_values", "aqrean_daqi_overall_values", "aqrean_nitrogen_monoxide_values",
     "aqrean_nitrogen_dioxide_values", "aqrean_daqi_nitrogen_dioxide_values", "aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "aqrean_ozone_values", "aqrean_daqi_ozone_values", "aqrean_pm2p5_values", "aqrean_daqi_pm2p5_values", "aqrean_pm10_values",
     "aqrean_daqi_pm10_values", "aqrean_sulfur_dioxide_values", "aqrean_daqi_sulfur_dioxide_values", "C(flood):hydro_rain_values",
     "C(flood):met_rain_values", "hydro_rain_values:met_rain_values", "met_tmax_values:aqrean_carbon_monoxide_values",
     "met_tmax_values:aqrean_nitrogen_monoxide_values", "met_tmax_values:aqrean_nitrogen_dioxide_values",
     "met_tmax_values:aqrean_nox_expressed_as_nitrogen_dioxide_values", "met_tmax_values:aqrean_ozone_values",
     "met_tmin_values:aqrean_carbon_monoxide_values", "met_tmin_values:aqrean_nitrogen_monoxide_values",
     "met_tmin_values:aqrean_nitrogen_dioxide_values", "met_tmin_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "met_tmin_values:aqrean_ozone_values", "aqrean_carbon_monoxide_values:aqrean_nitrogen_monoxide_values",
     "aqrean_carbon_monoxide_values:aqrean_nitrogen_dioxide_values",
     "aqrean_carbon_monoxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "aqrean_daqi_overall_values:aqrean_pm2p5_values", "aqrean_daqi_overall_values:aqrean_daqi_pm2p5_values",
     "aqrean_daqi_overall_values:aqrean_pm10_values", "aqrean_daqi_overall_values:aqrean_daqi_pm10_values",
     "aqrean_nitrogen_monoxide_values:aqrean_nitrogen_dioxide_values",
     "aqrean_nitrogen_monoxide_values:aqrean_daqi_nitrogen_dioxide_values",
     "aqrean_nitrogen_monoxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "aqrean_nitrogen_monoxide_values:aqrean_ozone_values", "aqrean_nitrogen_monoxide_values:aqrean_sulfur_dioxide_values",
     "aqrean_nitrogen_dioxide_values:aqrean_daqi_nitrogen_dioxide_values",
     "aqrean_nitrogen_dioxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "aqrean_nitrogen_dioxide_values:aqrean_sulfur_dioxide_values",
     "aqrean_daqi_nitrogen_dioxide_values:aqrean_nox_expressed_as_nitrogen_dioxide_values",
     "aqrean_nox_expressed_as_nitrogen_dioxide_values:aqrean_ozone_values",
     "aqrean_nox_expressed_as_nitrogen_dioxide_values:aqrean_sulfur_dioxide_values",
     "aqrean_ozone_values:aqrean_daqi_ozone_values", "aqrean_pm2p5_values:aqrean_daqi_pm2p5_values",
     "aqrean_pm2p5_values:aqrean_pm10_values", "aqrean_pm2p5_values:aqrean_daqi_pm10_values",
     "aqrean_daqi_pm2p5_values:aqrean_pm10_values", "aqrean_daqi_pm2p5_values:aqrean_daqi_pm10_values",
     "aqrean_pm10_values:aqrean_daqi_pm10_values", "aqrean_sulfur_dioxide_values:aqrean_daqi_sulfur_dioxide_values",
     "C(month)", "1|practice_id_sigma", "1|practice_id_offset", "date_code|practice_id_sigma", "date_code|practice_id_offset"]

a = [i for i in a if ":" in i]
b = [i for i in b if ":" in i]
print(len(a), len(b))