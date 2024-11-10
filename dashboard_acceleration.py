import lightningchart as lc
import random
import time

# Set license key
lc.set_license("LICENSE_KEY")

# Initialize dashboard
dashboard = lc.Dashboard(columns=3, rows=3, theme=lc.Themes.TurquoiseHexagon)

fuel_chart = dashboard.ChartXY(column_index=0, row_index=0, row_span=2)
fuel_chart.set_title('Fuel Consumption (L/100 km)')
fuel_series = fuel_chart.add_line_series()
fuel_chart.get_default_x_axis().set_title('Time (s)')  # Label for relative time in seconds

speed_gauge = dashboard.GaugeChart(column_index=1, row_index=0, row_span=2)
speed_gauge.set_title('Vehicle Speed (km/h)')
speed_gauge.set_interval(0, 200)
speed_gauge.set_angle_interval(start=180, end=0)
speed_gauge.set_bar_thickness(8)
speed_gauge.set_value_indicator_thickness(2)

efficiency_chart = dashboard.ChartXY(column_index=2, row_index=0, row_span=2)
efficiency_chart.set_title('Fuel Efficiency (L/km)')
efficiency_series = efficiency_chart.add_line_series()
efficiency_chart.get_default_x_axis().set_title('Time (s)')

distance_chart = dashboard.ChartXY(column_index=0, row_index=2, column_span=3)
distance_chart.set_title('Distance Traveled (km)')
distance_series = distance_chart.add_line_series()
distance_chart.get_default_x_axis().set_title('Time (s)')

dashboard.open(live=True) 

# Initialize simulation variables
elapsed_time = 0  # Start time at 0 seconds
fuel_consumption = 0  # Start from 0, gradually increase
speed = 0  # Start from 0, gradually increase
distance = 0
efficiency = 0  # Start from 0, calculate based on fuel consumption

# Set a target speed range for fluctuation
target_speed_min = 80
target_speed_max = 100
accelerating = True  # Flag to control acceleration phase

# Simulation loop for real-time updates
for i in range(1000):
    # Use elapsed time in seconds as the x-axis value
    elapsed_time += 1

    # Simulate vehicle speed with initial acceleration up to target range, then fluctuation
    if accelerating:
        speed += random.uniform(0.5, 1.5)  # Increase speed gradually
        if speed >= target_speed_min:
            accelerating = False  # Stop accelerating once we hit the target range
    else:
        speed += random.uniform(-2, 2)  # Fluctuate speed within target range
        speed = max(target_speed_min, min(speed, target_speed_max))  # Keep within range
    
    speed_gauge.set_value(speed)

    # Simulate fuel consumption based on speed
    if speed > 0:
        fuel_consumption = 5 + (speed / 40)  # Increase fuel consumption with speed
    fuel_series.append_sample(elapsed_time, fuel_consumption)

    # Calculate and update fuel efficiency based on fuel consumption
    efficiency = fuel_consumption / 100 if fuel_consumption > 0 else 0
    efficiency_series.append_sample(elapsed_time, efficiency)

    # Calculate and update distance traveled (cumulative)
    distance += (speed * (1/3600))  # Convert speed to distance per second (km)
    distance_series.append_sample(elapsed_time, distance)

    # Simulate a time delay for real-time effect
    time.sleep(0.1)