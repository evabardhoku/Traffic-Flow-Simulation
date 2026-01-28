# SENSORS.PY (Revised for Unit Consistency)

import pandas as pd
import numpy as np  # Make sure numpy is imported here

class Sensor:
    def __init__(self, cell_index, seconds_per_frame, cell_length_meters, is_off_ramp_sensor=False):
        self.cell_index = cell_index
        self.data = []
        self.active = False
        self.car_counter = 0

        self.seconds_per_frame = seconds_per_frame
        self.cell_length_meters = cell_length_meters

        self.is_off_ramp_sensor = is_off_ramp_sensor  # NEW: flag for off-ramp detection

    def start(self):
        self.active = True
        self.car_counter = 0
        self.data = []

    def stop(self):
        self.active = False

    def measure(self, cars, frame):
        if not self.active:
            return

        # Define cars based on sensor type
        if self.is_off_ramp_sensor:
            # --- OFF-RAMP SENSOR LOGIC ---
            # Detect cars that are in the off-ramp lane and near the end of off-ramp
            in_cell = [car for car in cars if car.get("lane") == "off_ramp" and int(car["x"]) == int(self.cell_index)]
        else:
            # --- MAINLINE SENSOR LOGIC ---
            in_cell = [car for car in cars if int(car["x"]) == self.cell_index and car.get("lane") in [0, 1]]

        # --- Density Calculation ---
        density_veh_per_meter = len(in_cell) / self.cell_length_meters

        # --- Speed Calculation ---
        car_speeds_m_per_s = []
        if in_cell:
            for car in in_cell:
                speed_cells_per_frame = car["v"]
                speed_meters_per_second = (speed_cells_per_frame * self.cell_length_meters) / self.seconds_per_frame
                car_speeds_m_per_s.append(speed_meters_per_second)
            avg_speed_m_per_s = np.mean(car_speeds_m_per_s)
        else:
            avg_speed_m_per_s = 0

        # --- Flow Calculation ---
        if self.is_off_ramp_sensor:
            # Count cars exiting off-ramp (rough estimation)
            exiting_cars = [car for car in cars if
                            car.get("lane") == "off_ramp" and
                            int(car["x"] - car["v"]) <= self.cell_index and int(car["x"]) > self.cell_index]
        else:
            exiting_cars = [car for car in cars if
                            int(car["x"] - car["v"]) == self.cell_index and int(car["x"]) > self.cell_index]

        flow_out_veh_per_frame = len(exiting_cars)
        self.car_counter += flow_out_veh_per_frame

        # --- Ramp inflow (only for on-ramp adjacent sensors) ---
        ramp_in = len([car for car in cars if car.get("lane") == "ramp" and int(car["x"]) == self.cell_index])

        self.data.append({
            "car_id": self.car_counter,
            "frame": frame,
            "density": density_veh_per_meter,
            "speed": avg_speed_m_per_s,
            "flow_out": flow_out_veh_per_frame,
            "ramp_inflow": ramp_in if not self.is_off_ramp_sensor else 0  # off-ramp sensors do not track ramp inflow
        })

    def get_table(self):
        return pd.DataFrame(self.data)
