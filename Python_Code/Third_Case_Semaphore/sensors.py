# SENSORS.PY (Revised for Unit Consistency)

import pandas as pd
import numpy as np  # Make sure numpy is imported here


class Sensor:
    # Modified __init__ to accept simulation constants
    def __init__(self, cell_index, seconds_per_frame, cell_length_meters):
        self.cell_index = cell_index
        self.data = []
        self.active = False
        self.car_counter = 0

        # Store these simulation constants for accurate measurements
        self.seconds_per_frame = seconds_per_frame
        self.cell_length_meters = cell_length_meters

    def start(self):
        """Starts recording data and resets the car counter and data list."""
        self.active = True
        self.car_counter = 0
        self.data = []  # Clear previous recording data when starting a new one

    def stop(self):
        """Stops recording data."""
        self.active = False

    def measure(self, cars, frame):
        """
        Measures traffic parameters (density, speed, flow) for the sensor's cell
        and appends them to the sensor's data.
        """
        if not self.active:
            return

        # Get cars currently in this cell (mainline lanes only)
        # Assumes car["x"] is in terms of cell index (e.g., cell 0, 1, 2...)
        in_cell = [car for car in cars if int(car["x"]) == self.cell_index and car.get("lane") in [0, 1]]

        # --- Calculate Density (vehicles per meter) ---
        # Density is the number of vehicles in the cell divided by the cell's length in meters.
        # This will be in veh/meter, which metanet_analysis.py expects before converting to veh/km.
        density_veh_per_meter = len(in_cell) / self.cell_length_meters

        # --- Calculate Average Speed (meters per second) ---
        # car["v"] from the main simulation is in 'cells per frame'.
        # To convert to m/s: (cells/frame * meters/cell) / (seconds/frame)
        car_speeds_m_per_s = []
        if in_cell:  # Avoid division by zero if no cars in cell
            for car in in_cell:
                speed_cells_per_frame = car["v"]
                speed_meters_per_second = (speed_cells_per_frame * self.cell_length_meters) / self.seconds_per_frame
                car_speeds_m_per_s.append(speed_meters_per_second)
            avg_speed_m_per_s = np.mean(car_speeds_m_per_s)
        else:
            avg_speed_m_per_s = 0  # No cars in cell, so average speed is 0

        # --- Estimate Flow Out (vehicles per frame) ---
        # Count cars that were in this cell in the previous frame and have now moved out.
        # This is an approximation of flow passing the sensor point.
        exiting_cars = [car for car in cars if
                        int(car["x"] - car["v"]) == self.cell_index and int(car["x"]) > self.cell_index]
        flow_out_veh_per_frame = len(exiting_cars)

        # --- Ramp Inflow (vehicles per frame) ---
        # This counts ramp cars that are currently in the sensor's cell.
        # If the sensor is at the merge, this can indicate ramp inflow.
        ramp_in = len([car for car in cars if car.get("lane") == "ramp" and int(car["x"]) == self.cell_index])

        # Update car counter (total number of cars that have exited this cell)
        self.car_counter += flow_out_veh_per_frame

        # Append the calculated metrics to the sensor's data list
        self.data.append({
            "car_id": self.car_counter,  # Total car index (1 to x) passing this sensor
            "frame": frame,  # Simulation frame number
            "density": density_veh_per_meter,  # Density in vehicles per meter
            "speed": avg_speed_m_per_s,  # Speed in meters per second
            "flow_out": flow_out_veh_per_frame,  # Flow in vehicles per frame
            "ramp_inflow": ramp_in  # Ramp inflow in vehicles per frame
        })

    def get_table(self):
        """Returns the collected sensor data as a Pandas DataFrame."""
        return pd.DataFrame(self.data)