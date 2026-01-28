import numpy as np

# --- ALINEA Controller Parameters ---
# Target density for the merge cell (veh/km) - this is the desired density ALINEA tries to maintain
# This corresponds to the critical density (or slightly below) where maximum flow is achieved.
# You need to tune this based on your road's fundamental diagram.
# For 12 cells, 5km road, CELL_LENGTH = 5000/12 = 416.67m.
# A density of 0.08 veh/cell translates to 0.08 / (CELL_LENGTH/1000) = 0.08 / 0.41667 = ~0.19 veh/km.
# If your cells represent a section of a single lane, and you want 20 veh/km, and CELL_LENGTH is 0.416km,
# then 20 veh/km * 0.416 km/cell = 8.32 vehicles per cell. This is very high.
# Let's assume TARGET_DENSITY is in "cars per cell" as computed by your simulation.
# Original TARGET_DENSITY = 0.08. If 1 car in cell 2 lanes -> density = 1/2 = 0.5.
# This was too low. Adjusted to aim for 1 car per lane segment (0.5 cars/lane).
TARGET_DENSITY = 0.5  # Adjusted: Aim for ~1 car in the merge cell (0.5 cars/lane)

# ALINEA gain parameter (Ki)
# This gain determines how aggressively the controller reacts to deviations from the target density.
# A higher Ki means faster reaction but can lead to oscillations.
# A lower Ki means slower reaction but more stable.
# Tune this value carefully.
ALINEA_GAIN = 0.05 # Adjusted from 0.5; a smaller gain for density control is generally more stable.

# Ramp flow limits
# R_MAX was 0.5, which prevented r_prev from ever being > 0.5, thus always showing "BLOCK".
# Increased R_MAX to 1.0 to allow for the "ALLOW" state to be reached.
R_MAX = 1.0  # Maximum ramp metering rate (e.g., 1.0 cars/second = 1 car every second)
R_MIN = 0.05 # Minimum ramp metering rate (e.g., 0.05 cars/second = 1 car every 20 seconds, allows some trickle)

# Congestion Threshold (cars per cell)
# If the merge cell density exceeds this, ALINEA can switch to a more restrictive "BLOCK" mode.
# Original CONGESTION_THRESHOLD_DENSITY = 0.12. If 1 car in cell 2 lanes -> density = 0.5.
# This was too low, causing immediate congestion. Adjusted to 1.0 (meaning 2 cars in the cell).
CONGESTION_THRESHOLD_DENSITY = 1.0 # Adjusted: If density in merge cell goes above 1.0 (2 cars), consider it congested.

# --- Global variable to store the previous metering rate ---
# This is crucial for ALINEA's iterative update
r_prev = R_MAX # Initialize with max rate, or a reasonable starting point

def compute_density(cars, cell_index):
    """
    Computes the average density (number of cars) in a specific cell across both mainline lanes.
    This function should match how your simulation counts cars in cells.

    Args:
        cars (list): List of car dictionaries.
        cell_index (int): The index of the cell for which to compute density.
                          Assumes cells are indexed from 0.

    Returns:
        float: The average number of cars in the specified cell per lane.
               (e.g., if 2 cars in cell 2 across 2 lanes, density is 1.0 car/lane)
    """
    cell_start_x = cell_index
    cell_end_x = cell_index + 1

    # Count cars in both mainline lanes (0 and 1)
    cars_in_cell = [car for car in cars if cell_start_x <= car["x"] < cell_end_x and car["lane"] in [0, 1]]

    # Density is number of cars in the cell / number of lanes (assuming 2 lanes for mainline)
    # If a cell represents a full road segment, then density = count / 1.
    # Given your visualization has 2 lanes, let's assume we divide by 2 here to get density per lane.
    # Adjust this if your 'density' is meant to be total cars in cell.
    num_lanes_mainline = 2
    if num_lanes_mainline > 0:
        return len(cars_in_cell) / num_lanes_mainline
    return 0.0


def congestion_at_merge(cars):
    """
    Checks if the merge cell (Cell 2) is congested based on a density threshold.
    This function is used to inform ALINEA about severe congestion.

    Args:
        cars (list): List of car dictionaries.

    Returns:
        bool: True if the merge cell is congested, False otherwise.
    """
    rho_merge_cell = compute_density(cars, cell_index=2)
    return rho_merge_cell > CONGESTION_THRESHOLD_DENSITY


def alinea(r_prev_in, rho_merge_cell, congestion=False):
    """
    Implements the ALINEA ramp metering algorithm.

    Args:
        r_prev_in (float): The metering rate applied in the previous control interval.
        rho_merge_cell (float): The measured density in the merge cell (Cell 2) at the current time.
        congestion (bool): True if the merge cell is deemed severely congested.

    Returns:
        tuple: (new_r_prev, raw_r_value)
               new_r_prev (float): The new metering rate to be applied (limited by R_MIN, R_MAX).
               raw_r_value (float): The calculated metering rate before applying limits.
    """
    global r_prev # Declare global to modify the module-level r_prev

    # ALINEA control law
    # r(k) = r(k-1) + Ki * (rho_target - rho_k)
    raw_r = r_prev_in + ALINEA_GAIN * (TARGET_DENSITY - rho_merge_cell)

    # Apply limits
    new_r = max(R_MIN, min(R_MAX, raw_r))

    # Congestion override: if severe congestion, set rate to minimum (BLOCK)
    if congestion:
        new_r = R_MIN

    r_prev = new_r # Update the global r_prev for the next iteration
    return new_r, raw_r
