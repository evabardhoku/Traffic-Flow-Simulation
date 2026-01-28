# metanet_analysis.py

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

# --- IMPORTANT: Ensure traffic_demand.py is in the same directory,
# --- or adjust the import path below.
try:
    from traffic_demand import format_time
    # SECONDS_PER_FRAME MUST BE THE SAME AS IN YOUR MAIN SIMULATION!
    # We will pass this value from the main script.
    # Placeholders here, will be overridden by passed values
    GLOBAL_SECONDS_PER_FRAME = 1.0
    GLOBAL_CELL_LENGTH_METERS = (5.0 * 1000) / 12
    GLOBAL_ROAD_LENGTH_KM = 5.0

except ImportError:
    print("Warning: Could not import format_time from traffic_demand.py.")
    print("         Please ensure traffic_demand.py is in the same directory or its path is correctly set.")
    GLOBAL_SECONDS_PER_FRAME = 1.0 # Default to 1.0s per frame if import fails
    GLOBAL_CELL_LENGTH_METERS = 416.66 # Default
    GLOBAL_ROAD_LENGTH_KM = 5.0
    def format_time(frame_seconds):
        hours = int(frame_seconds // 3600)
        minutes = int((frame_seconds % 3600) // 60)
        seconds = int(frame_seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

EXPORT_FOLDER = "exports"
all_sensor_data = []  # container for global analysis (will store aggregated DFs)

def ensure_export_dir():
    """Ensures the export directory exists."""
    if not os.path.exists(EXPORT_FOLDER):
        os.makedirs(EXPORT_FOLDER)

def handle_sensor_data(sensor_id, raw_df, seconds_per_frame, cell_length_meters, road_length_km):
    """
    Processes raw sensor data by aggregating it into time intervals for smoother analysis.
    This function will now perform the averaging.
    """
    global GLOBAL_SECONDS_PER_FRAME, GLOBAL_CELL_LENGTH_METERS, GLOBAL_ROAD_LENGTH_KM
    GLOBAL_SECONDS_PER_FRAME = seconds_per_frame
    GLOBAL_CELL_LENGTH_METERS = cell_length_meters
    GLOBAL_ROAD_LENGTH_KM = road_length_km

    ensure_export_dir()

    print(f"Processing raw data for Sensor {sensor_id}...")

    if raw_df.empty:
        print(f"No raw data recorded for Sensor {sensor_id}. Skipping processing.")
        all_sensor_data.append(pd.DataFrame()) # Add empty df to maintain list structure
        return

    # Convert raw frame-level data to standard units
    df = raw_df.copy() # Work on a copy
    df["sim_seconds"] = df["frame"] * GLOBAL_SECONDS_PER_FRAME
    df["speed_mps"] = df["speed"] # speed from sensor is already in m/s
    df["density_vpm"] = df["density"] # density from sensor is already in veh/m

    # --- Define Aggregation Interval ---
    # This is crucial for smoothing the data.
    # For example, aggregate every 5 minutes (300 seconds)
    # A smaller interval like 60 seconds (1 minute) might give more points
    # for Fundamental Diagrams if simulation time is limited.
    AGGREGATION_INTERVAL_SECONDS = 60 # Changed to 1 minute for more points

    # Create time bins
    df['time_bin_start_seconds'] = (df['sim_seconds'] // AGGREGATION_INTERVAL_SECONDS) * AGGREGATION_INTERVAL_SECONDS

    # --- Group and Aggregate Data ---
    aggregated_df = df.groupby('time_bin_start_seconds').agg(
        density_veh_per_meter=('density_vpm', 'mean'),
        avg_speed_m_per_s=('speed_mps', 'mean'),
        total_flow_count_in_bin=('flow_out', 'sum'),
        total_ramp_inflow_in_bin=('ramp_inflow', 'sum')
    ).reset_index()

    # --- Convert Aggregated Data to Desired Units ---
    aggregated_df["sim_hours"] = aggregated_df["time_bin_start_seconds"] / 3600.0
    aggregated_df["speed_kmh"] = aggregated_df["avg_speed_m_per_s"] * 3.6 # m/s to km/h
    aggregated_df["density_veh_per_km"] = aggregated_df["density_veh_per_meter"] * 1000 # veh/m to veh/km

    # Calculate Flow (veh/hour) for the aggregated interval
    aggregated_df["flow_veh_per_hour"] = (aggregated_df["total_flow_count_in_bin"] / AGGREGATION_INTERVAL_SECONDS) * 3600
    aggregated_df["ramp_inflow_veh_per_hour"] = (aggregated_df["total_ramp_inflow_in_bin"] / AGGREGATION_INTERVAL_SECONDS) * 3600

    aggregated_df = aggregated_df.fillna(0.0)

    filename = f"{EXPORT_FOLDER}/sensor_{sensor_id}_aggregated_data.csv"
    aggregated_df.to_csv(filename, index=False)
    print(f"✅ Sensor {sensor_id} aggregated data exported to {filename} ({len(aggregated_df)} aggregated points)")

    aggregated_df["sensor_id"] = sensor_id
    all_sensor_data.append(aggregated_df)

    plot_sensor_analysis(sensor_id, aggregated_df)

def plot_sensor_analysis(sensor_id, df):
    """
    Generates a suite of plots for a single sensor's data:
    - Time-series plots (Flow, Density, Speed over Time)
    - Fundamental Diagrams (Flow-Density, Speed-Density, Flow-Speed)
    - 3D surface plots (Optional, but can be useful if data is dense enough)
    """
    sim_hours = df["sim_hours"].values
    speed_kmh = df["speed_kmh"].values
    density_veh_per_km = df["density_veh_per_km"].values
    flow_veh_per_hour = df["flow_veh_per_hour"].values

    if len(df) < 2:
        print(f"⚠️ Not enough aggregated data ({len(df)} points) for any plots. Skipping for sensor {sensor_id}.")
        return

    estimated_critical_density = df["density_veh_per_km"].quantile(0.75) if not df["density_veh_per_km"].empty else 45.0
    if estimated_critical_density == 0: estimated_critical_density = 45.0

    # --- Time-Series Plots ---
    fig_ts, axs_ts = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig_ts.suptitle(f"Sensor {sensor_id} - Time Series Analysis", fontsize=16)

    axs_ts[0].plot(sim_hours, flow_veh_per_hour, color='blue', linewidth=2)
    axs_ts[0].set_ylabel("Flow (veh/hr)")
    axs_ts[0].set_title("Flow over Time")
    axs_ts[0].grid(True, linestyle=':', alpha=0.6)
    axs_ts[0].set_ylim(bottom=0)

    axs_ts[1].plot(sim_hours, density_veh_per_km, color='orange', linewidth=2)
    axs_ts[1].set_ylabel("Density (veh/km)")
    axs_ts[1].set_title("Density over Time")
    axs_ts[1].grid(True, linestyle=':', alpha=0.6)
    axs_ts[1].axhline(y=estimated_critical_density, color='red', linestyle='--', label='Est. Critical Density')
    axs_ts[1].legend()
    axs_ts[1].set_ylim(bottom=0)

    axs_ts[2].plot(sim_hours, speed_kmh, color='green', linewidth=2)
    axs_ts[2].set_ylabel("Speed (km/h)")
    axs_ts[2].set_title("Speed over Time")
    axs_ts[2].set_xlabel("Simulation Hour (h)")
    axs_ts[2].grid(True, linestyle=':', alpha=0.6)
    if not df["speed_kmh"].empty:
        axs_ts[2].axhline(y=np.mean(speed_kmh), color='gray', linestyle='--', label="Avg Speed")
    axs_ts[2].legend()
    axs_ts[2].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_time_series.png")
    plt.show()

    # --- Fundamental Diagrams ---
    fig_fd, axs_fd = plt.subplots(1, 3, figsize=(18, 6))
    fig_fd.suptitle(f"Sensor {sensor_id} - Fundamental Diagrams", fontsize=16)

    # 1. Flow-Density Diagram
    axs_fd[0].scatter(density_veh_per_km, flow_veh_per_hour, alpha=0.7, color='purple', s=40, label='Aggregated Points')
    # Add a line connecting the points to show the evolution over time
    axs_fd[0].plot(density_veh_per_km, flow_veh_per_hour, color='purple', linestyle='--', linewidth=1, alpha=0.5, label='Path Over Time')
    axs_fd[0].set_title("Flow-Density Diagram")
    axs_fd[0].set_xlabel("Density (veh/km)")
    axs_fd[0].set_ylabel("Flow (veh/hr)")
    axs_fd[0].grid(True, linestyle=':', alpha=0.6)
    axs_fd[0].axvline(x=estimated_critical_density, color='red', linestyle='--', label='Est. Critical Density')
    axs_fd[0].legend()
    axs_fd[0].set_xlim(left=0)
    axs_fd[0].set_ylim(bottom=0)

    # 2. Speed-Density Diagram
    axs_fd[1].scatter(density_veh_per_km, speed_kmh, alpha=0.7, color='darkgreen', s=40, label='Aggregated Points')
    # Add a line connecting the points to show the evolution over time
    axs_fd[1].plot(density_veh_per_km, speed_kmh, color='darkgreen', linestyle='--', linewidth=1, alpha=0.5, label='Path Over Time')
    axs_fd[1].set_title("Speed-Density Diagram")
    axs_fd[1].set_xlabel("Density (veh/km)")
    axs_fd[1].set_ylabel("Speed (km/h)")
    axs_fd[1].grid(True, linestyle=':', alpha=0.6)
    axs_fd[1].axvline(x=estimated_critical_density, color='red', linestyle='--', label='Est. Critical Density')
    axs_fd[1].legend()
    axs_fd[1].set_xlim(left=0)
    axs_fd[1].set_ylim(bottom=0)

    # 3. Flow-Speed Diagram
    axs_fd[2].scatter(speed_kmh, flow_veh_per_hour, alpha=0.7, color='darkblue', s=40, label='Aggregated Points')
    # Add a line connecting the points to show the evolution over time
    axs_fd[2].plot(speed_kmh, flow_veh_per_hour, color='darkblue', linestyle='--', linewidth=1, alpha=0.5, label='Path Over Time')
    axs_fd[2].set_title("Flow-Speed Diagram")
    axs_fd[2].set_xlabel("Speed (km/h)")
    axs_fd[2].set_ylabel("Flow (veh/hr)")
    axs_fd[2].grid(True, linestyle=':', alpha=0.6)
    if not df["flow_veh_per_hour"].empty:
        estimated_max_flow = df["flow_veh_per_hour"].max() * 0.9
        if estimated_max_flow == 0 and not df["flow_veh_per_hour"].empty: estimated_max_flow = df["flow_veh_per_hour"].max()
        axs_fd[2].axhline(y=estimated_max_flow, color='red', linestyle='--', label='Est. Max Flow')
    axs_fd[2].legend()
    axs_fd[2].set_xlim(left=0)
    axs_fd[2].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_fundamental_diagrams.png")
    plt.show()

    # --- 3D Surface Plots (Attempting with griddata for surface look) ---
    # To get a "surface" look with colors, we need to interpolate.
    # This might still be jagged if data is too sparse or patterns are complex.
    # It attempts to create a function Speed = f(Time, Density) or Speed = f(Time, Flow)
    if len(df) >= 4: # Griddata 'linear' method needs at least 4 points to form a surface.
        fig_3d = plt.figure(figsize=(16, 7))

        hours_min, hours_max = sim_hours.min(), sim_hours.max()
        dens_min, dens_max = density_veh_per_km.min(), density_veh_per_km.max()
        flows_min, flows_max = flow_veh_per_hour.min(), flow_veh_per_hour.max()
        speeds_min, speeds_max = speed_kmh.min(), speed_kmh.max()

        # Create a finer grid for interpolation
        num_grid_points = 50 # Increased for smoother surface if data allows
        hours_interp = np.linspace(hours_min, hours_max + 1e-9, num_grid_points) if hours_max > hours_min else np.array([hours_min])
        densities_interp = np.linspace(dens_min, dens_max + 1e-9, num_grid_points) if dens_max > dens_min else np.array([dens_min])
        flows_interp = np.linspace(flows_min, flows_max + 1e-9, num_grid_points) if flows_max > flows_min else np.array([flows_min])

        grid_x_dens, grid_y_dens = np.meshgrid(hours_interp, densities_interp)
        grid_x_flow, grid_y_flow = np.meshgrid(hours_interp, flows_interp)

        # Interpolate speed onto the 2D grid
        # Add a small random jitter to points if all points are collinear, to prevent griddata errors
        points_dens = np.vstack([sim_hours, density_veh_per_km]).T
        values_dens = speed_kmh
        if len(points_dens) > 0 and np.std(points_dens[:,0]) < 1e-9 and np.std(points_dens[:,1]) < 1e-9: # Check for near-collinear points
            points_dens = points_dens + np.random.rand(*points_dens.shape) * 1e-6

        speed_grid_dens = griddata(points_dens, values_dens, (grid_x_dens, grid_y_dens), method='linear', fill_value=0)

        points_flow = np.vstack([sim_hours, flow_veh_per_hour]).T
        values_flow = speed_kmh
        if len(points_flow) > 0 and np.std(points_flow[:,0]) < 1e-9 and np.std(points_flow[:,1]) < 1e-9:
            points_flow = points_flow + np.random.rand(*points_flow.shape) * 1e-6

        speed_grid_flow = griddata(points_flow, values_flow, (grid_x_flow, grid_y_flow), method='linear', fill_value=0)

        # Density vs Speed vs Time Plot (Surface)
        ax1 = fig_3d.add_subplot(121, projection='3d')
        surf1 = ax1.plot_surface(grid_x_dens, grid_y_dens, speed_grid_dens, cmap='viridis', edgecolor='none', alpha=0.8)
        fig_3d.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
        ax1.set_title(f"Sensor {sensor_id} - Speed (Color) vs. Density over Time")
        ax1.set_xlabel("Simulation Hour (h)")
        ax1.set_ylabel("Density (veh/km)")
        ax1.set_zlabel("Speed (km/h)")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.view_init(elev=30, azim=-45)
        ax1.set_zlim(speeds_min * 0.9, speeds_max * 1.1) if speeds_max > speeds_min else ax1.set_zlim(0, 100)

        # Flow vs Speed vs Time Plot (Surface)
        ax2 = fig_3d.add_subplot(122, projection='3d')
        surf2 = ax2.plot_surface(grid_x_flow, grid_y_flow, speed_grid_flow, cmap='inferno', edgecolor='none', alpha=0.8)
        fig_3d.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
        ax2.set_title(f"Sensor {sensor_id} - Speed (Color) vs. Flow over Time")
        ax2.set_xlabel("Simulation Hour (h)")
        ax2.set_ylabel("Flow (veh/hr)")
        ax2.set_zlabel("Speed (km/h)")
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.view_init(elev=30, azim=-45)
        ax2.set_zlim(speeds_min * 0.9, speeds_max * 1.1) if speeds_max > speeds_min else ax2.set_zlim(0, 100)

        plt.tight_layout()
        plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_3d_surfaces.png")
        plt.show()
    else:
        print(f"⚠️ Not enough aggregated data ({len(df)} points) to plot 3D surface for sensor {sensor_id}. Minimum 4 frames required for meaningful interpolation.")


def plot_global_analysis():
    """
    Performs and plots analysis across all sensors combined.
    This function will now work with the aggregated data.
    """
    if not all_sensor_data or all(df.empty for df in all_sensor_data):
        print("No sensor data recorded for global analysis.")
        return

    non_empty_dfs = [df for df in all_sensor_data if not df.empty]
    if not non_empty_dfs:
        print("No non-empty sensor dataframes for global analysis.")
        return
    combined_df = pd.concat(non_empty_dfs, ignore_index=True)

    print("\n✅ Performing global analysis on all sensor data (aggregated)...")

    ensure_export_dir()

    # --- Combined Time-Series Plots ---
    fig_combined_ts, axs_combined_ts = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig_combined_ts.suptitle("Combined Time Series Analysis (All Sensors)", fontsize=16)

    for sensor_id in combined_df["sensor_id"].unique():
        sensor_df = combined_df[combined_df["sensor_id"] == sensor_id]
        axs_combined_ts[0].plot(sensor_df["sim_hours"], sensor_df["flow_veh_per_hour"], label=f'S{sensor_id}', alpha=0.7, linewidth=1.5)
        axs_combined_ts[1].plot(sensor_df["sim_hours"], sensor_df["density_veh_per_km"], label=f'S{sensor_id}', alpha=0.7, linewidth=1.5)
        axs_combined_ts[2].plot(sensor_df["sim_hours"], sensor_df["speed_kmh"], label=f'S{sensor_id}', alpha=0.7, linewidth=1.5)

    axs_combined_ts[0].set_ylabel("Flow (veh/hr)")
    axs_combined_ts[0].set_title("Flow over Time")
    axs_combined_ts[0].grid(True, linestyle=':', alpha=0.6)
    axs_combined_ts[0].legend()
    axs_combined_ts[0].set_ylim(bottom=0)

    axs_combined_ts[1].set_ylabel("Density (veh/km)")
    axs_combined_ts[1].set_title("Density over Time")
    axs_combined_ts[1].grid(True, linestyle=':', alpha=0.6)
    axs_combined_ts[1].legend()
    axs_combined_ts[1].set_ylim(bottom=0)

    axs_combined_ts[2].set_ylabel("Speed (km/h)")
    axs_combined_ts[2].set_title("Speed over Time")
    axs_combined_ts[2].set_xlabel("Simulation Hour (h)")
    axs_combined_ts[2].grid(True, linestyle=':', alpha=0.6)
    axs_combined_ts[2].legend()
    axs_combined_ts[2].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{EXPORT_FOLDER}/combined_time_series.png")
    plt.show()

    # --- Combined Fundamental Diagrams ---
    fig_combined_fd, axs_combined_fd = plt.subplots(1, 3, figsize=(18, 6))
    fig_combined_fd.suptitle("Combined Fundamental Diagrams (All Sensors)", fontsize=16)

    # Combined Flow-Density
    axs_combined_fd[0].scatter(combined_df["density_veh_per_km"], combined_df["flow_veh_per_hour"],
                               c=combined_df["sensor_id"], cmap='tab10', alpha=0.6, s=50) # Color by sensor_id
    axs_combined_fd[0].set_title("Combined Flow-Density Diagram")
    axs_combined_fd[0].set_xlabel("Density (veh/km)")
    axs_combined_fd[0].set_ylabel("Flow (veh/hr)")
    axs_combined_fd[0].grid(True, linestyle=':', alpha=0.6)
    axs_combined_fd[0].set_xlim(left=0)
    axs_combined_fd[0].set_ylim(bottom=0)

    # Combined Speed-Density
    axs_combined_fd[1].scatter(combined_df["density_veh_per_km"], combined_df["speed_kmh"],
                               c=combined_df["sensor_id"], cmap='tab10', alpha=0.6, s=50)
    axs_combined_fd[1].set_title("Combined Speed-Density Diagram")
    axs_combined_fd[1].set_xlabel("Density (veh/km)")
    axs_combined_fd[1].set_ylabel("Speed (km/h)")
    axs_combined_fd[1].grid(True, linestyle=':', alpha=0.6)
    axs_combined_fd[1].set_xlim(left=0)
    axs_combined_fd[1].set_ylim(bottom=0)

    # Combined Flow-Speed
    axs_combined_fd[2].scatter(combined_df["speed_kmh"], combined_df["flow_veh_per_hour"],
                               c=combined_df["sensor_id"], cmap='tab10', alpha=0.6, s=50)
    axs_combined_fd[2].set_title("Combined Flow-Speed Diagram")
    axs_combined_fd[2].set_xlabel("Speed (km/h)")
    axs_combined_fd[2].set_ylabel("Flow (veh/hr)")
    axs_combined_fd[2].grid(True, linestyle=':', alpha=0.6)
    axs_combined_fd[2].set_xlim(left=0)
    axs_combined_fd[2].set_ylim(bottom=0)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{EXPORT_FOLDER}/combined_fundamental_diagrams.png")
    plt.show()