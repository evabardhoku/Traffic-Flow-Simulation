# metanet_analysis.py - Improved Version

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.optimize import curve_fit

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
    GLOBAL_SECONDS_PER_FRAME = 1.0  # Default to 1.0s per frame if import fails
    GLOBAL_CELL_LENGTH_METERS = 416.66  # Default
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


def greenshields_model(density, v_f, rho_jam):
    """Greenshields fundamental diagram model: v = v_f * (1 - rho/rho_jam)"""
    return v_f * (1 - density / rho_jam)


def triangular_model(density, v_f, rho_c, q_max):
    """Triangular fundamental diagram model"""
    speed = np.where(density <= rho_c,
                     v_f,
                     q_max / density)
    return np.maximum(speed, 0)


def handle_sensor_data(sensor_id, raw_df, seconds_per_frame, cell_length_meters, road_length_km):
    """
    Processes raw sensor data by aggregating it into time intervals for smoother analysis.
    This function will now perform the averaging with improved METANET-specific calculations.
    """
    global GLOBAL_SECONDS_PER_FRAME, GLOBAL_CELL_LENGTH_METERS, GLOBAL_ROAD_LENGTH_KM
    GLOBAL_SECONDS_PER_FRAME = seconds_per_frame
    GLOBAL_CELL_LENGTH_METERS = cell_length_meters
    GLOBAL_ROAD_LENGTH_KM = road_length_km

    ensure_export_dir()

    print(f"Processing raw data for Sensor {sensor_id}...")

    if raw_df.empty:
        print(f"No raw data recorded for Sensor {sensor_id}. Skipping processing.")
        all_sensor_data.append(pd.DataFrame())  # Add empty df to maintain list structure
        return

    # Convert raw frame-level data to standard units
    df = raw_df.copy()  # Work on a copy
    df["sim_seconds"] = df["frame"] * GLOBAL_SECONDS_PER_FRAME
    df["speed_mps"] = df["speed"]  # speed from sensor is already in m/s
    df["density_vpm"] = df["density"]  # density from sensor is already in veh/m

    # --- METANET-specific calculations ---
    # Remove invalid data points (negative values, NaN, etc.)
    df = df[(df["speed_mps"] >= 0) & (df["density_vpm"] >= 0) &
            (df["speed_mps"] <= 200) & (df["density_vpm"] <= 1.0)]  # Reasonable limits

    if df.empty:
        print(f"No valid data after filtering for Sensor {sensor_id}. Skipping processing.")
        all_sensor_data.append(pd.DataFrame())
        return

    # --- Define Aggregation Interval ---
    # For METANET analysis, shorter intervals are often better to capture dynamics
    AGGREGATION_INTERVAL_SECONDS = 30  # 30 seconds for better resolution

    # Create time bins
    df['time_bin_start_seconds'] = (df['sim_seconds'] // AGGREGATION_INTERVAL_SECONDS) * AGGREGATION_INTERVAL_SECONDS

    # --- Group and Aggregate Data with better statistics ---
    aggregated_df = df.groupby('time_bin_start_seconds').agg(
        density_veh_per_meter=('density_vpm', 'mean'),
        density_std=('density_vpm', 'std'),
        avg_speed_m_per_s=('speed_mps', 'mean'),
        speed_std=('speed_mps', 'std'),
        total_flow_count_in_bin=('flow_out', 'sum'),
        total_ramp_inflow_in_bin=('ramp_inflow', 'sum'),
        data_points_in_bin=('density_vpm', 'count')
    ).reset_index()

    # Filter out bins with too few data points
    aggregated_df = aggregated_df[aggregated_df['data_points_in_bin'] >= 1]

    if aggregated_df.empty:
        print(f"No aggregated data after filtering for Sensor {sensor_id}. Skipping processing.")
        all_sensor_data.append(pd.DataFrame())
        return

    # --- Convert Aggregated Data to Desired Units ---
    aggregated_df["sim_hours"] = aggregated_df["time_bin_start_seconds"] / 3600.0
    aggregated_df["speed_kmh"] = aggregated_df["avg_speed_m_per_s"] * 3.6  # m/s to km/h
    aggregated_df["density_veh_per_km"] = aggregated_df["density_veh_per_meter"] * 1000  # veh/m to veh/km

    # Calculate Flow (veh/hour) - METANET uses q = k * v relationship
    aggregated_df["flow_veh_per_hour"] = aggregated_df["density_veh_per_km"] * aggregated_df["speed_kmh"]

    # Also calculate flow from actual count data
    aggregated_df["measured_flow_veh_per_hour"] = (aggregated_df[
                                                       "total_flow_count_in_bin"] / AGGREGATION_INTERVAL_SECONDS) * 3600
    aggregated_df["ramp_inflow_veh_per_hour"] = (aggregated_df[
                                                     "total_ramp_inflow_in_bin"] / AGGREGATION_INTERVAL_SECONDS) * 3600

    # Remove invalid calculated values
    aggregated_df = aggregated_df.fillna(0.0)
    aggregated_df = aggregated_df[(aggregated_df["speed_kmh"] >= 0) &
                                  (aggregated_df["density_veh_per_km"] >= 0) &
                                  (aggregated_df["flow_veh_per_hour"] >= 0)]

    if aggregated_df.empty:
        print(f"No valid aggregated data for Sensor {sensor_id}. Skipping processing.")
        all_sensor_data.append(pd.DataFrame())
        return

    filename = f"{EXPORT_FOLDER}/sensor_{sensor_id}_aggregated_data.csv"
    aggregated_df.to_csv(filename, index=False)
    print(f"✅ Sensor {sensor_id} aggregated data exported to {filename} ({len(aggregated_df)} aggregated points)")

    aggregated_df["sensor_id"] = sensor_id
    all_sensor_data.append(aggregated_df)

    plot_sensor_analysis(sensor_id, aggregated_df)


def plot_sensor_analysis(sensor_id, df):
    """
    Generates improved plots for METANET analysis with better formatting and theoretical curves.
    """
    sim_hours = df["sim_hours"].values
    speed_kmh = df["speed_kmh"].values
    density_veh_per_km = df["density_veh_per_km"].values
    flow_veh_per_hour = df["flow_veh_per_hour"].values

    if len(df) < 2:
        print(f"⚠️ Not enough aggregated data ({len(df)} points) for plots. Skipping for sensor {sensor_id}.")
        return

    # Estimate fundamental diagram parameters
    max_density = np.max(density_veh_per_km) if len(density_veh_per_km) > 0 else 150
    max_speed = np.max(speed_kmh) if len(speed_kmh) > 0 else 120
    max_flow = np.max(flow_veh_per_hour) if len(flow_veh_per_hour) > 0 else 2000

    # Estimate critical density (where flow is maximum)
    if len(flow_veh_per_hour) > 0 and max_flow > 0:
        max_flow_idx = np.argmax(flow_veh_per_hour)
        critical_density = density_veh_per_km[max_flow_idx] if max_flow_idx < len(
            density_veh_per_km) else max_density * 0.3
    else:
        critical_density = max_density * 0.3

    # --- Time-Series Plots with improved styling ---
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig_ts, axs_ts = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig_ts.suptitle(f"METANET Sensor {sensor_id} - Time Series Analysis", fontsize=14, fontweight='bold')

    # Flow over time
    axs_ts[0].plot(sim_hours, flow_veh_per_hour, 'b-', linewidth=2, label='Calculated Flow')
    if 'measured_flow_veh_per_hour' in df.columns:
        axs_ts[0].plot(sim_hours, df['measured_flow_veh_per_hour'], 'r--', linewidth=1.5, alpha=0.7,
                       label='Measured Flow')
    axs_ts[0].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_ts[0].set_title("Traffic Flow Evolution", fontsize=12)
    axs_ts[0].grid(True, alpha=0.3)
    axs_ts[0].legend()
    axs_ts[0].set_ylim(bottom=0)

    # Density over time
    axs_ts[1].plot(sim_hours, density_veh_per_km, 'orange', linewidth=2)
    axs_ts[1].axhline(y=critical_density, color='red', linestyle='--', alpha=0.8, label='Critical Density')
    axs_ts[1].set_ylabel("Density (veh/km)", fontsize=12)
    axs_ts[1].set_title("Traffic Density Evolution", fontsize=12)
    axs_ts[1].grid(True, alpha=0.3)
    axs_ts[1].legend()
    axs_ts[1].set_ylim(bottom=0)

    # Speed over time
    axs_ts[2].plot(sim_hours, speed_kmh, 'green', linewidth=2)
    axs_ts[2].axhline(y=np.mean(speed_kmh), color='gray', linestyle='--', alpha=0.8,
                      label=f"Mean Speed ({np.mean(speed_kmh):.1f} km/h)")
    axs_ts[2].set_ylabel("Speed (km/h)", fontsize=12)
    axs_ts[2].set_title("Traffic Speed Evolution", fontsize=12)
    axs_ts[2].set_xlabel("Simulation Time (hours)", fontsize=12)
    axs_ts[2].grid(True, alpha=0.3)
    axs_ts[2].legend()
    axs_ts[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_time_series.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Fundamental Diagrams with theoretical curves ---
    fig_fd, axs_fd = plt.subplots(1, 3, figsize=(18, 6))
    fig_fd.suptitle(f"METANET Sensor {sensor_id} - Fundamental Diagrams", fontsize=14, fontweight='bold')

    # 1. Flow-Density Diagram
    axs_fd[0].scatter(density_veh_per_km, flow_veh_per_hour, alpha=0.7, color='purple', s=50,
                      edgecolors='black', linewidths=0.5, label='Observed Data')

    # Fit theoretical curves if we have enough data
    if len(density_veh_per_km) > 5:
        density_range = np.linspace(0, max(max_density * 1.2, 150), 100)
        try:
            # Try to fit Greenshields model for flow-density
            jam_density = max_density * 1.5 if max_density > 0 else 150
            free_flow_speed = max_speed if max_speed > 0 else 120
            theoretical_flow = density_range * greenshields_model(density_range, free_flow_speed / 3.6,
                                                                  jam_density / 1000) * 3.6
            axs_fd[0].plot(density_range, theoretical_flow, 'r-', linewidth=2, alpha=0.8,
                           label='Theoretical (Greenshields)')
        except:
            pass

    axs_fd[0].axvline(x=critical_density, color='red', linestyle='--', alpha=0.8, label='Critical Density')
    axs_fd[0].set_title("Flow-Density Relationship", fontsize=12)
    axs_fd[0].set_xlabel("Density (veh/km)", fontsize=12)
    axs_fd[0].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_fd[0].grid(True, alpha=0.3)
    axs_fd[0].legend()
    axs_fd[0].set_xlim(left=0)
    axs_fd[0].set_ylim(bottom=0)

    # 2. Speed-Density Diagram
    axs_fd[1].scatter(density_veh_per_km, speed_kmh, alpha=0.7, color='darkgreen', s=50,
                      edgecolors='black', linewidths=0.5, label='Observed Data')

    # Theoretical speed-density curve
    if len(density_veh_per_km) > 5:
        try:
            jam_density = max_density * 1.5 if max_density > 0 else 150
            free_flow_speed = max_speed if max_speed > 0 else 120
            density_range = np.linspace(0, jam_density, 100)
            theoretical_speed = greenshields_model(density_range / 1000, free_flow_speed / 3.6,
                                                   jam_density / 1000) * 3.6
            axs_fd[1].plot(density_range, theoretical_speed, 'r-', linewidth=2, alpha=0.8,
                           label='Theoretical (Greenshields)')
        except:
            pass

    axs_fd[1].axvline(x=critical_density, color='red', linestyle='--', alpha=0.8, label='Critical Density')
    axs_fd[1].set_title("Speed-Density Relationship", fontsize=12)
    axs_fd[1].set_xlabel("Density (veh/km)", fontsize=12)
    axs_fd[1].set_ylabel("Speed (km/h)", fontsize=12)
    axs_fd[1].grid(True, alpha=0.3)
    axs_fd[1].legend()
    axs_fd[1].set_xlim(left=0)
    axs_fd[1].set_ylim(bottom=0)

    # 3. Flow-Speed Diagram
    axs_fd[2].scatter(speed_kmh, flow_veh_per_hour, alpha=0.7, color='darkblue', s=50,
                      edgecolors='black', linewidths=0.5, label='Observed Data')

    if max_flow > 0:
        axs_fd[2].axhline(y=max_flow, color='red', linestyle='--', alpha=0.8, label=f'Max Flow ({max_flow:.0f} veh/h)')

    axs_fd[2].set_title("Flow-Speed Relationship", fontsize=12)
    axs_fd[2].set_xlabel("Speed (km/h)", fontsize=12)
    axs_fd[2].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_fd[2].grid(True, alpha=0.3)
    axs_fd[2].legend()
    axs_fd[2].set_xlim(left=0)
    axs_fd[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_fundamental_diagrams.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- 3D Surface Plots (Improved) ---
    if len(df) >= 10:  # Need more points for meaningful surface
        plot_3d_surfaces(sensor_id, df)
    else:
        print(f"⚠️ Not enough data points ({len(df)}) for 3D surface plots. Need at least 10 points.")


def plot_3d_surfaces(sensor_id, df):
    """Create improved 3D surface plots for METANET analysis"""
    sim_hours = df["sim_hours"].values
    speed_kmh = df["speed_kmh"].values
    density_veh_per_km = df["density_veh_per_km"].values
    flow_veh_per_hour = df["flow_veh_per_hour"].values

    fig = plt.figure(figsize=(16, 8))

    # Create grids for interpolation
    hours_range = np.linspace(sim_hours.min(), sim_hours.max(), 30)
    density_range = np.linspace(density_veh_per_km.min(), density_veh_per_km.max(), 30)
    flow_range = np.linspace(flow_veh_per_hour.min(), flow_veh_per_hour.max(), 30)

    # 1. Time-Density-Speed surface
    ax1 = fig.add_subplot(121, projection='3d')

    # Create meshgrid
    H, D = np.meshgrid(hours_range, density_range)

    # Interpolate speed values
    try:
        points = np.column_stack([sim_hours, density_veh_per_km])
        S = griddata(points, speed_kmh, (H, D), method='cubic', fill_value=0)
        S = np.maximum(S, 0)  # Ensure non-negative speeds

        surf1 = ax1.plot_surface(H, D, S, cmap='viridis', alpha=0.8, edgecolor='none')
        ax1.scatter(sim_hours, density_veh_per_km, speed_kmh, c='red', s=20, alpha=0.8)

        ax1.set_xlabel('Time (hours)', fontsize=10)
        ax1.set_ylabel('Density (veh/km)', fontsize=10)
        ax1.set_zlabel('Speed (km/h)', fontsize=10)
        ax1.set_title(f'Sensor {sensor_id}: Speed Surface\n(Time vs Density)', fontsize=12)

        # Add colorbar
        plt.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20, label='Speed (km/h)')

    except Exception as e:
        print(f"Could not create first 3D surface: {e}")
        ax1.text(0.5, 0.5, 0.5, "Insufficient data\nfor surface plot",
                 transform=ax1.transAxes, ha='center', va='center')

    # 2. Time-Flow-Speed surface
    ax2 = fig.add_subplot(122, projection='3d')

    try:
        H2, F = np.meshgrid(hours_range, flow_range)
        points2 = np.column_stack([sim_hours, flow_veh_per_hour])
        S2 = griddata(points2, speed_kmh, (H2, F), method='cubic', fill_value=0)
        S2 = np.maximum(S2, 0)

        surf2 = ax2.plot_surface(H2, F, S2, cmap='plasma', alpha=0.8, edgecolor='none')
        ax2.scatter(sim_hours, flow_veh_per_hour, speed_kmh, c='red', s=20, alpha=0.8)

        ax2.set_xlabel('Time (hours)', fontsize=10)
        ax2.set_ylabel('Flow (veh/h)', fontsize=10)
        ax2.set_zlabel('Speed (km/h)', fontsize=10)
        ax2.set_title(f'Sensor {sensor_id}: Speed Surface\n(Time vs Flow)', fontsize=12)

        plt.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20, label='Speed (km/h)')

    except Exception as e:
        print(f"Could not create second 3D surface: {e}")
        ax2.text(0.5, 0.5, 0.5, "Insufficient data\nfor surface plot",
                 transform=ax2.transAxes, ha='center', va='center')

    plt.tight_layout()
    plt.savefig(f"{EXPORT_FOLDER}/sensor_{sensor_id}_3d_surfaces.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_global_analysis():
    """
    Performs improved global analysis across all sensors with METANET-specific features.
    """
    if not all_sensor_data or all(df.empty for df in all_sensor_data):
        print("No sensor data recorded for global analysis.")
        return

    non_empty_dfs = [df for df in all_sensor_data if not df.empty]
    if not non_empty_dfs:
        print("No non-empty sensor dataframes for global analysis.")
        return

    combined_df = pd.concat(non_empty_dfs, ignore_index=True)
    print(
        f"\n✅ Performing global METANET analysis on {len(combined_df)} data points from {len(non_empty_dfs)} sensors...")

    ensure_export_dir()

    # Export combined data
    combined_df.to_csv(f"{EXPORT_FOLDER}/combined_sensor_data.csv", index=False)

    # --- Enhanced Combined Time-Series Plots ---
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    fig_combined_ts, axs_combined_ts = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig_combined_ts.suptitle("METANET Global Analysis - All Sensors Combined", fontsize=16, fontweight='bold')

    colors = plt.cm.Set1(np.linspace(0, 1, len(combined_df["sensor_id"].unique())))

    for i, sensor_id in enumerate(sorted(combined_df["sensor_id"].unique())):
        sensor_df = combined_df[combined_df["sensor_id"] == sensor_id]
        color = colors[i % len(colors)]

        axs_combined_ts[0].plot(sensor_df["sim_hours"], sensor_df["flow_veh_per_hour"],
                                label=f'Sensor {sensor_id}', color=color, linewidth=2, alpha=0.8)
        axs_combined_ts[1].plot(sensor_df["sim_hours"], sensor_df["density_veh_per_km"],
                                label=f'Sensor {sensor_id}', color=color, linewidth=2, alpha=0.8)
        axs_combined_ts[2].plot(sensor_df["sim_hours"], sensor_df["speed_kmh"],
                                label=f'Sensor {sensor_id}', color=color, linewidth=2, alpha=0.8)

    axs_combined_ts[0].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_combined_ts[0].set_title("Traffic Flow Evolution - All Sensors", fontsize=12)
    axs_combined_ts[0].grid(True, alpha=0.3)
    axs_combined_ts[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_ts[0].set_ylim(bottom=0)

    axs_combined_ts[1].set_ylabel("Density (veh/km)", fontsize=12)
    axs_combined_ts[1].set_title("Traffic Density Evolution - All Sensors", fontsize=12)
    axs_combined_ts[1].grid(True, alpha=0.3)
    axs_combined_ts[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_ts[1].set_ylim(bottom=0)

    axs_combined_ts[2].set_ylabel("Speed (km/h)", fontsize=12)
    axs_combined_ts[2].set_title("Traffic Speed Evolution - All Sensors", fontsize=12)
    axs_combined_ts[2].set_xlabel("Simulation Time (hours)", fontsize=12)
    axs_combined_ts[2].grid(True, alpha=0.3)
    axs_combined_ts[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_ts[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{EXPORT_FOLDER}/combined_time_series.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Enhanced Combined Fundamental Diagrams ---
    fig_combined_fd, axs_combined_fd = plt.subplots(1, 3, figsize=(18, 6))
    fig_combined_fd.suptitle("METANET Global Fundamental Diagrams", fontsize=16, fontweight='bold')

    # Create a colormap for sensors
    unique_sensors = sorted(combined_df["sensor_id"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_sensors)))

    # Combined Flow-Density with better visualization
    for i, sensor_id in enumerate(unique_sensors):
        sensor_data = combined_df[combined_df["sensor_id"] == sensor_id]
        axs_combined_fd[0].scatter(sensor_data["density_veh_per_km"], sensor_data["flow_veh_per_hour"],
                                   color=colors[i], alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
                                   label=f'Sensor {sensor_id}')

    axs_combined_fd[0].set_title("Combined Flow-Density Diagram", fontsize=12)
    axs_combined_fd[0].set_xlabel("Density (veh/km)", fontsize=12)
    axs_combined_fd[0].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_combined_fd[0].grid(True, alpha=0.3)
    axs_combined_fd[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_fd[0].set_xlim(left=0)
    axs_combined_fd[0].set_ylim(bottom=0)

    # Combined Speed-Density
    for i, sensor_id in enumerate(unique_sensors):
        sensor_data = combined_df[combined_df["sensor_id"] == sensor_id]
        axs_combined_fd[1].scatter(sensor_data["density_veh_per_km"], sensor_data["speed_kmh"],
                                   color=colors[i], alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
                                   label=f'Sensor {sensor_id}')

    axs_combined_fd[1].set_title("Combined Speed-Density Diagram", fontsize=12)
    axs_combined_fd[1].set_xlabel("Density (veh/km)", fontsize=12)
    axs_combined_fd[1].set_ylabel("Speed (km/h)", fontsize=12)
    axs_combined_fd[1].grid(True, alpha=0.3)
    axs_combined_fd[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_fd[1].set_xlim(left=0)
    axs_combined_fd[1].set_ylim(bottom=0)

    # Combined Flow-Speed
    for i, sensor_id in enumerate(unique_sensors):
        sensor_data = combined_df[combined_df["sensor_id"] == sensor_id]
        axs_combined_fd[2].scatter(sensor_data["speed_kmh"], sensor_data["flow_veh_per_hour"],
                                   color=colors[i], alpha=0.7, s=60, edgecolor='black', linewidth=0.5,
                                   label=f'Sensor {sensor_id}')

    axs_combined_fd[2].set_title("Combined Flow-Speed Diagram", fontsize=12)
    axs_combined_fd[2].set_xlabel("Speed (km/h)", fontsize=12)
    axs_combined_fd[2].set_ylabel("Flow (veh/h)", fontsize=12)
    axs_combined_fd[2].grid(True, alpha=0.3)
    axs_combined_fd[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs_combined_fd[2].set_xlim(left=0)
    axs_combined_fd[2].set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(f"{EXPORT_FOLDER}/combined_fundamental_diagrams.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- Statistical Summary ---
    print("\n" + "=" * 80)
    print("METANET GLOBAL ANALYSIS SUMMARY")
    print("=" * 80)

    for sensor_id in unique_sensors:
        sensor_data = combined_df[combined_df["sensor_id"] == sensor_id]
        print(f"\nSensor {sensor_id} Statistics:")
        print(f"  Data Points: {len(sensor_data)}")
        print(f"  Avg Flow: {sensor_data['flow_veh_per_hour'].mean():.1f} veh/h")
        print(f"  Max Flow: {sensor_data['flow_veh_per_hour'].max():.1f} veh/h")
        print(f"  Avg Density: {sensor_data['density_veh_per_km'].mean():.1f} veh/km")
        print(f"  Max Density: {sensor_data['density_veh_per_km'].max():.1f} veh/km")
        print(f"  Avg Speed: {sensor_data['speed_kmh'].mean():.1f} km/h")
        print(f"  Min Speed: {sensor_data['speed_kmh'].min():.1f} km/h")

    # Overall statistics
    print(f"\nOverall Network Statistics:")
    print(f"  Total Data Points: {len(combined_df)}")
    print(f"  Network Avg Flow: {combined_df['flow_veh_per_hour'].mean():.1f} veh/h")
    print(f"  Network Max Flow: {combined_df['flow_veh_per_hour'].max():.1f} veh/h")
    print(f"  Network Avg Density: {combined_df['density_veh_per_km'].mean():.1f} veh/km")
    print(f"  Network Max Density: {combined_df['density_veh_per_km'].max():.1f} veh/km")
    print(f"  Network Avg Speed: {combined_df['speed_kmh'].mean():.1f} km/h")
    print(f"  Network Min Speed: {combined_df['speed_kmh'].min():.1f} km/h")

    # Save summary statistics
    summary_stats = []
    for sensor_id in unique_sensors:
        sensor_data = combined_df[combined_df["sensor_id"] == sensor_id]
        summary_stats.append({
            'sensor_id': sensor_id,
            'data_points': len(sensor_data),
            'avg_flow_veh_per_hour': sensor_data['flow_veh_per_hour'].mean(),
            'max_flow_veh_per_hour': sensor_data['flow_veh_per_hour'].max(),
            'avg_density_veh_per_km': sensor_data['density_veh_per_km'].mean(),
            'max_density_veh_per_km': sensor_data['density_veh_per_km'].max(),
            'avg_speed_kmh': sensor_data['speed_kmh'].mean(),
            'min_speed_kmh': sensor_data['speed_kmh'].min()
        })

    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(f"{EXPORT_FOLDER}/sensor_summary_statistics.csv", index=False)

    print(f"\n✅ Global analysis complete. Results exported to {EXPORT_FOLDER}/")
    print("=" * 80)


def analyze_traffic_patterns():
    """
    Advanced traffic pattern analysis for METANET validation.
    """
    if not all_sensor_data or all(df.empty for df in all_sensor_data):
        print("No sensor data available for traffic pattern analysis.")
        return

    non_empty_dfs = [df for df in all_sensor_data if not df.empty]
    if not non_empty_dfs:
        print("No non-empty sensor dataframes for pattern analysis.")
        return

    combined_df = pd.concat(non_empty_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("TRAFFIC PATTERN ANALYSIS")
    print("=" * 80)

    # Identify congestion periods (low speed, high density)
    congestion_threshold_speed = combined_df['speed_kmh'].quantile(0.25)  # Bottom 25% of speeds
    congestion_threshold_density = combined_df['density_veh_per_km'].quantile(0.75)  # Top 25% of densities

    congested_data = combined_df[
        (combined_df['speed_kmh'] <= congestion_threshold_speed) &
        (combined_df['density_veh_per_km'] >= congestion_threshold_density)
        ]

    print(f"Congestion Analysis:")
    print(f"  Speed threshold: {congestion_threshold_speed:.1f} km/h")
    print(f"  Density threshold: {congestion_threshold_density:.1f} veh/km")
    print(
        f"  Congested periods: {len(congested_data)} data points ({len(congested_data) / len(combined_df) * 100:.1f}%)")

    if len(congested_data) > 0:
        print(f"  Avg congested speed: {congested_data['speed_kmh'].mean():.1f} km/h")
        print(f"  Avg congested density: {congested_data['density_veh_per_km'].mean():.1f} veh/km")
        print(f"  Avg congested flow: {congested_data['flow_veh_per_hour'].mean():.1f} veh/h")

    # Free flow analysis (high speed, low density)
    freeflow_threshold_speed = combined_df['speed_kmh'].quantile(0.75)  # Top 25% of speeds
    freeflow_threshold_density = combined_df['density_veh_per_km'].quantile(0.25)  # Bottom 25% of densities

    freeflow_data = combined_df[
        (combined_df['speed_kmh'] >= freeflow_threshold_speed) &
        (combined_df['density_veh_per_km'] <= freeflow_threshold_density)
        ]

    print(f"\nFree Flow Analysis:")
    print(f"  Speed threshold: {freeflow_threshold_speed:.1f} km/h")
    print(f"  Density threshold: {freeflow_threshold_density:.1f} veh/km")
    print(f"  Free flow periods: {len(freeflow_data)} data points ({len(freeflow_data) / len(combined_df) * 100:.1f}%)")

    if len(freeflow_data) > 0:
        print(f"  Avg free flow speed: {freeflow_data['speed_kmh'].mean():.1f} km/h")
        print(f"  Avg free flow density: {freeflow_data['density_veh_per_km'].mean():.1f} veh/km")
        print(f"  Avg free flow flow: {freeflow_data['flow_veh_per_hour'].mean():.1f} veh/h")


def export_metanet_parameters():
    """
    Export estimated METANET parameters based on observed data.
    """
    if not all_sensor_data or all(df.empty for df in all_sensor_data):
        print("No sensor data available for parameter estimation.")
        return

    non_empty_dfs = [df for df in all_sensor_data if not df.empty]
    if not non_empty_dfs:
        return

    combined_df = pd.concat(non_empty_dfs, ignore_index=True)

    print("\n" + "=" * 80)
    print("METANET PARAMETER ESTIMATION")
    print("=" * 80)

    # Estimate fundamental diagram parameters
    max_observed_speed = combined_df['speed_kmh'].max()
    max_observed_density = combined_df['density_veh_per_km'].max()
    max_observed_flow = combined_df['flow_veh_per_hour'].max()

    # Estimate free flow speed (95th percentile of speeds)
    free_flow_speed = combined_df['speed_kmh'].quantile(0.95)

    # Estimate jam density (extrapolate from max observed density)
    jam_density_estimate = max_observed_density * 1.3  # Conservative estimate

    # Estimate critical density (density at maximum flow)
    if len(combined_df) > 0:
        max_flow_idx = combined_df['flow_veh_per_hour'].idxmax()
        critical_density = combined_df.loc[max_flow_idx, 'density_veh_per_km']
    else:
        critical_density = max_observed_density * 0.3

    parameters = {
        'free_flow_speed_kmh': free_flow_speed,
        'jam_density_veh_per_km': jam_density_estimate,
        'critical_density_veh_per_km': critical_density,
        'max_observed_flow_veh_per_hour': max_observed_flow,
        'max_observed_speed_kmh': max_observed_speed,
        'max_observed_density_veh_per_km': max_observed_density,
        'cell_length_meters': GLOBAL_CELL_LENGTH_METERS,
        'road_length_km': GLOBAL_ROAD_LENGTH_KM,
        'simulation_timestep_seconds': GLOBAL_SECONDS_PER_FRAME
    }

    print("Estimated METANET Parameters:")
    for param, value in parameters.items():
        if 'speed' in param or 'flow' in param or 'density' in param:
            print(f"  {param}: {value:.2f}")
        else:
            print(f"  {param}: {value}")

    # Export parameters
    ensure_export_dir()
    param_df = pd.DataFrame([parameters])
    param_df.to_csv(f"{EXPORT_FOLDER}/estimated_metanet_parameters.csv", index=False)

    print(f"\n✅ Parameters exported to {EXPORT_FOLDER}/estimated_metanet_parameters.csv")
    print("=" * 80)


# Main execution function
def run_complete_analysis():
    """
    Run the complete METANET analysis workflow.
    """
    print("Starting complete METANET analysis...")
    plot_global_analysis()
    analyze_traffic_patterns()
    export_metanet_parameters()
    print("\n✅ Complete METANET analysis finished!")


if __name__ == "__main__":
    print("METANET Analysis Module Loaded")
    print("Use handle_sensor_data() to process individual sensor data")
    print("Use plot_global_analysis() to analyze all sensors combined")
    print("Use run_complete_analysis() for complete workflow")