import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import fsolve

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
file_path = r'线路条件数据.xlsx'
# Load the relevant sheets into dataframes
station_df = pd.read_excel(file_path, sheet_name='station', engine='openpyxl')
curve_df = pd.read_excel(file_path, sheet_name='curve', engine='openpyxl')
grad_df = pd.read_excel(file_path, sheet_name='grad', engine='openpyxl')
basicinfo_df = pd.read_excel(file_path, sheet_name='BasicInfo', engine='openpyxl')
rotational_mass_coefficient = 1.08
brake_delay = 0.7
cut_acceleration_delay = 1
brake_deceleration = 1.2
acc = 0.5
normal_deceleration = 0.8


def kmh_to_ms(speed_kmh):
    return speed_kmh / 3.6


def ms_to_kmh(speed_ms):
    return speed_ms * 3.6


def get_previous_station_distance(distance):
    previous_station_distances = station_df[station_df['限速终点（m）'] <= distance]['限速终点（m）']
    if not previous_station_distances.empty:
        return previous_station_distances.max()
    else:
        return 0


def get_previous_limit_change_distance(current_distance):
    # 初始化当前限速值
    current_speed_limit = speed_limits[int(current_distance) - 1]

    # 倒序遍历距离，寻找限速变化点
    for d in range(int(current_distance) - 1, -1, -1):
        if speed_limits[d] != current_speed_limit:
            return d + 1  # 返回变化点的下一个距离
    return 0  # 如果没有变化点，则返回0


def get_acceleration(distance):
    previous_distance = get_previous_station_distance(distance)
    # Find the grad row that corresponds to the given distance
    grad_row = grad_df[(grad_df['坡道始点（m）'] <= distance) & (grad_df['坡道终点（m）'] >= distance)]

    if grad_row.empty:
        raise ValueError(f"No gradient data found for distance {distance}")

    # Initialize the worst acceleration
    worst_acceleration = -np.inf

    for _, row in grad_row.iterrows():
        slope = row['坡度']
        up_down_flag = row['上下坡标志']

        # Calculate acceleration
        if up_down_flag == 1:  # Uphill
            acceleration = slope / rotational_mass_coefficient
        else:  # Downhill or flat
            acceleration = -slope / rotational_mass_coefficient

        # Check if this is the worst acceleration
        if acceleration > worst_acceleration:
            worst_acceleration = acceleration

    return worst_acceleration


# def calculate_deceleration_distance(current_speed, next_speed, current_distance):
#     # Calculate speed change during brake establishment delay
#     slope_acceleration = get_acceleration(current_distance)
#     speed_after_delay = current_speed + slope_acceleration * brake_delay
#
#     # Calculate the total deceleration including slope
#     total_deceleration = brake_deceleration + slope_acceleration
#
#     # Calculate the distance needed to decelerate to the next speed after the delay
#     delta_speed = next_speed - speed_after_delay
#     deceleration_distance = (speed_after_delay ** 2 - next_speed ** 2) / (2 * total_deceleration)
#
#     # Calculate total distance including the delay period
#     distance_during_delay = current_speed * brake_delay + 0.5 * slope_acceleration * brake_delay ** 2
#     total_distance = distance_during_delay + deceleration_distance
#
#     return total_distance
def calculate_deceleration_distance(current_speed_kmh, next_speed_kmh, current_distance):
    current_speed = kmh_to_ms(current_speed_kmh)
    next_speed = kmh_to_ms(next_speed_kmh)
    # Calculate speed change during cut acceleration delay
    # slope_acceleration = get_acceleration(current_distance)
    slope_acceleration = 0
    combined_acceleration = acc + slope_acceleration
    speed_after_cut_acceleration = current_speed + combined_acceleration * cut_acceleration_delay

    # 计算可用距离 TODO 这里有错
    max_allowed_distance = current_distance - get_previous_limit_change_distance(current_distance)

    # Calculate distance during cut acceleration delay
    distance_during_cut_acceleration = current_speed * cut_acceleration_delay + 0.5 * combined_acceleration * cut_acceleration_delay ** 2

    # Update current speed and distance after cut acceleration delay
    current_speed = speed_after_cut_acceleration
    # current_distance += distance_during_cut_acceleration

    # Calculate speed change during brake establishment delay
    speed_after_brake_delay = current_speed + slope_acceleration * brake_delay

    # Calculate the total deceleration including slope
    total_deceleration = brake_deceleration + slope_acceleration

    # Calculate the distance needed to decelerate to the next speed after the delay
    delta_speed = next_speed - speed_after_brake_delay
    deceleration_distance = (speed_after_brake_delay ** 2 - next_speed ** 2) / (2 * total_deceleration)

    # Calculate total distance including the delay period
    distance_during_brake_delay = current_speed * brake_delay + 0.5 * slope_acceleration * brake_delay ** 2
    total_distance = distance_during_cut_acceleration + distance_during_brake_delay + deceleration_distance
    if total_distance < max_allowed_distance:
        # Update speed_limits_minus_4
        start_cut_acceleration_distance = int(current_distance - total_distance)
        end_cut_acceleration_distance = int(current_distance - (total_distance - distance_during_cut_acceleration))
        end_brake_delay_distance = int(current_distance - deceleration_distance)
        final_distance = int(current_distance)

        # First segment: cut acceleration delay
        for d in range(start_cut_acceleration_distance, end_cut_acceleration_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                # speed_limits_minus_4[d] = current_speed - combined_acceleration * (d - start_cut_acceleration_distance)
                s = d - start_cut_acceleration_distance + 1
                speed_limits_minus_4[d] = ms_to_kmh(np.sqrt(current_speed ** 2 + 2 * combined_acceleration * s))

        # Second segment: brake establishment delay
        for d in range(end_cut_acceleration_distance, end_brake_delay_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                s = d - end_cut_acceleration_distance + 1
                # speed_limits_minus_4[d] = speed_after_cut_acceleration - slope_acceleration * (d - end_cut_acceleration_distance)
                speed_limits_minus_4[d] = ms_to_kmh(
                    np.sqrt(speed_after_cut_acceleration ** 2 + 2 * slope_acceleration * s))
        # Third segment: deceleration
        for d in range(end_brake_delay_distance, final_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                # speed_limits_minus_4[d] = speed_after_brake_delay + total_deceleration * (d - end_brake_delay_distance)
                s = d - end_brake_delay_distance + 1
                speed_limits_minus_4[d] = ms_to_kmh(np.sqrt(speed_after_brake_delay ** 2 - 2 * total_deceleration * s))
        return total_distance
    else:
        def equation(v):
            return ((v + 0.5) ** 2 - v ** 2) + (v + 0.5) * 0.7 + (
                    (v + 0.5) ** 2 - next_speed ** 2) / 2.4 - max_allowed_distance

        v_initial_guess = current_speed
        v_solution = fsolve(equation, v_initial_guess)[0]
        if v_solution < 0:
            raise ValueError("No valid solution found for initial speed.")
        # Update speed_limits_minus_4 based on the new initial speed v_solution
        remaining_distance = max_allowed_distance
        remaining_speed = v_solution
        start_distance = int(current_distance - remaining_distance)

        distance_during_cut_acceleration = v_solution * cut_acceleration_delay + 0.5 * acc * cut_acceleration_delay ** 2
        speed_after_cut_acceleration = v_solution + acc * cut_acceleration_delay
        distance_during_brake_delay = speed_after_cut_acceleration * brake_delay
        deceleration_distance = (speed_after_cut_acceleration ** 2 - next_speed ** 2) / (2 * total_deceleration)
        start_cut_acceleration_distance = int(current_distance - max_allowed_distance)
        end_cut_acceleration_distance = start_cut_acceleration_distance + int(distance_during_cut_acceleration)
        end_brake_delay_distance = end_cut_acceleration_distance + int(distance_during_brake_delay)
        final_distance = end_brake_delay_distance + int(deceleration_distance)

        # First segment: cut acceleration delay
        for d in range(start_cut_acceleration_distance, end_cut_acceleration_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                s = d - start_cut_acceleration_distance + 1
                speed_limits_minus_4[d] = ms_to_kmh(np.sqrt(v_solution ** 2 + 2 * acc * s))

        # Second segment: brake establishment delay
        for d in range(end_cut_acceleration_distance, end_brake_delay_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                speed_limits_minus_4[d] = ms_to_kmh(speed_after_cut_acceleration)

        # Third segment: deceleration
        for d in range(end_brake_delay_distance, final_distance):
            if d >= 0 and d < len(speed_limits_minus_4):
                s = d - end_brake_delay_distance + 1
                speed_limits_minus_4[d] = ms_to_kmh(
                    np.sqrt(speed_after_cut_acceleration ** 2 - 2 * total_deceleration * s))

        return max_allowed_distance


def calculate_deceleration_distance_SBI(current_speed_kmh, next_speed_kmh, current_distance):
    current_speed = kmh_to_ms(current_speed_kmh)
    next_speed = kmh_to_ms(next_speed_kmh)
    slope_acceleration = 0
    combined_acceleration = acc + slope_acceleration
    max_allowed_distance = current_distance - get_previous_limit_change_distance(current_distance)
    total_deceleration = brake_deceleration + slope_acceleration
    total_distance = (current_speed ** 2 - next_speed ** 2) / (2 * total_deceleration)
    if total_distance < max_allowed_distance:
        start_distance = int(current_distance - total_distance)
        end_distance = int(current_distance)
        for d in range(start_distance, end_distance):
            if d >= 0 and d < len(speed_limits_minus_7):
                s = d - start_distance + 1
                speed_limits_minus_7[d] = ms_to_kmh(np.sqrt(current_speed ** 2 - 2 * total_deceleration * s))
        return total_distance
    else:
        def equation(v):
            return ((v + 0.5) ** 2 - v ** 2) + (v + 0.5) * 0.7 + (
                    (v + 0.5) ** 2 - next_speed ** 2) / 2.4 - max_allowed_distance

        v_initial_guess = current_speed
        v_solution = fsolve(equation, v_initial_guess)[0]
        if v_solution < 0:
            raise ValueError("No valid solution found for initial speed.")
        remaining_distance = max_allowed_distance
        remaining_speed = v_solution
        start_distance = int(current_distance - remaining_distance)
        final_distance = int(current_distance)
        for d in range(start_distance, final_distance):
            if d >= 0 and d < len(speed_limits_minus_7):
                s = d - start_distance + 1
                speed_limits_minus_7[d] = ms_to_kmh(np.sqrt(v_solution ** 2 - 2 * acc * s))
        return max_allowed_distance


# Extracting relevant columns from each dataframe
station_limits = station_df[['限速起点（m）', '限速终点（m）', '限速值（km/h）']]
curve_limits = curve_df[['限速起点（m）', '限速终点（m）', '限速值（km/h）']]

# Renaming columns for consistency
station_limits.columns = ['start', 'end', 'speed_limit']
curve_limits.columns = ['start', 'end', 'speed_limit']

# Combining both dataframes
combined_limits = pd.concat([station_limits, curve_limits])

# Sorting by start point for better visualization
combined_limits = combined_limits.sort_values(by='start').reset_index(drop=True)

# Define the maximum distance based on the provided data
max_distance = max(combined_limits['end'])

# Create an array to hold the speed limits for every meter
speed_limits = np.full(int(max_distance) + 1, np.inf)

# Iterate through the combined limits and update the speed limits array
for _, row in combined_limits.iterrows():
    start = int(row['start'])
    end = int(row['end'])
    speed_limits[start:end + 1] = np.minimum(speed_limits[start:end + 1], row['speed_limit'])

# Replace any remaining np.inf with a reasonable default value (e.g., maximum allowed speed)
max_allowed_speed = basicinfo_df['线路最大限速（km/h）'].max()
speed_limits[speed_limits == np.inf] = max_allowed_speed

# 找到限速变小的点
decrease_points_EBI = [(i, speed_limits[i] - 4) for i in range(1, len(speed_limits)) if
                       speed_limits[i] < speed_limits[i - 1]]
decrease_points_SBI = [(i, speed_limits[i] - 7) for i in range(1, len(speed_limits)) if
                       speed_limits[i] < speed_limits[i - 1]]

# Create arrays for the additional lines
speed_limits_minus_4 = speed_limits - 4
speed_limits_minus_7 = speed_limits - 7

# 打印
print("Points where the speed limit decreases:")
for point in decrease_points_EBI:
    print(f"Distance: {point[0]} m, Speed Limit: {point[1]} km/h")
    d = calculate_deceleration_distance(speed_limits_minus_4[point[0] - 1], point[1], point[0])
    calculate_deceleration_distance_SBI(speed_limits_minus_7[point[0] - 1], point[1]-3, point[0])
    # print(d)

# Plotting the static speed limit chart for every meter
plt.figure(figsize=(100, 6))
plt.plot(range(len(speed_limits)), speed_limits, color='blue', label='当前限速')
plt.plot(range(len(speed_limits_minus_4)), speed_limits_minus_4, color='red', linestyle='--', label='EBI')
plt.plot(range(len(speed_limits_minus_7)), speed_limits_minus_7, color='green', linestyle='--', label='SBI')
for point in decrease_points_EBI:
    plt.plot(point[0], point[1], 'ro')  # red circles for decrease points

plt.xlabel('行驶距离 (m)')
plt.ylabel('限制速度 (km/h)')
plt.title('静态限速图（每米）')
plt.ylim(0, max(speed_limits) + 30)
plt.xlim(0, len(speed_limits))

plt.grid(True)
plt.legend()
plt.show()
