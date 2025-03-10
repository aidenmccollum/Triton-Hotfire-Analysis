import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mdot import mdot_from_sys

# Read the Excel file
df = pd.read_excel('Mar_7_TritonHF.xlsx')


# Create LOX pressure plot
plt.figure()
mask = (df['Time (s)'] >= 382) & (df['Time (s)'] <= 392)

plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'LOX Reg Downstream (psi)'], label='LOX Reg Downstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'LOX Tank Upstream (psi)'], label='LOX Tank Upstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'LOX Tank Downstream (psi)'], label='LOX Tank Downstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'LOX Engine Manifold (psi)'], label='LOX Engine Manifold')

plt.title('LOX System Pressures vs Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Pressure (psi)', fontsize=12)
plt.grid(True)
plt.legend()
plt.xlim(-2, 6)
plt.savefig('results/lox_system_pressures.png', dpi=300, bbox_inches='tight')

# Create fuel pressure plot
plt.figure()

plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'Fuel Reg Downstream (psi)'], label='Fuel Reg Downstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'Fuel Tank Upstream (psi)'], label='Fuel Tank Upstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'Fuel Tank Downstream (psi)'], label='Fuel Tank Downstream')
plt.plot(df.loc[mask, 'Time (s)']-384.6, df.loc[mask, 'Fuel Engine Manifold (psi)'], label='Fuel Engine Manifold')

plt.title('Fuel System Pressures vs Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Pressure (psi)', fontsize=12)
plt.grid(True)
plt.legend()
plt.xlim(-2, 6)
plt.savefig('results/Fuel_system_pressures.png', dpi=300, bbox_inches='tight')



################# Thrust plot ###################
#extracting the raw readings
load_a = df['Thrust LC A (lbf)']
load_b = df['Thrust LC B (lbf)']
load_c = df['Thrust LC C (lbf)']

thrust_dfs = []

#performing the transform from raw readings to thrust
for load_df in [load_a, load_b, load_c]:
    #values added to the raw voltages
    m = 2900.13
    b= 0.03433

    #load cell specs
    excitation_voltage = 10
    mv_v = 3.433
    load_max = 1000

    raw_voltage = ((load_df-b)/m)*1000#converting to mv
    load = (raw_voltage/(mv_v*excitation_voltage))*load_max
    thrust_dfs.append(load)


#Create the plot
plt.figure()
# letters = ["A", "B", "C"]
# index = 0
mask = (df['Time (s)'] >= 382) & (df['Time (s)'] <= 392)
# for thrust_df in thrust_dfs:
#     plt.plot(df.loc[mask, 'Time (s)'], thrust_df[mask], linewidth=2, label=f'Load Cell {letters[index]}', alpha=0.5)
#     index += 1

# Sum the thrust values from all load cells
total_thrust = sum(thrust_dfs)
plt.plot(df.loc[mask, 'Time (s)']-384.6, total_thrust[mask], linewidth=2, label='Triton Thrust (raw)', color='gray', alpha=0.5)

# Apply a rolling average to smooth the total thrust data
window_size = 50  # Adjust this value to change smoothing amount
smoothed_thrust = total_thrust[mask].rolling(window=window_size, center=True).mean()
plt.plot(df.loc[mask, 'Time (s)']-384.6, smoothed_thrust, linewidth=2, label='Triton Thrust (smoothed)', color='blue')

# Read the prometheus file
prom_df = pd.read_csv('promethesus.csv')
prom_mask = (prom_df['Time'] >= 196) & (prom_df['Time'] <= 206)
#plt.plot(prom_df.loc[prom_mask, 'Time']+186-384.6, prom_df.loc[prom_mask, 'Force (lbs)'], label='Prometheus (for scale)', color='red')

# Customize the plot
plt.title('Thrust vs Time - Triton Hot Fire', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Thrust (lbf)', fontsize=12)
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor grid lines every 0.5 seconds
plt.legend()
plt.xlim(-2, 6)

# Save and show the plot
plt.savefig('results/thrust_plot.png', dpi=300, bbox_inches='tight')







##############  creating an ISP plot ###########################

#determining the mass flow over the burn
burn_mask = (df['Time (s)'] >= 384.8) & (df['Time (s)'] <= 389.4)
long_mask = (df['Time (s)'] >= 379.8) & (df['Time (s)'] <= 394.4)
pre_mask =  (df['Time (s)'] >= 379.8) & (df['Time (s)'] <= 384)
post_mask =  (df['Time (s)'] >= 389.6) & (df['Time (s)'] <= 394.5)
# Get LOX weight data
lox_weight = df.loc[burn_mask, 'LOX Tank Weight (lbf)']
fuel_weight = df.loc[burn_mask, 'Fuel Tank Weight (lbf)']
lox_weight_long = df.loc[long_mask, 'LOX Tank Weight (lbf)']
fuel_weight_long = df.loc[long_mask, 'Fuel Tank Weight (lbf)']

# Smooth the data with rolling average
window_size = 50
lox_weight_smooth = lox_weight_long.rolling(window=window_size, center=True).mean().tolist()
fuel_weight_smooth = fuel_weight_long.rolling(window=window_size, center=True).mean().tolist()

# Create plot for propellant weights
time = df.loc[burn_mask, 'Time (s)']
long_time = df.loc[long_mask, 'Time (s)']

plt.figure()
plt.plot(long_time - 384.6, lox_weight_smooth, label='LOX Weight', color='blue', alpha=0.25)
plt.plot(long_time - 384.6, fuel_weight_smooth, label='Fuel Weight', color='red', alpha=0.25)
# Add linear regression lines
# lox_fit = np.polyfit(long_time.tolist(), lox_weight_long.tolist(), 1)
# fuel_fit = np.polyfit(long_time.tolist(), fuel_weight_long.tolist(), 1)


# # Create polynomial function
# lox_weight_avg = np.poly1d(lox_fit)
# fuel_weight_avg = np.poly1d(fuel_fit)


# plt.plot(time-384.6, np.poly1d(lox_fit)(time), color='blue', label='LOX Average Weight')
# plt.plot(time-384.6, np.poly1d(fuel_fit)(time), color='red', label='Fuel Average Weight')


lox_weight_pre = df.loc[pre_mask, 'LOX Tank Weight (lbf)'].mean()
lox_weight_post = df.loc[post_mask, 'LOX Tank Weight (lbf)'].mean()

fuel_weight_pre = df.loc[pre_mask, 'Fuel Tank Weight (lbf)'].mean()
fuel_weight_post = df.loc[post_mask, 'Fuel Tank Weight (lbf)'].mean()

print(f"hand lox calcs: {((lox_weight_pre-lox_weight_post)/4.72)/2.205}")
print(f"hand fuel calcs: {((fuel_weight_pre-fuel_weight_post)/4.72)/2.205}")


# Calculate line of best fit points for fuel weights
x_fuel_fit = [df.loc[pre_mask, 'Time (s)'].values[-1]+0.25, df.loc[post_mask, 'Time (s)'].values[0]]
y_fuel_fit = [fuel_weight_pre, fuel_weight_post]
fuel_fit = np.polyfit(x_fuel_fit, y_fuel_fit, 1)
plt.plot([x - 384.6 for x in x_fuel_fit], np.poly1d(fuel_fit)(x_fuel_fit), color='red', label='Fuel Weight Trendline')

#repeating the process for lox
x_lox_fit = [df.loc[pre_mask, 'Time (s)'].values[-1], df.loc[post_mask, 'Time (s)'].values[0]]
y_lox_fit = [lox_weight_pre, lox_weight_post]
lox_fit = np.polyfit(x_lox_fit, y_lox_fit, 1)
plt.plot([x - 384.6 for x in x_lox_fit], np.poly1d(lox_fit)(x_lox_fit), color='blue', label='LOX Weight Trendline')
plt.title('Propellant Weights vs Time', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Weight (lbf)', fontsize=12)
plt.grid(True)
plt.legend()
plt.savefig('results/propellant_weights.png', dpi=300, bbox_inches='tight')

mdot_lox_tank = abs(lox_fit[0])
mdot_fuel_tank = abs(fuel_fit[0])
mdot_tot_tank = mdot_fuel_tank+mdot_lox_tank

print(f"Average LOX mass flow rate FROM TANK: {(mdot_lox_tank/2.205):.4f} kg/s")
print(f"Average fuel mass flow rate FROM TANK: {(mdot_fuel_tank/2.205):.4f} kg/s")


mdot_ox_sys, mdot_fuel_sys = mdot_from_sys()

window_size = 50
mdot_fuel_smooth = mdot_fuel_sys.rolling(window=window_size, center=True).mean()
mdot_lox_smooth = mdot_ox_sys.rolling(window=window_size, center=True).mean()

mdot_total_sys = mdot_fuel_smooth+mdot_lox_smooth

mdot_fuel_avg = mdot_fuel_smooth.mean()
mdot_lox_avg = mdot_lox_smooth.mean()
print(f"Average LOX mass flow rate FROM SYSTEM: {mdot_lox_avg:.4f} kg/s")
print(f"Average fuel mass flow rate FROM SYSTEM: {mdot_fuel_avg:.4f} kg/s")

plt.figure()
plt.plot(df.loc[burn_mask, 'Time (s)']- 384.6, mdot_fuel_sys, color='gray', alpha=0.5, label='Fuel mdot (raw)')
plt.plot(df.loc[burn_mask, 'Time (s)']- 384.6, mdot_fuel_smooth, color='red', label='Fuel mdot (smoothed)')
plt.plot(df.loc[burn_mask, 'Time (s)']- 384.6, mdot_ox_sys, color='gray', alpha=0.5, label='LOX mdot (raw)')
plt.plot(df.loc[burn_mask, 'Time (s)']- 384.6, mdot_lox_smooth, color='blue', label='LOX mdot (smoothed)')
plt.xlabel('Time (s)')
plt.ylabel('Mass Flow Rate (kg/s)')
plt.title('Propellant Mass Flow Rate vs Time')
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor grid lines every 0.5 seconds
plt.legend()
plt.xlim(0, 4.6)
plt.savefig('results/mass_flow_vs_time.png', dpi=300, bbox_inches='tight')

#OF plot
plt.figure()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, mdot_lox_smooth/mdot_fuel_smooth, label='O/F Ratio')
plt.xlabel('Time (s)')
plt.ylabel('O/F Ratio')
plt.title('Mixture Ratio vs Time')
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.legend()
plt.xlim(0, 4.6)
plt.savefig('results/OF_ratio.png', dpi=300, bbox_inches='tight')


#determining the chamber pressure by finding dp across injector
cda_inj_fu = 2.102052000000000e-05
cda_inj_ox = 2.85E-05

dP_inj = (((mdot_fuel_sys/cda_inj_fu)**2)/(2*789))/6895

dP_inj_ox = (((mdot_ox_sys/cda_inj_ox)**2)/(2*1141))/6895

Pc = df.loc[burn_mask, 'Fuel Engine Manifold (psi)']-dP_inj
Pc_ox = df.loc[burn_mask, 'LOX Engine Manifold (psi)']-dP_inj_ox
plt.figure()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, Pc.values, color='gray', alpha=0.5, label='Chamber Pressure from Fuel Sensor (raw)')
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, Pc_ox.values, color='gray', alpha=0.5, label='Chamber Pressure from Ox Sensor (raw)')

# Apply smoothing
Pc_smooth = pd.Series(Pc).rolling(window=window_size, center=True).mean()
Pc_smooth_ox = pd.Series(Pc_ox).rolling(window=window_size, center=True).mean()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, Pc_smooth, color='red', label='Chamber Pressure from fuel sensor (smoothed)')
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, Pc_smooth_ox, color='blue', label='Chamber Pressure from ox sensor(smoothed)')

plt.xlabel('Time (s)')
plt.ylabel('Chamber Pressure (psi)')
plt.title('Chamber Pressure vs Time')
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.legend()
plt.xlim(0, 5)
plt.savefig('results/chamber_pressure.png', dpi=300, bbox_inches='tight')




#plotting the isp
isp = total_thrust[burn_mask]*4.44822/(mdot_total_sys*9.81)
isp_smooth = isp.rolling(window=window_size*3, center=True).mean()

plt.figure()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, isp, linewidth=2, label='Triton ISP (raw)', color='gray', alpha=0.5)
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, isp_smooth, linewidth=2, label='Triton ISP (smoothed)', color='blue')
plt.title('Specific Impulse vs Time - Triton Hot Fire', fontsize=14)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Specific Impulse (s)', fontsize=12)
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))  # Minor grid lines every 0.5 seconds
plt.legend()
plt.xlim(0, 5)
plt.savefig('results/isp.png', dpi=300, bbox_inches='tight')



#FINALLY plotting the characteristic velocity
nozzle_area = 1152.509089 *10**-6
c_star = Pc*6895*nozzle_area/mdot_total_sys
c_star_theory = 400*6895*nozzle_area/1.9

plt.figure()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, c_star, color='gray', alpha=0.5, label='C* (raw)')
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, [c_star_theory]*len(df.loc[burn_mask]), '--', color='red', label='C* Theory')

# Apply smoothing
c_star_smooth = pd.Series(c_star).rolling(window=window_size*3, center=True).mean()
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, c_star_smooth, color='blue', label='C* (smoothed)')

plt.xlabel('Time (s)')
plt.ylabel('Characteristic Velocity (m/s)')
plt.title('Characteristic Velocity vs Time')
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.legend()
plt.xlim(0, 5)
plt.savefig('results/characteristic_velocity.png', dpi=300, bbox_inches='tight')


# Plot C* efficiency
plt.figure()
c_star_eff = (c_star/c_star_theory)*100
c_star_eff_smooth = pd.Series(c_star_eff).rolling(window=window_size*3, center=True).mean()

plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, c_star_eff, color='gray', alpha=0.5, label='C* Efficiency (raw)')
plt.plot(df.loc[burn_mask, 'Time (s)']-384.6, c_star_eff_smooth, color='blue', label='C* Efficiency (smoothed)')

plt.xlabel('Time (s)')
plt.ylabel('C* Efficiency')
plt.title('C* Efficiency vs Time')
plt.grid(True, which='major', alpha=0.5)
plt.grid(True, which='minor', alpha=0.2)
plt.minorticks_on()
plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.legend()
plt.xlim(0, 5)
plt.savefig('results/c_star_efficiency.png', dpi=300, bbox_inches='tight')




plt.show()


# Create a new DataFrame for the thrust values
thrust_data = pd.DataFrame({
    'Thrust A (lbf)': thrust_dfs[0],
    'Thrust B (lbf)': thrust_dfs[1],
    'Thrust C (lbf)': thrust_dfs[2],
    'Total Thrust (lbf)': total_thrust
})

# Write to Excel file
with pd.ExcelWriter('Mar_7_TritonHF_processed.xlsx') as writer:
    df.to_excel(writer, sheet_name='Raw Data', index=False)
    # Combine DataFrames
    combined_df = pd.concat([df, thrust_data], axis=1)
    combined_df.to_excel(writer, sheet_name='Processed Data', index=False)
    thrust_data.to_excel(writer, sheet_name='Thrust Only', index=False)