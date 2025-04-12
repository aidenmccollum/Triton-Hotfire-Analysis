import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mdot_from_sys(start,stop):

    # Read the Excel file
    df = pd.read_csv('Apr_11_TritonHF.csv')
    burn_mask = (df['Time (s)'] >= start) & (df['Time (s)'] <= stop)

    #cda of the system
    cda_sysf = 2.2037198954519152e-05
    cda_sysox= 5.7375829755755985e-05

    # cda_sysf = 2.8643430403494142e-05
    # cda_sysox = 4.871850334397766e-05

    #defining the fuel characteristics
    rho_fuel = 789
    rho_ox = 1141


    dP_fuel = (df.loc[burn_mask, 'Fuel Tank Downstream (psi)']-df.loc[burn_mask, 'Fuel Engine Manifold (psi)'])*6895
    dP_ox = (df.loc[burn_mask, 'LOX Reg Downstream (psi)']-20-df.loc[burn_mask, 'LOX Engine Manifold (psi)'])*6895

    mdot_fuel = cda_sysf*(2*rho_fuel*dP_fuel)**0.5
    mdot_lox = cda_sysox*(2*rho_ox*dP_ox)**0.5

    return mdot_lox, mdot_fuel


