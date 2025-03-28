import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def mdot_from_sys():

    # Read the Excel file
    df = pd.read_excel('Mar_7_TritonHF.xlsx')
    burn_mask = (df['Time (s)'] >= 384.8) & (df['Time (s)'] <= 389.4)

    #cda of the system
    cda_sysf = 2.7189351186894784e-05
    cda_sysox= 5.49554605806529e-05

    # cda_sysf = 2.8643430403494142e-05
    # cda_sysox = 4.871850334397766e-05

    #defining the fuel characteristics
    rho_fuel = 789
    rho_ox = 1141


    dP_fuel = (df.loc[burn_mask, 'Fuel Tank Downstream (psi)']-df.loc[burn_mask, 'Fuel Engine Manifold (psi)'])*6895
    dP_ox = (df.loc[burn_mask, 'LOX Tank Downstream (psi)']-df.loc[burn_mask, 'LOX Engine Manifold (psi)'])*6895

    mdot_fuel = cda_sysf*(2*rho_fuel*dP_fuel)**0.5
    mdot_lox = cda_sysox*(2*rho_ox*dP_ox)**0.5

    return mdot_lox, mdot_fuel


