%% Justin Robinson

clear
close all
clc

Pa2PSI = 6894.7573;  % Constant to change Pa to PSI
Pmanifold = 400;  % Manifold pressure in PSI

mdot = 1.9;  % Total engine mass flow rate
of = 1.2;  % O/F ratio of engine

Fdensity = 789;  % Fuel density
Odensity = 1141;  % Oxidizer density

OXOrficeArea = 0.00005938; %% Change when actual values are given
FOrficeArea = 0.00005938;

SysFuelCdA = 0.00002058330588;
SysOxCdA = 0.00007119840625;

    OxCD = .51 ;
    FuelCD = .3540;
    
    
    FcdA = FOrficeArea * FuelCD;  % Fuel Side CdA
    OcdA = OXOrficeArea * OxCD;  % Ox Side CdA

    % Reg Calculations
    Fmdot = mdot / (of + 1);  % Fuel side mdot
    Omdot = of * Fmdot;  % Ox side mdot

    % Manifold Pressure drop
    dPFuel = (((Fmdot)^2) / ((2 * Fdensity) * (FcdA^2))) / Pa2PSI;  % Fuel side Pressure drop
    dPOx = (((Omdot)^2) / ((2 * Odensity) * (OcdA^2))) / Pa2PSI;  % Ox side Pressure drop

    % System Pressure drop
    SysFueldP = (((Fmdot)^2) / ((2 * Fdensity) * (SysFuelCdA^2))) / Pa2PSI;
    SysOxdP = (((Omdot)^2) / ((2 * Odensity) * (SysOxCdA^2))) / Pa2PSI;

    % Total regulator pressure
    Fuelreg = Pmanifold + dPFuel + SysFueldP;  % Total fuel reg pressure 
    Oxreg = Pmanifold + dPOx + SysOxdP;  % Total Ox reg pressure

    fprintf(' Fuel reg pressure: %.2f PSI | Ox reg pressure: %.2f PSI | Ox manifold dp: %.2f PSI |fuel manifold dp: %.2f PSI\n', Fuelreg, Oxreg, dPOx, dPFuel);


