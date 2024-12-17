// #include <iostream>
#include <algorithm>

#define ATM 101.325 // kPa
#define PI 3.141592
#define K 1.4
#define R 0.287 // kJ/kg/K

double orifice_base(double press_inlet, double press_outlet, double temperature) {
    double mass_flowrate; // [ kg/s ]

    double critical_press_ratio = pow(2/(K + 1), K/(K - 1));
    double press_ratio = press_outlet/press_inlet;
    double R_ = 1000*R; // [ J/kg/K ]
    
    if (press_inlet >= press_outlet) {
        if (press_ratio <= critical_press_ratio) {
            mass_flowrate = (1000*press_inlet/sqrt(R_*temperature))*sqrt(K*pow(2/(K + 1), (K + 1)/(K - 1)));
        } else {
            mass_flowrate = (1000*press_inlet/sqrt(R_*temperature))*sqrt(2*K/(K - 1))*sqrt(pow(press_ratio, 2/K) - pow(press_ratio, (K + 1)/K));
        }
    } else {
        mass_flowrate = 0;
    }

    return mass_flowrate; // Unit: [ kg/s ]
}

extern "C" { 
    double solenoid_valve(double press_inlet, double press_outlet, double current) {
        double solenoid_valve_diameter = 1.6*0.01; // [ m ]
        double solenoid_valve_area = 0.25*PI*solenoid_valve_diameter*solenoid_valve_diameter;
        double Cd_valve_per_k = 1.2453e-06;
        double current_constant = 1579.43421193886;
        double min_spring_force = 273.667595049522;
        double temperature = 293.15;

        double coeff = solenoid_valve_diameter*PI*Cd_valve_per_k*(current_constant*current + solenoid_valve_area*1000*press_inlet - min_spring_force);
        
        double mass_flowrate = coeff*orifice_base(press_inlet, press_outlet, temperature);

        return (mass_flowrate >= 0) ? mass_flowrate : 0;

    }
}
