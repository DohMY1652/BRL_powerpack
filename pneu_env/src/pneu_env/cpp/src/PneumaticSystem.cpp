#include <iostream>
#include "PneumaticSystem.h"

PneumaticSystem::PneumaticSystem() :
    motor_angular_velocity_(3000 * (2*PI/60)), // [ rpm -> rad/s ]

    pos_chamber_volume_(0.75), // [ L ]
    neg_chamber_volume_(0.4), // [ L ]

    diff_press_pos_(0), // [ kPa/s ]
    diff_press_neg_(0), // [ kPa/s ]
    diff_press_pis1_(0), // [ kPa/s ]
    diff_press_pis2_(0), // [ kPa/s ]

    pump_in_mass_flowrate_(0), // [ kg/s ]
    pump_out_mass_flowrate_(0), // [ kg/s ]
    pos_solenoid_valve_mass_flowrate_(0), // [ kg/s ] 
    neg_solenoid_valve_mass_flowrate_(0), // [ kg/s ]
    pos_chamber_mass_flowrate_(0), // [ kg/s ] 
    neg_chamber_mass_flowrate_(0), // [ kg/s ]

    temperature_(293.15), // [ K ]
    pump_temperature_(323.15), // [ K ]

    slider_crank_radius_(2), // [ cm ]
    slider_crank_lod_(7), // [ cm ]

    piston_diameter_(7), // [ cm ]
    piston_gap_(0.1), // [ cm ]

    check_valve_area_(148*0.1/180), // [ cm * cm ]

    solenoid_valve_diameter_(1.6), // [ cm ]
    solenoid_valve_gap_(0.16), // [ cm ]

    Cd_valve_per_k_(1.2453e-06),
    current_constant_(1579.43421193886),
    min_spring_force_(273.667595049522),

    Cd_pump_in_(1.0150850183877944),
    Cd_pump_out_(0.023304388618618336),

    is_verbose_(false),
    name_("system")
{
    piston_area_ = 0.25*PI*pow(0.01*piston_diameter_,2); // [ m*m ]
    piston_dead_volume_ = 0.01*piston_area_*piston_gap_; // [ m*m*m ]

    solenoid_valve_area_ = 0.25*PI*pow(0.01*solenoid_valve_diameter_, 2); // [ m*m ]
}

void PneumaticSystem::model(
    double motor_angle,
    double press_pis1, double press_pis2,
    double press_pos, double press_neg,
    double ctrl_pos, double ctrl_neg
) {
    double piston1_position = piston_position(motor_angle, 0); // [ m ]
    double piston2_position = piston_position(motor_angle, PI); // [ m ]
    double piston1_velocity = piston_velocity(motor_angle, motor_angular_velocity_, 0); // [ m/s ]
    double piston2_velocity = piston_velocity(motor_angle, motor_angular_velocity_, PI); // [ m/s ]

    double piston1_volume = piston_volume(piston1_position); // [ m*m*m ]
    double piston2_volume = piston_volume(piston2_position); // [ m*m*m ]
    double piston1_diff_volume = piston_diff_volume(piston1_velocity); // [ m*m*m/s ]
    double piston2_diff_volume = piston_diff_volume(piston2_velocity); // [ m*m*m/s ]

    double piston1_out_mass_flowrate = Cd_pump_out_*check_valve_base(press_pis1, press_pos);
    double piston2_out_mass_flowrate = Cd_pump_out_*check_valve_base(press_pis2, press_pos);
    double piston1_in_mass_flowrate = Cd_pump_in_*check_valve_base(press_neg, press_pis1);
    double piston2_in_mass_flowrate = Cd_pump_in_*check_valve_base(press_neg, press_pis2);

    pump_out_mass_flowrate_ = piston1_out_mass_flowrate + piston2_out_mass_flowrate;
    pump_in_mass_flowrate_ = piston1_in_mass_flowrate + piston2_in_mass_flowrate;

    diff_press_pis1_ = piston_diff_pressure(press_pis1, piston1_volume, piston1_diff_volume, piston1_in_mass_flowrate - piston1_out_mass_flowrate);
    diff_press_pis2_ = piston_diff_pressure(press_pis2, piston2_volume, piston2_diff_volume, piston2_in_mass_flowrate - piston2_out_mass_flowrate);

    pos_solenoid_valve_mass_flowrate_ = solenoid_valve(press_pos, ATM, 0.165*ctrl_pos);
    neg_solenoid_valve_mass_flowrate_ = solenoid_valve(ATM, press_neg, 0.165*ctrl_neg);

    pos_chamber_mass_flowrate_ = piston1_out_mass_flowrate + piston2_out_mass_flowrate - pos_solenoid_valve_mass_flowrate_;
    neg_chamber_mass_flowrate_ = - piston1_in_mass_flowrate - piston2_in_mass_flowrate + neg_solenoid_valve_mass_flowrate_;

    diff_press_pos_ = chamber_diff_pressure(press_pos, 0.001*pos_chamber_volume_, pos_chamber_mass_flowrate_);
    diff_press_neg_ = chamber_diff_pressure(press_neg, 0.001*neg_chamber_volume_, neg_chamber_mass_flowrate_);
}

double PneumaticSystem::piston_position(double motor_angle, double phase) {
    double r = 0.01*slider_crank_radius_; // [ cm -> m ]
    double l = 0.01*slider_crank_lod_; // [ cm -> m ]
    double angle = fmod(motor_angle + phase, 2*PI); // [ rad ]

    // Unit: [ m ]
    return r*cos(angle) + sqrt(l*l - pow(r*sin(angle), 2)) + r - l;
}

double PneumaticSystem::piston_velocity(double motor_angle, double motor_angular_velocity, double phase) {
    double r = 0.01*slider_crank_radius_; // [ cm -> m ]
    double l = 0.01*slider_crank_lod_; // [ cm -> m ]
    double angle = fmod(motor_angle + phase, 2*PI); // [ rad ]
    double angular_velocity = motor_angular_velocity; // [ rad/s ]

    // Unit: [ m/s ]    
    return (- r*sin(angle) - r*r*sin(angle)*cos(angle)/sqrt(l*l - pow(r*sin(angle),2)))*angular_velocity;
}

double PneumaticSystem::piston_volume(double piston_position) {
    double r = 0.01*slider_crank_radius_; // [ m ]
    // Unit: [ m*m*m ]
    return piston_area_*(2*r - piston_position) + piston_dead_volume_;
}

double PneumaticSystem::piston_diff_volume(double piston_velocity) {
    // Unit: [ m*m*m/s ]
    return - piston_area_*piston_velocity;
}

double PneumaticSystem::orifice_base(double press_inlet, double press_outlet, double temperature) {
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

double PneumaticSystem::check_valve_base(double press_inlet, double press_outlet) {
    double coeff = 0.01*0.01*check_valve_area_; // [ m*m ]
    // Unit: [ kg/s ]
    return coeff*orifice_base(press_inlet, press_outlet, pump_temperature_);
}

double PneumaticSystem::solenoid_valve(double press_inlet, double press_outlet, double current) {
    double coeff = solenoid_valve_diameter_*PI*Cd_valve_per_k_*(current_constant_*current + solenoid_valve_area_*1000*press_inlet - min_spring_force_);
    double mass_flowrate = coeff*orifice_base(press_inlet, press_outlet, temperature_);
    return (mass_flowrate >= 0) ? mass_flowrate : 0;
}

double PneumaticSystem::piston_diff_pressure(double press, double volume, double volumic_flowrate, double mass_flowrate) {
    // press [ kPa ], volume [ m*m*m ], volumic_flowrate [ m*m*m/s ], mass_flowrate [ kg/s ]
    // Unit: [ kPa/s ]
    return - press*volumic_flowrate/volume + mass_flowrate*R*pump_temperature_/volume;
}

double PneumaticSystem::chamber_diff_pressure(double press, double volume, double mass_flowrate) {
    // press [ kPa ], volume [ m*m*m ], mass_flowrate [ kg/s ]
    // Unit: [ kPa/s ]
    return mass_flowrate*R*temperature_/volume;
}

void PneumaticSystem::set_chamber_volume(double pos_chamber_volume, double neg_chamber_volume) {
    pos_chamber_volume_ = pos_chamber_volume;
    neg_chamber_volume_ = neg_chamber_volume;
    if (is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Chamber volume" << std::endl;
        std::cout << "        ----------" << std::endl;
        std::cout << "        > Postive channel chamber volume : " << pos_chamber_volume_ << std::endl;
        std::cout << "        > negaive channel chamber volume : " << neg_chamber_volume_ << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSystem::set_discharge_coefficients(
    double Cd_pump_in, double Cd_pump_out
){
    Cd_pump_in_ = Cd_pump_in;
    Cd_pump_out_ = Cd_pump_out;

    if (is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Discharge coefficients" << std::endl;
        std::cout << "        ----------" << std::endl;
        std::cout << "        > Pump inlet discharge coefficient : " << Cd_pump_in_ << std::endl;
        std::cout << "        > Pump outlet discharge coefficient : " << Cd_pump_out_ << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSystem::set_solenoid_valve_constants(double Cd_valve_per_k, double current_constant, double min_spring_force) {
    Cd_valve_per_k_ = Cd_valve_per_k;
    current_constant_ = current_constant;
    min_spring_force_ = min_spring_force;

    if (is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Solenoid valve coefficients" << std::endl;
        std::cout << "        ----------" << std::endl;
        std::cout << "        > Valve discharge coefficient per spring constant : " << Cd_valve_per_k_ << std::endl;
        std::cout << "        > Current constant : " << current_constant_ << std::endl;
        std::cout << "        > Minimum spring force : " << min_spring_force_ << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSystem::show_mass_flowrates() {
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Mass flowrates" << std::endl;
    std::cout << "        ----------" << std::endl;
    std::cout << "        > Pump inlet mass flowrate : " << pump_in_mass_flowrate_ << std::endl;
    std::cout << "        > Pump outlet mass flowrate : " << pump_out_mass_flowrate_ << std::endl;
    std::cout << "        > Positive channel solenoid valve mass flowrate : " << pos_solenoid_valve_mass_flowrate_ << std::endl;
    std::cout << "        > Negative channel solenoid valve mass flowrate : " << neg_solenoid_valve_mass_flowrate_ << std::endl;
    std::cout << "        > Positive channel chamber mass flowrate : " << pos_chamber_mass_flowrate_ << std::endl;
    std::cout << "        > Negative channel chamber mass flowrate : " << neg_chamber_mass_flowrate_ << std::endl;
    std::cout << "        ----------" << std::endl;
}

void PneumaticSystem::show_discharge_coefficients() {
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Discharge coefficients" << std::endl;
    std::cout << "        ----------" << std::endl;
    std::cout << "        > Pump inlet discharge coefficient : " << Cd_pump_in_ << std::endl;
    std::cout << "        > Pump outlet discharge coefficient : " << Cd_pump_out_ << std::endl;
    std::cout << "        ----------" << std::endl;
}

void PneumaticSystem::show_params() {
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> parameters information" << std::endl;
    std::cout << "        ----------" << std::endl;
    std::cout << "        << Discharge coefficients >>" << std::endl;
    std::cout << "        > Pump inlet discharge coefficient : " << Cd_pump_in_ << std::endl;
    std::cout << "        > Pump outlet discharge coefficient : " << Cd_pump_out_ << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        << Solenoid valve parameters >>" << std::endl;
    std::cout << "        > Valve discharge coefficient per spring constant : " << Cd_valve_per_k_ << std::endl;
    std::cout << "        > Current constant : " << current_constant_ << std::endl;
    std::cout << "        > Minimum spring force : " << min_spring_force_ << std::endl;
    std::cout << "        " << std::endl;
    std::cout << "        << Common parameters >>" << std::endl;
    std::cout << "        > motor angular velocity : " << motor_angular_velocity_*(60/2/PI) << " RPM" << std::endl;
    std::cout << "        > positive channel chamber volume : " << pos_chamber_volume_ << " L" << std::endl;
    std::cout << "        > negative channel chamber volume : " << neg_chamber_volume_ << " L" << std::endl;
    std::cout << "        > temperature of inside pump : " << pump_temperature_ << " K" << std::endl;
    std::cout << "        > temperature of outside pump : " << temperature_ << " K" << std::endl;
    std::cout << "        > slider crank radius : " << slider_crank_radius_ << " cm" << std::endl;
    std::cout << "        > slider crank lod : " << slider_crank_lod_ << " cm" << std::endl;
    std::cout << "        > piston diameter : " << piston_diameter_ << " cm" << std::endl;
    std::cout << "        > piston gap : " << piston_gap_ << " cm" << std::endl;
    std::cout << "        > check valve area : " << check_valve_area_ << " cm^2" << std::endl;
    std::cout << "        > solenoid valve diameter : " << solenoid_valve_diameter_ << " cm" << std::endl;
    std::cout << "        > solenoid valve gap : " << solenoid_valve_gap_ << " cm" << std::endl;
    std::cout << "        > heat capacity ratio : " << K << "" << std::endl;
    std::cout << "        > specific ideal gas constant : " << R << " kJ/(kg*K)" << std::endl;
    std::cout << "        ----------" << std::endl;
}

void PneumaticSystem::verbose(){
    is_verbose_ = true;
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Verbose" << std::endl;
}

void PneumaticSystem::quiet() {
    is_verbose_ = false;
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic system C++) " << name_ << " ==> Quiet" << std::endl;
}

void PneumaticSystem::clear_lines(int number) { 
    for (int i = 0; i < number; i++) {
        std::cout << "\033[A\033[2K\r" << std::flush; 
    }
    // std::cout << "" << std::endl;
}