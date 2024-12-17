#ifndef PNEUMATICSYSTEM_H
#define PNEUMATICSYSTEM_H 

#include <string>
#include "PneumaticLogger.h"

#define ATM 101.325
#define PI 3.14159265358979
#define K 1.4
#define R 0.287 // Specific gas constant [ kJ/kg/K ]

class PneumaticSystem {
public:
    PneumaticSystem();

    void model(
            double motor_angle,
            double press_pis1, double press_pis2,
            double press_pos, double press_neg,
            double ctrl_pos, double ctrl_neg
    );

    void set_chamber_volume(double pos_chamber_volume, double neg_chamber_volume);
    void set_solenoid_valve_constants(double Cd_valve_per_k, double current_constant, double min_spring_force);
    void set_discharge_coefficients(double Cd_pump_in, double Cd_pump_out);

    void set_name(const std::string& name) { name_ = name; };

    void verbose();
    void quiet();
    bool is_verbose() { return is_verbose_; };
    void clear_lines(int number);

    double get_motor_angular_velocity() const { return motor_angular_velocity_; };
    double get_diff_press_pos() const { return diff_press_pos_; };
    double get_diff_press_neg() const { return diff_press_neg_; };
    double get_diff_press_pis1() const { return diff_press_pis1_; };
    double get_diff_press_pis2() const { return diff_press_pis2_; };

    double get_pump_in_mass_flowrate() const { return pump_in_mass_flowrate_; };
    double get_pump_out_mass_flowrate() const { return pump_out_mass_flowrate_; };
    double get_pos_solenoid_valve_mass_flowrate() const { return pos_solenoid_valve_mass_flowrate_; };
    double get_neg_solenoid_valve_mass_flowrate() const { return neg_solenoid_valve_mass_flowrate_; };
    double get_pos_chamber_mass_flowrate() const { return pos_chamber_mass_flowrate_; };
    double get_neg_chamber_mass_flowrate() const { return neg_chamber_mass_flowrate_; };

    std::string get_name() const {return name_; };
    
    void show_discharge_coefficients();
    void show_mass_flowrates();
    void show_params();

private:
    double diff_press_pos_;
    double diff_press_neg_;
    double motor_angular_velocity_;
    double diff_press_pis1_;
    double diff_press_pis2_;

    double pump_in_mass_flowrate_, pump_out_mass_flowrate_;
    double pos_solenoid_valve_mass_flowrate_, neg_solenoid_valve_mass_flowrate_;
    double pos_chamber_mass_flowrate_, neg_chamber_mass_flowrate_;

    double pos_chamber_volume_, neg_chamber_volume_;

    const double temperature_, pump_temperature_;
    const double slider_crank_radius_, slider_crank_lod_;
    
    const double piston_diameter_, piston_gap_; // the smallest gap inside piston
    double piston_area_;
    double piston_dead_volume_;

    const double check_valve_area_;
    const double solenoid_valve_diameter_, solenoid_valve_gap_;

    double Cd_valve_per_k_;
    double current_constant_;
    double min_spring_force_;
    double solenoid_valve_area_;

    double Cd_pump_in_, Cd_pump_out_;

    bool is_verbose_;
    std::string name_;

    double piston_position(double motor_angle, double phase); // [ m ]
    double piston_velocity(double motor_angle, double motor_angular_velocity, double phase); // [ m/s ]
    double piston_volume(double piston_position); // [ m*m*m ]
    double piston_diff_volume(double piston_velocity); // [ m*m*m/s ]
    double piston_diff_pressure(double press, double volume, double volumic_flowrate, double mass_flowrate);
    double orifice_base(double press_inlet, double press_outlet, double temperature);
    double check_valve_base(double press_inlet, double press_outlet);
    double solenoid_valve(double press_inlet, double press_outlet, double ctrl);
    double chamber_diff_pressure(double press, double volume, double mass_flowrate);
};

#endif










