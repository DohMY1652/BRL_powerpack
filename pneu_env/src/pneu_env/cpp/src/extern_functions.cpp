#include "PneumaticSimulator.h"

extern "C" {
    PneumaticSimulator* create_pneumatic_simulator() { return new PneumaticSimulator(); }

    void simulate(PneumaticSimulator* ps, double simulation_time, double ctrl_pos, double ctrl_neg) { return ps -> simulate(simulation_time, ctrl_pos, ctrl_neg); }

    void show_state(PneumaticSimulator* ps) { return ps -> show_state(); }

    void reset_curr_time(PneumaticSimulator* ps) { return ps -> reset_curr_time(); }
    void set_curr_time(PneumaticSimulator* ps, double curr_time) { return ps -> set_curr_time(curr_time); }
    void set_time_step(PneumaticSimulator* ps, double time_step) { return ps -> set_time_step(time_step); }
    void set_press(PneumaticSimulator* ps, double press_pos, double press_neg) { return ps -> set_press(press_pos, press_neg); }
    void set_piston_press(PneumaticSimulator* ps, double press_pis1, double press_pis2) { return ps -> set_piston_press(press_pis1, press_pis2); }
    void set_logger(PneumaticSimulator* ps, const char* file_path) { return ps -> set_logger(file_path); }
    void set_name(PneumaticSimulator* ps, const char* name) { return ps -> set_name(name); } 

    double get_curr_time(PneumaticSimulator* ps) { return ps -> get_curr_time(); }
    double get_press_pos(PneumaticSimulator* ps) { return ps -> get_press_pos(); }
    double get_press_neg(PneumaticSimulator* ps) { return ps -> get_press_neg(); }
    double get_press_pis1(PneumaticSimulator* ps) { return ps -> get_press_pis1(); }
    double get_press_pis2(PneumaticSimulator* ps) { return ps -> get_press_pis2(); }

    void set_chamber_volume(PneumaticSimulator*ps, double pos_chamber_volume, double neg_chamber_volume) { return ps -> get_system() -> set_chamber_volume(pos_chamber_volume, neg_chamber_volume); }
    void set_discharge_coefficients(PneumaticSimulator* ps, double Cd_pump_in, double Cd_pump_out) { return ps -> get_system() -> set_discharge_coefficients(Cd_pump_in, Cd_pump_out); }
    void set_solenoid_valve_constants(PneumaticSimulator* ps, double Cd_valve_per_k, double current_constant, double min_spring_force) { return ps -> get_system() -> set_solenoid_valve_constants(Cd_valve_per_k, current_constant, min_spring_force); }

    void verbose(PneumaticSimulator* ps) { return ps -> get_system() -> verbose(); }
    void quiet(PneumaticSimulator* ps) { return ps -> get_system() -> quiet(); }
    void clear_lines(PneumaticSimulator* ps, int number) { return ps -> get_system() -> clear_lines(number); }

    double get_motor_angular_velocity(PneumaticSimulator* ps) { return ps -> get_system() -> get_motor_angular_velocity(); }
    double get_diff_press_pos(PneumaticSimulator* ps) { return ps -> get_system() -> get_diff_press_pos(); }
    double get_diff_press_neg(PneumaticSimulator* ps) { return ps -> get_system() -> get_diff_press_neg(); }
    double get_diff_press_pis1(PneumaticSimulator* ps) { return ps -> get_system() -> get_diff_press_pis1(); }
    double get_diff_press_pis2(PneumaticSimulator* ps) { return ps -> get_system() -> get_diff_press_pis2(); }

    double get_pump_in_mass_flowrate(PneumaticSimulator* ps) { return ps -> get_system() -> get_pump_in_mass_flowrate(); }
    double get_pump_out_mass_flowrate(PneumaticSimulator* ps) { return ps -> get_system() -> get_pump_out_mass_flowrate(); }
    double get_pos_solenoid_valve_mass_flowrate(PneumaticSimulator* ps) { return ps -> get_system() -> get_pos_solenoid_valve_mass_flowrate(); }
    double get_neg_solenoid_valve_mass_flowrate(PneumaticSimulator* ps) { return ps -> get_system() -> get_neg_solenoid_valve_mass_flowrate(); }
    double get_pos_chamber_mass_flowrate(PneumaticSimulator* ps) { return  ps -> get_system() -> get_pos_chamber_mass_flowrate(); }
    double get_neg_chamber_mass_flowrate(PneumaticSimulator* ps) { return  ps -> get_system() -> get_neg_chamber_mass_flowrate(); }

    void show_discharge_coefficients(PneumaticSimulator* ps) { return ps -> get_system() -> show_discharge_coefficients(); }
    void show_mass_flowrates(PneumaticSimulator* ps) { return ps -> get_system() -> show_mass_flowrates(); }
    void show_params(PneumaticSimulator* ps) { return ps -> get_system() -> show_params(); }

    void destroy_logger(PneumaticSimulator* ps) { return ps -> destroy_logger(); }
    void destroy_simulator(PneumaticSimulator* ps) { delete ps; }
}