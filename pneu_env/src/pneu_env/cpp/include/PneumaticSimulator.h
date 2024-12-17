#ifndef PNEUMATICSIMULATOR_H
#define PNEUMATICSIMULATOR_H

#include <string>
#include "PneumaticSystem.h"
#include "PneumaticLogger.h"

#define ATM 101.325

class PneumaticSimulator {
public:
    PneumaticSimulator();
    ~PneumaticSimulator();

    void simulate(double simulation_time, double ctrl_pos, double ctrl_neg);
    void simulate_once(double ctrl_pos, double ctrl_neg);

    void show_state();

    void reset_curr_time() { curr_time_ = 0; };
    void set_curr_time(double curr_time) { curr_time_ = curr_time; };
    void set_time_step(double time_step);

    void set_press(double press_pos, double press_neg);
    void set_piston_press(double press_pis1, double press_pis2);

    void set_logger(const char* file_name);
    void set_name(const char* name);

    PneumaticSystem* get_system() const { return system_; };

    double get_curr_time() const { return curr_time_; };
    double get_press_pis1() const { return press_pis1_; };
    double get_press_pis2() const { return press_pis2_; };
    double get_press_pos() const { return press_pos_; };
    double get_press_neg() const { return press_neg_; };

    void destroy_logger();

private:
    PneumaticSystem* system_;
    PneumaticLogger* logger_;

    double time_step_;

    double curr_time_;
    double motor_angle_;
    double press_pis1_, press_pis2_;
    double press_pos_, press_neg_;

    bool is_log_;
    bool is_verbose_;
};

#endif