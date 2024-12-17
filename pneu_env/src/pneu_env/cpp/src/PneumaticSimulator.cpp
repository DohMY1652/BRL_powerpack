#include <iostream>
#include "PneumaticSimulator.h"

PneumaticSimulator::PneumaticSimulator(): 
    system_(new PneumaticSystem()),
    time_step_(0.0001),
    curr_time_(0),
    motor_angle_(0),
    press_pis1_(ATM),
    press_pis2_(ATM),
    press_pos_(ATM),
    press_neg_(ATM),

    logger_(nullptr), 
    is_log_(false),
    is_verbose_(false)
{}

PneumaticSimulator::~PneumaticSimulator() {
    if (is_log_) { 
        delete logger_; 
        if (get_system() -> is_verbose()) {
            std::cout << "" << std::endl;
            std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Log done" << std::endl;
        }
    }
    if (get_system() -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Close" << std::endl;
    }
    delete system_;
}

void PneumaticSimulator::simulate(double simulation_time, double ctrl_pos, double ctrl_neg) {
    int num_steps = (int)(simulation_time/time_step_) + 1;
    for (int i = 0; i < num_steps; i++) {
        simulate_once(ctrl_pos, ctrl_neg);
    }
}

void PneumaticSimulator::simulate_once(double ctrl_pos, double ctrl_neg) {
    // Runge-Kutta Integrator
    get_system() -> model(
        motor_angle_,
        press_pis1_, press_pis2_,
        press_pos_, press_neg_,
        ctrl_pos, ctrl_neg
    );

    double k1_motor_angular_velocity = get_system() -> get_motor_angular_velocity();
    double k1_diff_press_pis1 = get_system() -> get_diff_press_pis1();
    double k1_diff_press_pis2 = get_system() -> get_diff_press_pis2();
    double k1_diff_press_pos = get_system() -> get_diff_press_pos();
    double k1_diff_press_neg = get_system() -> get_diff_press_neg();

    double i2_motor_angle = motor_angle_ + k1_motor_angular_velocity*time_step_/3;
    double i2_press_pis1 = press_pis1_ + k1_diff_press_pis1*time_step_/3;
    double i2_press_pis2 = press_pis2_ + k1_diff_press_pis2*time_step_/3;
    double i2_press_pos = press_pos_ + k1_diff_press_pos*time_step_/3;
    double i2_press_neg = press_neg_ + k1_diff_press_neg*time_step_/3;

    get_system() -> model(
        i2_motor_angle,
        i2_press_pis1, i2_press_pis2,
        i2_press_pos, i2_press_neg,
        ctrl_pos, ctrl_neg
    );

    double k2_motor_angular_velocity = get_system() -> get_motor_angular_velocity();
    double k2_diff_press_pis1 = get_system() -> get_diff_press_pis1();
    double k2_diff_press_pis2 = get_system() -> get_diff_press_pis2();
    double k2_diff_press_pos = get_system() -> get_diff_press_pos();
    double k2_diff_press_neg = get_system() -> get_diff_press_neg();

    double i3_motor_angle = motor_angle_ - k1_motor_angular_velocity*time_step_/3 + k2_motor_angular_velocity*time_step_;
    double i3_press_pis1 = press_pis1_ - k1_diff_press_pis1*time_step_/3 + k2_diff_press_pis1*time_step_;
    double i3_press_pis2 = press_pis2_ - k1_diff_press_pis2*time_step_/3 + k2_diff_press_pis2*time_step_;
    double i3_press_pos = press_pos_ - k1_diff_press_pos*time_step_/3 + k2_diff_press_pos*time_step_;
    double i3_press_neg = press_neg_ - k1_diff_press_neg*time_step_/3 + k2_diff_press_neg*time_step_;

    get_system() -> model(
        i3_motor_angle,
        i3_press_pis1, i3_press_pis2,
        i3_press_pos, i3_press_neg,
        ctrl_pos, ctrl_neg
    );

    double k3_motor_angular_velocity = get_system() -> get_motor_angular_velocity();
    double k3_diff_press_pis1 = get_system() -> get_diff_press_pis1();
    double k3_diff_press_pis2 = get_system() -> get_diff_press_pis2();
    double k3_diff_press_pos = get_system() -> get_diff_press_pos();
    double k3_diff_press_neg = get_system() -> get_diff_press_neg();

    double i4_motor_angle = motor_angle_ + k1_motor_angular_velocity*time_step_ - k2_motor_angular_velocity*time_step_ + k3_motor_angular_velocity*time_step_;
    double i4_press_pis1 = press_pis1_ + k1_diff_press_pis1*time_step_ - k2_diff_press_pis1*time_step_ + k3_diff_press_pis1*time_step_;
    double i4_press_pis2 = press_pis2_ + k1_diff_press_pis2*time_step_ - k2_diff_press_pis2*time_step_ + k3_diff_press_pis2*time_step_;
    double i4_press_pos = press_pos_ + k1_diff_press_pos*time_step_ - k2_diff_press_pos*time_step_ + k3_diff_press_pos*time_step_;
    double i4_press_neg = press_neg_ + k1_diff_press_neg*time_step_ - k2_diff_press_neg*time_step_ + k3_diff_press_neg*time_step_;

    get_system() -> model(
        i4_motor_angle,
        i4_press_pis1, i4_press_pis2,
        i4_press_pos, i4_press_neg,
        ctrl_pos, ctrl_neg
    );

    double k4_motor_angular_velocity = get_system() -> get_motor_angular_velocity();
    double k4_diff_press_pis1 = get_system() -> get_diff_press_pis1();
    double k4_diff_press_pis2 = get_system() -> get_diff_press_pis2();
    double k4_diff_press_pos = get_system() -> get_diff_press_pos();
    double k4_diff_press_neg = get_system() -> get_diff_press_neg();

    curr_time_ = curr_time_ + time_step_;
    motor_angle_ = fmod(motor_angle_ + k1_motor_angular_velocity*time_step_/8  + 3*k2_motor_angular_velocity*time_step_/8 + 3*k3_motor_angular_velocity*time_step_/8 + k4_motor_angular_velocity*time_step_/8, 2*PI);
    press_pis1_ = press_pis1_ + k1_diff_press_pis1*time_step_/8 + 3*k2_diff_press_pis1*time_step_/8 + 3*k3_diff_press_pis1*time_step_/8 + k4_diff_press_pis1*time_step_/8;
    press_pis2_ = press_pis2_ + k1_diff_press_pis2*time_step_/8 + 3*k2_diff_press_pis2*time_step_/8 + 3*k3_diff_press_pis2*time_step_/8 + k4_diff_press_pis2*time_step_/8;
    press_pos_ = press_pos_ + k1_diff_press_pos*time_step_/8 + 3*k2_diff_press_pos*time_step_/8 + 3*k3_diff_press_pos*time_step_/8 + k4_diff_press_pos*time_step_/8;
    press_neg_ = press_neg_ + k1_diff_press_neg*time_step_/8 + 3*k2_diff_press_neg*time_step_/8 + 3*k3_diff_press_neg*time_step_/8 + k4_diff_press_neg*time_step_/8;

    if (is_log_) { 
        logger_ -> write(
            get_curr_time(), // curr_time
            get_press_pis1(), // press_pis1
            get_press_pis2(), // press_pis2
            get_press_pos(), // press_pos
            get_press_neg(), // press_neg
            ctrl_pos,
            ctrl_neg
        );
    }
    if (get_system() -> is_verbose()) { 
        show_state(); 
        std::cout << "        > Positive channel control : " << ctrl_pos << std::endl;
        std::cout << "        > Negative channel control : " << ctrl_neg << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSimulator::set_time_step(double time_step) { 
    time_step_ = time_step; 
    if (get_system() -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Time step: " << time_step_ << std::endl;
    }
}

void PneumaticSimulator::set_press(double press_pos, double press_neg) {
    press_pos_ = press_pos;
    press_neg_ = press_neg;
    if (get_system() -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Chamber pressure" << std::endl;
        std::cout << "        ----------" << std::endl;
        std::cout << "        > Positive channel chamber pressure : " << press_pos_ << std::endl;
        std::cout << "        > Negative channel chamber pressure : " << press_neg_ << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSimulator::set_piston_press(double press_pis1, double press_pis2) {
    press_pis1_ = press_pis1;
    press_pis2_ = press_pis2;
    if (get_system() -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic system C++) " << get_system() -> get_name() << " ==> Piston pressure" << std::endl;
        std::cout << "        ----------" << std::endl;
        std::cout << "        > Pressure inside of piston 1 : " << press_pis1_ << std::endl;
        std::cout << "        > Pressure inside of piston 2 : " << press_pis2_ << std::endl;
        std::cout << "        ----------" << std::endl;
    }
}

void PneumaticSimulator::show_state() {
    std::cout << "" << std::endl;
    std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> State" << std::endl;
    std::cout << "        ----------" << std::endl;
    std::cout << "        > Time : " << curr_time_ << std::endl;
    std::cout << "        > Pressure inside of piston 1 : " << press_pis1_ << std::endl;
    std::cout << "        > Pressure inside of piston 2 : " << press_pis2_ << std::endl;
    std::cout << "        > Positive channel chamber pressure : " << press_pos_ << std::endl;
    std::cout << "        > Negative channel chamber pressure : " << press_neg_ << std::endl;
    std::cout << "        ----------" << std::endl;
}

void PneumaticSimulator::set_logger(const char* file_path) {
    if (logger_ != nullptr) { delete logger_; }
    is_log_ = true;

    std::string str(file_path);
    logger_ = new PneumaticLogger(file_path);
    if (system_ -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Log: " << file_path << std::endl;
    }

    logger_ -> write_header();
}

void PneumaticSimulator::set_name(const char* name) {
    std::string str(name);
    system_ -> set_name(name);
    if (system_ -> is_verbose()) {
        std::cout << "" << std::endl;
        std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Name: " << get_system() -> get_name() << std::endl;
    }
}

void PneumaticSimulator::destroy_logger() {
    if (is_log_) {
        is_log_ = false;
        delete logger_; 
        if (get_system() -> is_verbose()) {
            std::cout << "" << std::endl;
            std::cout << "[ INFO] (Pneumatic simulator C++) " << get_system() -> get_name() << " ==> Log done" << std::endl;
        }
    }
}



