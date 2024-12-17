#include <iostream>
#include "PneumaticLogger.h"

PneumaticLogger::PneumaticLogger() {
    std::string file_path = "log/log.csv";
    file_.open(file_path, std::ios::out | std::ios::app);
    if (!file_.is_open()) { throw std::runtime_error("[ERROR] Pneumatic logger C++ ==> Could not open file: " + file_path); }
}

PneumaticLogger::PneumaticLogger(const std::string& file_path) {
    file_.open(file_path, std::ios::out | std::ios::trunc);
    if (!file_.is_open()) { throw std::runtime_error("[ERROR] Pneumatic logger C++ ==> Could not open file: " + file_path); }
}

PneumaticLogger::~PneumaticLogger() { 
    if (file_.is_open()) { file_.close(); } 
}

void PneumaticLogger::write_header() {
    file_ << "curr_time,";
    file_ << "press_pis1,";
    file_ << "press_pis2,";
    file_ << "press_pos,";
    file_ << "press_neg,";
    file_ << "ctrl_pos,";
    file_ << "ctrl_neg";
    file_ << "\n";
}

void PneumaticLogger::write(
    double curr_time,
    double press_pis1, double press_pis2,
    double press_pos, double press_neg,
    double ctrl_pos, double ctrl_neg
) {
    file_ << curr_time; file_ << ",";
    file_ << press_pis1; file_ << ",";
    file_ << press_pis2; file_ << ",";
    file_ << press_pos; file_ << ",";
    file_ << press_neg; file_ << ",";
    file_ << ctrl_pos; file_ << ",";
    file_ << ctrl_neg; 
    file_ << "\n";
}

