#ifndef PNEUMATICLOGGER_H
#define PNEUMATICLOGGER_H

#include <fstream>
#include <string>
#include <filesystem>

class PneumaticLogger {
public:
    PneumaticLogger();
    PneumaticLogger(const std::string& file_path);
    ~PneumaticLogger();
    void write_header();
    void write(
        double curr_time,
        double press_pis1, double press_pis2,
        double press_pos, double press_neg,
        double ctrl_pos, double ctrl_neg
    );

private:
    std::ofstream file_;
    void create_directory_(const std::string& file_name);
};


#endif