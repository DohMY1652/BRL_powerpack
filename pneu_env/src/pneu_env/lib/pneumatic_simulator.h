#include <iostream>
#include <fstream>

#define V1_ 0.75
#define V2_ 0.4
#define TS 0.0001
#define T_ 323.15
#define T_OUT 293.15
#define PI 3.14159265358979
#define R 0.287
#define K 1.4
#define ATM 101.325
#define X_DIM 6
#define OBS 3
#define ACT 2
#define M_W_RPM_ 3000
#define SC_R_ 2
#define SC_L_ 7
#define V_D_ 7
#define O_S_ 148*0.1/180
#define SV_D_ 1.6
#define SV_G_ 0.16

#define CPOS_ 0.07305250251836186
#define CNEG_ 0.08286618511922166
#define C1OUT_ 0.023304388618618336
#define C1IN_ 1.0150850183877944
#define C2OUT_ 0.023304388618618336
#define C2IN_ 1.0150850183877944
#define CLPOS_ 0
#define CLNEG_ 0

struct PneumaticCT
{
private:
    double M_W_RPM;
    double SC_R, SC_L;
    double V_D, V_S, V_DV, V_MAX_V;
    double O_S;
    double SV_D, SV_G;
    double T;
    double V1, V2;
    double CPOS, CNEG, C1OUT, C1IN, C2OUT, C2IN, CLPOS, CLNEG;
    double* dxdt;
    double* mass_flowrate;
    void slider_crank(double angle, double angular_velocity, double phase, double* piston);
    void volume(double piston_pos, double piston_vel, double* V_dVdt);
    double orifice(double P_inlet, double P_outlet, double Cd);
    double leakage(double P_inlet, double P_outlet, double Cd);
    double pressure(double P, double V, double dVdt, double dmdt);
    double solenoid_valve(double P_inlet, double P_outlet, double ratio, double Cd);
    double chamber(double dmdt, double V);

public:
    PneumaticCT();
    void set_volume(double volume1, double volume2);
    void set_discharge_coeff(
        double CdPOS,
        double CdNEG,
        double Cd1IN,
        double Cd1OUT,
        double Cd2IN,
        double Cd2OUT,
        double ClPOS,
        double ClNEG
    );
    double* model(double* x, double* u);
    double* get_mass_flowrate();
};

struct PneumaticSimulator
{
public:
    PneumaticSimulator();
    ~PneumaticSimulator() {}
    static PneumaticSimulator& get_instance();
    void set_init_env(double pos_press, double neg_press);
    void pneumaticDT(double* xk, double* uk, double Ts, double* xk1);
    double get_time();
    double* get_mass_flowrate();
    double* step(double* control, double time_step);
    void set_volume(double volume1, double volume2);
    void set_discharge_coeff(
        double CdPOS,
        double CdNEG,
        double Cd1IN,
        double Cd1OUT,
        double Cd2IN,
        double C2OUT,
        double ClPOS,
        double ClNEG
    );
    void time_reset();

private:
    PneumaticCT *pneumaticCT;
    double* xk0;
    double* k;
    double* observation;
    double* mass_flowrate;
    double* mass_flowrate_;
    std::string filename;
};
