#include "pneumatic_simulator.h"

static PneumaticSimulator pneumatic_simulator;

PneumaticSimulator::PneumaticSimulator() 
{
    pneumaticCT = new PneumaticCT;
    
    xk0 = new double[X_DIM];
    xk0[0] = 0;
    xk0[1] = ATM;
    xk0[2] = ATM;
    xk0[3] = PI/2;
    xk0[4] = ATM;
    xk0[5] = ATM;

    k = new double[X_DIM];

    observation = new double[OBS];
    mass_flowrate = new double[7];
    mass_flowrate_ = new double[6];
    std::cout << "[ INFO] Pneumatic Simulator ==> Initialized" << std::endl;

    // filename = "log.csv";
    // std::ofstream outFile(filename);
    // outFile << "time, Ppos, Pneg, P1, P2, dPpos, dPneg, dP1, dP2" << std::endl;
    // outFile.close();
}

PneumaticSimulator& PneumaticSimulator::get_instance() { return pneumatic_simulator; }

void PneumaticSimulator::set_init_env(double pos_press, double neg_press)
{
    xk0[1] = pos_press;
    xk0[2] = neg_press;
    // xk0[4] = pos_press;
    // xk0[5] = neg_press;
    // xk0[4] = neg_press;
    // xk0[5] = neg_press;
    xk0[3] = 0;
    // xk0[4] = 0.086318*pos_press + 38.4313*neg_press + 7.0908;
    // xk0[5] = 0.046441*pos_press + 0.99813*neg_press -13.9509;
    xk0[4] = -0.0043396*pos_press + 38.543*neg_press + 16.4587;
    xk0[5] = -0.0067114*pos_press + 0.83468*neg_press + 4.5209;
    // xk0[5] = 6;
    		
    // std::cout << "[ INFO] Pneumatic Simulator ==> Env initialized: POS " << xk0[1] << " NEG " << xk0[2] << std::endl; 
}

void PneumaticSimulator::set_volume(double volume1, double volume2)
{
    pneumaticCT -> set_volume(volume1, volume2);
    std::cout << "[ INFO] Pneumatic Simulator ==> Vol initialized: POS " << volume1 << " NEG " << volume2 << std::endl; 
}

void PneumaticSimulator::set_discharge_coeff(
    double CdPOS,
    double CdNEG,
    double Cd1IN,
    double Cd1OUT,
    double Cd2IN,
    double Cd2OUT,
    double ClPOS,
    double ClNEG
)
{
    pneumaticCT -> set_discharge_coeff(
        CdPOS,
        CdNEG,
        Cd1IN,
        Cd1OUT,
        Cd2IN,
        Cd2OUT,
        ClPOS,
        ClNEG
    );
    std::cout << "[ INFO] Pneumatic Simulator ==> Discharge Coefficient Initialized" << std::endl;
    std::cout << "[ INFO] CPOS : " << CdPOS << " CNEG : " << CdNEG << std::endl;
    std::cout << "[ INFO] C1IN : " << Cd1IN << " C1OUT: " << Cd1OUT << std::endl;
    std::cout << "[ INFO] C2IN : " << Cd2IN << " C2OUT: " << Cd2OUT << std::endl;
    std::cout << "[ INFO] CLPOS: " << ClPOS << " CLNEG: " << ClNEG << std::endl;
}

void PneumaticSimulator::pneumaticDT(double* xk, double* uk, double Ts, double* xk1)
{
    double* k1 = new double[X_DIM];
    double* k2 = new double[X_DIM];
    double* k3 = new double[X_DIM];
    double* k4 = new double[X_DIM];

    double* i2 = new double[X_DIM];
    double* i3 = new double[X_DIM];
    double* i4 = new double[X_DIM];

    // Runge-Kutta Integrator
    k = pneumaticCT -> model(xk, uk);
    for (int i = 0; i < X_DIM; i++) k1[i] = k[i];
    for (int i = 0; i < X_DIM; i++)
        i2[i] = xk[i] + k1[i]*Ts/3; 

    k = pneumaticCT -> model(i2, uk);
    for (int i = 0; i < X_DIM; i++) k2[i] = k[i];
    for (int i = 0; i < X_DIM; i++)
        i3[i] = xk[i] - k1[i]*Ts/3 + k2[i]*Ts; 

    k = pneumaticCT -> model(i3, uk);
    for (int i = 0; i < X_DIM; i++) k3[i] = k[i];
    for (int i = 0; i < X_DIM; i++)
        i4[i] = xk[i] + k1[i]*Ts - k2[i]*Ts + k3[i]*Ts; 
    
    k = pneumaticCT -> model(i4, uk);
    for (int i = 0; i < X_DIM; i++) k4[i] = k[i];

    for (int i = 0; i < X_DIM; i++)
        xk1[i] = xk[i] + Ts*k1[i]/8 + Ts*3*k2[i]/8 + Ts*3*k3[i]/8 + Ts*k4[i]/8;

    xk1[3]= fmod(xk1[3], 2*PI);
    // std::cout << "[ INFO] Time: " << xk1[0] << std::endl;

    // std::ofstream outFile(filename, std::ios::app);
    // outFile << xk1[0] << ",";
    // outFile << xk1[1] << ",";
    // outFile << xk1[2] << ",";
    // outFile << xk1[4] << ",";
    // outFile << xk1[5] << ",";
    // outFile << k1[1] << ",";
    // outFile << k1[2] << ",";
    // outFile << k1[4] << ",";
    // outFile << k1[5] << std::endl;
    // outFile.close();

    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;

    delete[] i2;
    delete[] i3;
    delete[] i4;
}

double* PneumaticSimulator::step(double* control, double time_step)
{
    double* xk1 = new double[X_DIM];
    int n = time_step/TS;

    for (int i = 0; i < n; i++)
    {
        pneumaticDT(xk0, control, TS, xk1);
        for (int j = 0; j < X_DIM; j++) xk0[j] = xk1[j];
    }

    observation[0] = xk0[0];
    observation[1] = xk0[1];
    observation[2] = xk0[2];
    
    // std::cout << "[ INFO] T : " << xk0[0] << " P1: " << xk0[1] << " P2: " << xk0[2] << std::endl; 

    delete[] xk1;

    return observation;
}

double PneumaticSimulator::get_time() { return observation[0]; }

double* PneumaticSimulator::get_mass_flowrate() 
{   
    mass_flowrate_ = pneumaticCT -> get_mass_flowrate();
    mass_flowrate[0] = observation[0];
    mass_flowrate[1] = mass_flowrate_[0];
    mass_flowrate[2] = mass_flowrate_[1];
    mass_flowrate[3] = mass_flowrate_[2];
    mass_flowrate[4] = mass_flowrate_[3];
    mass_flowrate[5] = mass_flowrate_[4];
    mass_flowrate[6] = mass_flowrate_[5];

    return mass_flowrate; 
}

void PneumaticSimulator::time_reset() { 
    xk0[0] = 0; 
}

extern "C" {
    double get_time() {return PneumaticSimulator::get_instance().get_time();}
    double* step(double* control, double time_step) { return PneumaticSimulator::get_instance().step(control, time_step); }
    void set_init_env(double pos_press, double neg_press){ return PneumaticSimulator::get_instance().set_init_env(pos_press, neg_press); }
    void set_volume(double volume1, double volume2) { return PneumaticSimulator::get_instance().set_volume(volume1, volume2); }
    void set_discharge_coeff(
        double CdPOS, double CdNEG, 
        double Cd1IN, double Cd1OUT, 
        double Cd2IN, double C2OUT, 
        double ClPOS, double ClNEG
    )
    {
        return PneumaticSimulator::get_instance().set_discharge_coeff(
            CdPOS, CdNEG, Cd1IN, Cd1OUT, Cd2IN, C2OUT, ClPOS, ClNEG
        );
    }
    double* get_mass_flowrate() { return PneumaticSimulator::get_instance().get_mass_flowrate(); }
    void time_reset() {return PneumaticSimulator::get_instance().time_reset();}
}
