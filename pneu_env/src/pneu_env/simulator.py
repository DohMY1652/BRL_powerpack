from ctypes import CDLL, POINTER, c_void_p, c_double, c_int, c_char_p

import numpy as np
from collections import deque

from typing import Callable, Tuple, Dict

from pneu_utils.utils import get_pkg_path, delete_lines

def get_lib(lib_name: str = "libPneumaticSimulator.so") -> CDLL:
    lib = CDLL(f"{get_pkg_path('pneu_env')}/src/pneu_env/cpp/lib/{lib_name}")
        
    lib.create_pneumatic_simulator.restype = POINTER(c_void_p)
    lib.simulate.argtypes = [POINTER(c_void_p), c_double, c_double, c_double] # simulation_time, action_pos, action_neg
    
    lib.reset_curr_time.argtypes = [POINTER(c_void_p)] # set curr_time = 0
    lib.set_curr_time.argtypes = [POINTER(c_void_p), c_double] # curr_time
    lib.set_time_step.argtypes = [POINTER(c_void_p), c_double] # time_step
    lib.set_press.argtypes = [POINTER(c_void_p), c_double, c_double] # press_pos, press_neg
    lib.set_piston_press.argtypes = [POINTER(c_void_p), c_double, c_double] # press_pis1, press_pis2
    lib.set_logger.argtypes = [POINTER(c_void_p), c_char_p]
    lib.set_name.argtypes = [POINTER(c_void_p), c_char_p]

    lib.get_curr_time.argtypes = [POINTER(c_void_p)]
    lib.get_curr_time.restype = c_double
    lib.get_press_pos.argtypes = [POINTER(c_void_p)]
    lib.get_press_pos.restype = c_double
    lib.get_press_neg.argtypes = [POINTER(c_void_p)]
    lib.get_press_neg.restype = c_double
    lib.get_press_pis1.argtypes = [POINTER(c_void_p)]
    lib.get_press_pis1.restype = c_double
    lib.get_press_pis2.argtypes = [POINTER(c_void_p)]
    lib.get_press_pis2.restype = c_double

    lib.set_chamber_volume.argtypes = [POINTER(c_void_p), c_double, c_double] # pos_chamber_volume, neg_chamber_volume
    lib.set_discharge_coefficients.argtypes = [
        POINTER(c_void_p), 
        c_double, c_double, # Cd_pump_in, Cd_pump_out
        c_double, c_double # Cd_valve_pos, Cd_valve_neg
    ]
    lib.verbose.argtypes = [POINTER(c_void_p)]
    lib.quiet.argtypes = [POINTER(c_void_p)]
    lib.clear_lines.argtypes = [POINTER(c_void_p), c_int]

    lib.show_state.argtypes = [POINTER(c_void_p)]
    lib.show_discharge_coefficients.argtypes = [POINTER(c_void_p)]
    lib.show_params.argtypes = [POINTER(c_void_p)]
    
    lib.destroy_logger.argtypes = [POINTER(c_void_p)]
    lib.destroy_simulator.argtypes = [POINTER(c_void_p)] 

    return lib

class PneuSim():
    def __init__(
        self,
        freq: float = 50,
        delay: float = 0,
        noise: Tuple[float] = (0, 0), # Gaussian noise std
        action_low: Tuple[float] = (0, 0),
        action_high: Tuple[float] = (1, 1),
        init_press_pos: float = 101.325,
        init_press_neg: float = 101.325,
        name: str = "simulator"
    ):
        self.freq = freq
        self.time_step = 1/freq
        self.set_action_map_func(action_low, action_high)
        self.name = name
        
        self.lib = get_lib()
        self.simulator = self.lib.create_pneumatic_simulator()
        self.lib.set_name(self.simulator, self.name.encode("utf-8"))
        
        self.set_press_pis_func(
            press_pis1_func = lambda press_pos, press_neg: 101.325,
            press_pis2_func = lambda press_pos, press_neg: 101.325,
        )
        self.set_pressure(
            press_pos = init_press_pos,
            press_neg = init_press_neg
        )
        self.set_process_action_func(
            pos_process_action_func = lambda x: x,
            neg_process_action_func = lambda x: x,
        )
        self.set_delay(delay)
        self.noise_mean, self.noise_std = noise

        self.info = dict(
            curr_obs = np.array([
                self.lib.get_curr_time(self.simulator),
                self.lib.get_press_pos(self.simulator),
                self.lib.get_press_neg(self.simulator)
            ], dtype=np.float64),
            curr_obs_without_noise = np.array([
                self.lib.get_curr_time(self.simulator),
                self.lib.get_press_pos(self.simulator),
                self.lib.get_press_neg(self.simulator)
            ], dtype=np.float64),
            action = np.array([1.0, 1.0], dtype=np.float64),
            processed_action = np.array([1.0, 1.0], dtype=np.float64)
        )
    
    def observe(
        self,
        action: np.ndarray,
        goal: np.ndarray = np.array([101.325, 101.325])
    ) -> np.ndarray:
        self.action_buffer.appendleft(action)

        curr_obs, simulation_info = self.simulate(
            self.time_step,
            self.action_buffer[-1]
        )

        return curr_obs, simulation_info

    def simulate(
        self,
        time_step: float,
        action: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        processed_action = self.process_action(action)
        self.lib.simulate(
            self.simulator,
            time_step,
            *processed_action
        )

        curr_time = self.lib.get_curr_time(self.simulator)
        press_pos = self.lib.get_press_pos(self.simulator)
        press_neg = self.lib.get_press_neg(self.simulator)
        
        curr_obs_without_noise = np.array([curr_time, press_pos, press_neg])
        curr_obs = curr_obs_without_noise + self.get_noise()

        self.info = dict(
            curr_obs = curr_obs,
            curr_obs_without_noise = curr_obs_without_noise,
            action = processed_action,
            action_without_processing = action
        )

        return curr_obs, self.info

    def set_delay(
        self,
        delay: float
    ) -> None:
        num = int(delay/self.time_step)
        self.action_buffer = deque(maxlen=num + 1)
    
    def get_noise(self) -> np.ndarray:
        noise = np.random.normal(self.noise_mean, self.noise_std, 2)
        return np.r_[0, noise]

    def set_pressure(
        self,
        press_pos: float,
        press_neg: float
    ) -> None:
        self.lib.set_press(
            self.simulator,
            press_pos,
            press_neg
        )
        self.lib.set_piston_press(
            self.simulator,
            self.get_press_pis1(press_pos, press_neg),
            self.get_press_pis2(press_pos, press_neg),
        )
    
    def set_action_map_func(
        self,
        action_low: Tuple[float],
        action_high: Tuple[float]
    ) -> Callable[[np.ndarray], np.ndarray]:
        action_low_np = np.array(action_low, dtype=np.float64)
        action_high_np = np.array(action_high, dtype=np.float64)
        action_scale = (action_high_np - action_low_np)/2
        action_bias = (action_high_np + action_low_np)/2
        self.map_action = lambda action: action_scale*np.clip(action, -1, 1) + action_bias
    
    def set_process_action_func(
        self,
        pos_process_action_func: Callable[[float], float],
        neg_process_action_func: Callable[[float], float]
    ) -> Callable[[np.ndarray], np.ndarray]:
        def process_action(action: np.ndarray):
            mapped_action = self.map_action(action)
            action_pos = pos_process_action_func(mapped_action[0])
            action_neg = neg_process_action_func(mapped_action[1])
            return np.array([action_pos, action_neg], dtype=np.float64)
        self.process_action = lambda action: process_action(action)
    
    def set_press_pis_func(
        self,
        press_pis1_func: Callable[[float], float],
        press_pis2_func: Callable[[float], float]
    ) -> None: 
        self.get_press_pis1 = press_pis1_func
        self.get_press_pis2 = press_pis2_func
    
    def set_discharge_coefficients(
        self,
        Cd_pump_in: float,
        Cd_pump_out: float,
        Cd_valve_pos: float,
        Cd_valve_neg: float
    ) -> None:
        self.lib.set_discharge_coefficients(
            self.simulator,
            Cd_pump_in, Cd_pump_out,
            Cd_valve_pos, Cd_valve_neg
        )
    
    def set_action_map_func(
        self,
        action_low: Tuple,
        action_high: Tuple
    ) -> Callable[[np.ndarray], np.ndarray]:
        action_low_np = np.array(action_low, dtype=np.float64)
        action_high_np = np.array(action_high, dtype=np.float64)
        action_scale = (action_high_np - action_low_np)/2
        action_bias = (action_high_np + action_low_np)/2
        self.map_action = lambda action: action_scale*np.clip(action, -1, 1) + action_bias
    
    def verbose_cpp_simulator(self) -> None:
        self.lib.verbose(self.simulator)
    
    def quiet_cpp_simulator(self) -> None:
        self.lib.quiet(self.simulator)
    
    def show_observation(self) -> int:
        line_num = super().show_observation(
            self.info["curr_obs"],
            self.info["action"]
        )
        addtional_lines = [
            f"\t----------",
            f"\tPress w/o noise      : ( {self.info['curr_obs_without_noise'][1]:8.4f}, {self.info['curr_obs_without_noise'][2]:8.4f} )",
            f"\tAction w/o processing: ( {self.info['action_without_processing'][0]:8.4f}, {self.info['action_without_processing'][1]:8.4f} )",
            f"\t----------"
        ]
        for line in addtional_lines:
            print(line)
        return line_num + len(addtional_lines)
    
    def show_cpp_simulator_state(self) -> None:
        self.lib.show_state(self.simulator)

    def show_cpp_simulator_params(self) -> None:
        self.lib.show_params(self.simulator)

    def show_observation(
        self,
        observation: np.ndarray,
        action: np.ndarray
    ) -> int:
        lines = [
            f"[ INFO] (Pneumatic simulator Python) {self.name} ==> Observation",
            f"\tTime  : {observation[0]:10.4f}",
            f"\tPress : ( {observation[1]:8.4f}, {observation[2]:8.4f} )",
            f"\tAction: ( {action[0]:8.4f}, {action[1]:8.4f} )"
        ]
        for line in lines:
            print(line)
        
        return len(lines)
    
if __name__ == "__main__":
    sim = PneuSim()
    for _ in range(1000):
        curr_obs, _ = sim.observe(
            np.array([0, 0])
        )
        line_len = sim.show_observation(curr_obs, np.array([0, 0]))
        delete_lines(line_len)
        
    