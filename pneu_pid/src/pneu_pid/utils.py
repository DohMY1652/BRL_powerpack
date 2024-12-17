import numpy as np

from typing import Tuple, Dict, Any

import json

from pneu_pid.dob import DisturbanceObserver
from pneu_pid.tf import TransFunc
from pneu_ref.ctrl_ref import CtrlTraj
from pneu_utils.utils import get_pkg_path

def get_dobs(freq: float) -> Tuple[DisturbanceObserver, DisturbanceObserver]:
    pos_dob = make_dob("pos", freq)
    neg_dob = make_dob("neg", freq)
    return pos_dob, neg_dob

def make_dob(field: str, freq: float) -> DisturbanceObserver:
    tf_datas = load_json_data(field)
    invGxQ, Q = set_transfer_functions(tf_datas, freq, field) 
    dob = DisturbanceObserver(invGxQ, Q)
    return dob

def load_json_data(field: str) -> Dict[str, Any]:
    # load datas from the json file
    json_file_name = f"{get_pkg_path('pneu_pid')}/src/pneu_pid/tf_{field}.json"
    with open(json_file_name, "r") as f:
        tf_datas = json.load(f)
    
    return tf_datas

def set_transfer_functions(
    tf_datas: Dict[str, Any],
    freq: float,
    field: str
) -> Tuple[TransFunc, TransFunc]:
    # Define base pre-processing function and post-processing function
    def base_pre_processing_function(
        x: np.ndarray, 
        x_min: np.ndarray,
        x_max: np.ndarray
    ) -> np.ndarray:
        return (x - x_min)/(x_max - x_min)
    
    def base_post_processing_function(
        x: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray
    ) -> np.ndarray:
        return x*(x_max - x_min) + x_min
    
    u_min = tf_datas["input_min"]
    u_max = tf_datas["input_max"]
    y_min = tf_datas["output_min"]
    y_max = tf_datas["output_max"]
    if field == "pos":
        invGxQ_pre_process_function = lambda y: base_pre_processing_function(y, y_min, y_max)
        invGxQ_post_process_function = lambda u: base_post_processing_function(1 - u, u_min, u_max)
        Q_pre_process_function = lambda u: 1- base_pre_processing_function(u, u_min, u_max)
        Q_post_process_function = lambda u: base_post_processing_function(1 - u, u_min, u_max)
    else:
        invGxQ_pre_process_function = lambda y: base_pre_processing_function(y, y_min, y_max)
        invGxQ_post_process_function = lambda u: base_post_processing_function(u, u_min, u_max)
        Q_pre_process_function = lambda u: base_pre_processing_function(u, u_min, u_max)
        Q_post_process_function = lambda u: base_post_processing_function(u, u_min, u_max)
    
    # Define invGxQ transfer function
    invGxQ = TransFunc(
        num = tf_datas["normalized_invGxQ_num"][0],
        den = tf_datas["normalized_invGxQ_den"][0],
        freq = freq
    )
    invGxQ.set_pre_process(func=invGxQ_pre_process_function)
    invGxQ.set_post_process(func=invGxQ_post_process_function)

    # Define Q transfer function
    Q = TransFunc(
        num = tf_datas["Q_num"][0],
        den = tf_datas["Q_den"][0],
        freq = freq
    )
    Q.set_pre_process(func=Q_pre_process_function)
    Q.set_post_process(func=Q_post_process_function)

    return invGxQ, Q