'''
Author: Maonan Wang
Date: 2025-01-13 18:57:22
LastEditTime: 2025-01-16 17:19:56
LastEditors: pangay 1623253042@qq.com
Description: Convert SUMO Net to 3D
FilePath: /VLM-TSC/sumonet_to_3d.py
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path

from tshub.tshub_env3d.vis3d_sumonet_convert.sumonet_to_tshub3d import SumoNet3D

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    netxml = path_convert("./TSCScenario/SumoNets/train_four_345/env/4phases.net.xml")

    sumonet_to_3d = SumoNet3D(net_file=netxml)
    sumonet_to_3d.to_glb(glb_dir=path_convert(f"./TSCScenario/3d_assets/"))