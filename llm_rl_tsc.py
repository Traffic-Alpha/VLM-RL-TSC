'''
@Author: WANG Maonan
@Author: PangAoyu
@Date: 2023-09-08 18:57:35
@Description: 使用训练好的 RL Agent 进行测试
LastEditTime: 2025-01-16 21:22:57
'''
import torch
from langchain_openai import ChatOpenAI

from loguru import logger
from tshub.utils.format_dict import dict_to_str
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from TSCEnvironment.tsc_env import TSCEnvironment
from TSCEnvironment.tsc_env_wrapper import TSCEnvWrapper
from TSCAssistant.tsc_assistant import TSCAgent


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from utils.readConfig import read_config
from utils.make_tsc_env import make_env

path_convert = get_abs_path(__file__)
set_logger(path_convert('./'))

if __name__ == '__main__':
    sumo_cfg = path_convert("./TSCScenario/SumoNets/train_four_345/env/train_four_345.sumocfg")
    # Init Chat
    config = read_config()
    openai_proxy = config['OPENAI_PROXY']
    openai_api_key = config['OPENAI_API_KEY']
    chat = ChatOpenAI(
        model=config['OPENAI_API_MODEL'], 
        temperature=0.0,
        openai_api_key=openai_api_key, 
        openai_proxy=openai_proxy
    )

    # #########
    # Init Env
    # #########
    trip_info = path_convert(f'./Result/LLM.tripinfo.xml')
    params = {
        'tls_id':'J1',
        'num_seconds':300,
        'sumo_cfg':sumo_cfg,
        'use_gui':True,
        'log_file':'./log_test/',
        'trip_info':trip_info,
    }
    env = SubprocVecEnv([make_env(env_index=f'{i}', **params) for i in range(1)])#获取env信息
    env = VecNormalize.load(load_path=path_convert('./models/last_vec_normalize.pkl'), venv=env)
    env.training = False # 测试的时候不要更新
    env.norm_reward = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = path_convert('./models/last_rl_model.zip')

    model = PPO.load(model_path, env=env, device=device)


    # 使用模型进行测试
    dones = False # 默认是 False
    sim_step = 0
    obs = env.reset()

    tsc_agent = TSCAgent(llm=chat, verbose=True)
    while not dones:

        action, _state = model.predict(obs, deterministic=True)
        if sim_step>4:
            action=tsc_agent.agent_run(sim_step=sim_step, action=action, obs=obs, infos=infos), #加入 obs wrapper
        obs, rewards, dones, infos = env.step(action)
        sim_step += 1
    print('***********rewards************',rewards)
    env.close()