'''
@Author: Pang Aoyu
@Date: 2023-09-04 20:51:49
@Description: traffic light control LLM Agent
LastEditTime: 2025-01-19 05:52:45
'''
import numpy as np
from typing import List
from loguru import logger
from PIL import Image
import requests
from io import BytesIO
import os
from openai import OpenAI

from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationTokenBufferMemory
from  langchain.chains.conversation.base import ConversationChain
from langchain.prompts import ChatPromptTemplate   #导入聊天提示模板
from langchain.chains.llm import LLMChain   #导入LLM链。

from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
import openai
import requests
import base64
from utils.readConfig import read_config
#from google.cloud import vision
#from google.cloud.vision import types
# 设置 OpenAI API 密钥

from TSCAssistant.tsc_agent_prompt import SYSTEM_MESSAGE_SUFFIX
from TSCAssistant.tsc_agent_prompt import (
    SYSTEM_MESSAGE_SUFFIX,
    SYSTEM_MESSAGE_PREFIX,
    HUMAN_MESSAGE,
    FORMAT_INSTRUCTIONS,
    TRAFFIC_RULES,
    DECISION_CAUTIONS,
    HANDLE_PARSING_ERROR
)

class TSCAgent:
    def __init__(self, 
                 llm:ChatOpenAI, 
                 verbose:bool=True,state:float=[] ) -> None:
        self.tls_id='J1'
        self.llm = llm # ChatGPT Model
        self.tools = [] # agent 可以使用的 tools
        self.state= state
        config = read_config()
        self.api_key  = config['OPENAI_API_KEY']
        openai.api_key = self.api_key
        self.client = OpenAI(  api_key = self.api_key,  # This is the default and can be omitted 
            )
        #self.file_callback = create_file_callback(path_convert('../agent.log'))
        self.first_prompt=ChatPromptTemplate.from_template(   
                    'You can ONLY use one of the following actions: \n action:0 action:1 action:2 action:3'
    
                    )
        self.second_prompt=ChatPromptTemplate.from_template(   
                    " The action is {Action}, Your explanation was `{Occupancy}` \n To check decision safety: "
                    )
        self.memory = ConversationTokenBufferMemory(llm=self.llm, max_token_limit=2048)
        
        self.chain_one = LLMChain(llm=llm, prompt=self.first_prompt)
        
        self.chain_two = LLMChain(llm=llm, prompt=self.second_prompt,output_key="safety")
        '''
        self.assessment =  SequentialChain(chains=[self.chain_one, self.chain_two],
                                      input_variables=["Action"],
                                      #output_variables=["sfaty"],
                                      verbose=True) #构建路由链 还是构建顺序链， 需要构建提示模板
        '''
        memory = ConversationBufferMemory()
        self.assessment = ConversationChain( llm=llm, memory = memory, verbose=True )
        self.phase2movements={
                        "Phase 0": ["E0_s","-E1_s"],
                        "Phase 1": ["E0_l","-E1_l"],
                        "Phase 2": ["-E3_s","-E2_s"],
                        "Phase 3": ["-E3_l","-E2_l"],
                    } 
        self.movement_ids=["E0_s","-E1_s","-E1_l","E0_l","-E3_s","-E2_s","-E3_l","-E2_l"]       
    def image_to_base64(self,image_path):
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
        return img_b64            
    def get_image_description(self,image_path: str) -> str:
        """使用 Google Vision API 获取图像描述"""
        with open(image_path, 'rb') as image_file:
            content = image_file.read()

        #image = types.Image(content=content)
        response = self.client.label_detection(image=image)
        labels = response.label_annotations
        descriptions = [label.description for label in labels]
        return ", ".join(descriptions)
    
    def analyze_image_with_gpt(self, prompt, img_url):

        api_key = self.api_key
        model = "gpt-4o"  # 你可以使用正确的模型名称
        url = "https://api.openai.com/v1/chat/completions"
        img_b64 = self.image_to_base64(img_url)
        response = self.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{'png'};base64,{img_b64}"},
                    },
                ],
            }
        ],
    )
        img_data = f"data:image/png;base64,{img_b64}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # 构建请求的数据体
        data = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": img_data  # 传递图像的 URL
                            }
                        }
                    ]
                },
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt  # 传递文本描述
                        }
                    ]
                }
            ]
        }

        # 发送请求到 OpenAI API
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            # 如果请求成功，返回 GPT 的响应内容
            response_data = response.json()
            return response_data['choices'][0]['message']['content']
        
        else:
            # 如果请求失败，返回错误信息
            return f"Error: {response.status_code}, {response.text}"
    
    def analyze_description(self, description: str) -> str:
        """
        使用 GPT 分析图像描述并返回分析结果
        """
        response = openai.Completion.create(
            model="gpt-4",
            prompt=f"请分析以下图片内容：{description}",
            max_tokens=100,
            temperature=0.7
        )
        return response['choices'][0]['text'].strip()


    def agent_run(self, sim_step:float, action:int=0, obs:float=[], infos: list={}):
        """_summary_

        Args:
            tls_id (str): _description_
            sim_step (float): _description_

        1. 现在的每个action的 movement 
        2. 这个phase 所包含的movement 的平均占有率
        3. 判断动作是否可行
        """
        logger.info(f"SIM: Decision at step {sim_step} is running:")
        #occupancy = self.get_occupancy(obs)
        Action = action
        step_time = sim_step
        # step_time=int(step_time)
        #Occupancy=infos[0]['movement_occ']
        Occupancy = 0
        #jam_length_meters=infos[0]['jam_length_meters']
        jam_length_meters = 0
        #movement_ids=infos[0]['movement_ids']
        movement_ids = 0
        #last_step_vehicle_id_list=infos[0]['last_step_vehicle_id_list']
        last_step_vehicle_id_list = 0 
        #information_missing = infos[0]['information_missing']
        information_missing = 0
        #missing_id = infos[0]['missing_id']
        missing_id = 0
        #rescue_movement_ids=self.get_rescue_movement_ids(last_step_vehicle_id_list,movement_ids)
        rescue_movement_ids = 0
        # 要进行处理 如个存在缺失数值
        image_path = './sensor_images/sensor_image_0.png'
        prompt = 'Please Help Me analysize this picture.'
        description = self.analyze_image_with_gpt(prompt, image_path)
        print('description',description)
        review_template="""
        decision:  Traffic light decision-making judgment  whether the Action is reasonable in the current state.
        explanations: Your explanation about your decision, described your suggestions to the Crossing Guard. The analysis should be as detailed as possible, including the possible benefits of each action.
        final_action: ONLY the number of Action you suggestion, 0, 1, 2 or 3
        
        Format the output as JSON with the following keys:  
        decision
        expalanations
        final_action


        observation: {observation}
        {format_instructions}
        """
        prompt = ChatPromptTemplate.from_template(template=review_template)
        decision = ResponseSchema(name="decision",
                             description="Judgment whether the RL Agent's Action is reasonable in the current state, you can only choose reasonable or unreasonable. ")
        expalanations = ResponseSchema(name="expalanations",
                             description="Your explaination about your decision, described your suggestions to the Crossing Guard. The analysis should be as detailed as possible, including the possible benefits of your action.")        
        final_action = ResponseSchema(name="final_action",
                             description="ONLY the number of Action you final get, 0, 1, 2 or 3")
        response_schemas = [decision, 
                    expalanations,
                    final_action]
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = output_parser.get_format_instructions()
        #prompt_template = ChatPromptTemplate.from_template(ans_template)
        #messages = prompt_template.format_messages(text=observation)
        observation=(f"""
            You, the 'traffic signal light', are now controlling the traffic signal in the junction with ID `{self.tls_id}`.
            The step time is:"{step_time}"
            The decision RL Agent make thie step is `Action：{Action}`. 
            The vehicles mean occupancy of each movement is:`{Occupancy}`. 
            The number of cars waiting in each movement is：`{jam_length_meters}`. 
            Now these movements exist emergency vehicles： `{rescue_movement_ids}`. 
            Phase to Movement: '{self.phase2movements}'
            Is there a information loss now： '{information_missing}'
            The loss ID is：'{missing_id}'
            Please make decision for the traffic signal light.You have to work with the **Static State** and **Action** of the 'traffic light'. 
            Then you need to analyze whether the current 'Action' is reasonable based on the intersection occupancy rate, and finally output your decision.
            There are the actions that will occur and their corresponding phases：
        
                - Action 0： Phase 0
                - Action 1： Phase 1
                - Action 2： Phase 2
                - Action 3： Phase 3

            Here are your attentions points:
            {DECISION_CAUTIONS}
            
            Let's take a deep breath and think step by step. Once you made a final decision, output it in the following format: \n 
             
            """)
        messages = prompt.format_messages(observation=observation, format_instructions=format_instructions)
        print(messages[0].content)
        logger.info('RL:'+messages[0].content) #加入RL解析 输入到日志文件
        r = self.llm(messages)
        output_dict = output_parser.parse(r.content)
        print(r.content)
        logger.info('RL:'+r.content)
        final_action=output_dict.get('final_action')
        logger.info('RL:'+final_action)
        print('-'*10)
        final_action=int(final_action)
        return final_action