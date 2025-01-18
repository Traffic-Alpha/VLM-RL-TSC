'''
Author: pangay 1623253042@qq.com
Date: 2025-01-19 04:24:01
LastEditors: pangay 1623253042@qq.com
LastEditTime: 2025-01-19 04:41:34
FilePath: /VLM-RL-TSC/TSCAssistant/VLM-TSC.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import openai
from PIL import Image
import requests
from io import BytesIO

# 设置 OpenAI API 密钥
openai.api_key = "your_openai_api_key"

# 加载图片并提取描述
def get_image_description(image_path):
    with open(image_path, "rb") as img_file:
        response = openai.Image.create(
            file=img_file,
            model="dall-e-clip",  # 使用图像处理模型
            purpose="answers"
        )
    return response['data'][0]['text']

# 使用 GPT 分析图片描述
def analyze_description(description):
    response = openai.Completion.create(
        model="gpt-4",
        prompt=f"请分析以下图片内容：{description}",
        max_tokens=100,
        temperature=0.7
    )
    return response['choices'][0]['text'].strip()

# 图片路径
img_path = "path_to_your_image.jpg"

# 获取图片描述并分析
image_description = get_image_description(img_path)
print("图片描述：", image_description)

analysis = analyze_description(image_description)
print("分析结果：", analysis)
