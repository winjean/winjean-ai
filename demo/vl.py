from modelscope import (
    snapshot_download,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from auto_gptq import AutoGPTQForCausalLM

# model_dir = snapshot_download("E:/llm/qwen/Qwen-VL-Chat-Int4", revision='v1.0.0')

import torch
torch.manual_seed(1234)

model_dir = "E:/llm/qwen/Qwen-VL-Chat-Int4"

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# use cuda device
# model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True,use_safetensors=True).eval()
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map={"cpu": True}, low_cpu_mem_usage=True, trust_remote_code=True).eval()

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
    {'text': '这是什么'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)
# 图中是一名年轻女子在沙滩上和她的狗玩耍，狗的品种可能是拉布拉多。她们坐在沙滩上，狗的前腿抬起来，似乎在和人类击掌。两人之间充满了信任和爱。

# 2nd dialogue turn
response, history = model.chat(tokenizer, '输出"狗"的检测框', history=history)
print(response)

image = tokenizer.draw_bbox_on_latest_picture(response, history)
if image:
    image.save('1.jpg')
else:
    print("no box")


if __name__ == '__main__':
    pass
