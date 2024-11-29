# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
#
# # 创建一个情感分析任务的pipeline
# nlp_pipeline = pipeline(task=Tasks.sentiment_analysis, model='damo/nlp_bert_sentiment_analysis')
#
# # 使用pipeline进行推理
# result = nlp_pipeline('I love using ModelScope!')
# print(result)


# from modelscope.msdatasets import MsDataset
# from modelscope.pipelines import pipeline
#
# inputs = ['今天天气不错，适合出去游玩', '这本书很好，建议你看看']
# dataset = MsDataset.load(inputs, target='sentence')
# word_segmentation = pipeline('word-segmentation')
# outputs = word_segmentation(dataset)
# for o in outputs:
#     print(o)


# from modelscope.models import Model
# from modelscope.pipelines import pipeline
# from modelscope.utils.constant import Tasks
# from PIL import Image
# import requests
#
# model = Model.from_pretrained('damo/multi-modal_gemm-vit-large-patch14_generative-multi-modal-embedding')
# p = pipeline(task=Tasks.generative_multi_modal_embedding, model=model)
#
# url = 'http://clip-multimodal.oss-cn-beijing.aliyuncs.com/lingchen/demo/dogs.jpg'
# image = Image.open(requests.get(url, stream=True).raw)
# text = 'dogs playing in the grass'
#
# img_embedding = p.forward({'image': image})['img_embedding']
# print('image embedding: {}'.format(img_embedding))
#
# text_embedding = p.forward({'text': text})['text_embedding']
# print('text embedding: {}'.format(text_embedding))
#
# image_caption = p.forward({'image': image, 'captioning': True})['caption']
# print('image caption: {}'.format(image_caption))

# from modelscope.models import Model
# model = Model.from_pretrained('qwen/Qwen1.5-0.5B')

from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['low_cpu_mem_usage'] = 'True'

model_name = "qwen/Qwen2.5-0.5B-Instruct"
local_model_path = r"E:\model\qwen"

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# model.save_pretrained(local_model_path)
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(local_model_path)

# 验证加载本地模型
loaded_model = AutoModelForCausalLM.from_pretrained(local_model_path)
loaded_tokenizer = AutoTokenizer.from_pretrained(local_model_path)

prompt = """
你是日志精灵，你专注于处理用户提供的日志数据，通过智能解析生成固定格式的解码器。
你的能力有:
    - 自动识别日志格式
    - 快速生成解码器
    - 支持多种日志类型解析
    
提供如下格式化输出:
decoder:
    parent: useradd
    name: useradd-newusr
    conditions:
        - regex:
            field: message
            pattern:'new user'
    processors:
        - regex:
            field: message
            offset: whole
            pattern: 'new user:\s+name=(\S+)\..*'
            targets:['host.user.name']
            
    请解析如下日志内容：
        www.pipixia.org---100.122.17.191 - - [06/Oct/2024:03:46:21 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white,green,blue,red&sort=_default HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=black%2Cwhite%2Cgreen%2Cblue%2Cred" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.130.32"
        www.pipixia.org---100.122.17.142 - - [06/Oct/2024:03:48:25 +0800] "GET /index.php/index/lists?catname=wc&lang=zh-cn&color=black,white HTTP/1.1" 200 7530 "http://www.pipixia.org/index.php/index/lists?catname=wc&lang=zh-cn&color=blue%2Cblack%2Cwhite" "Mozilla/5.0 (Linux; Android 7.0;) AppleWebKit/537.36 (KHTML, like Gecko) Mobile Safari/537.36 (compatible; PetalBot;+https://webmaster.petalsearch.com/site/petalbot)" "10.179.80.116, 114.119.151.174"            
"""
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = loaded_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = loaded_tokenizer([text], return_tensors="pt").to(loaded_model.device)

generated_ids = loaded_model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = loaded_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

if __name__ == '__main__':
    pass
