from modelscope import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


model_dir = "e:/llm/qwen/Qwen-VL-Chat-Int4"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if not hasattr(tokenizer, 'model_dir'):
    tokenizer.model_dir = model_dir

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="cpu",
    trust_remote_code=True
).eval()
# Either a local path or an url between <img></img> tags.
image_path = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
response, history = model.chat(tokenizer, query=f'<img>{image_path}</img>这是什么', history=None)
print(response)

if __name__ == '__main__':
    pass
