from transformers import AutoModel, AutoTokenizer

model_name = "text-to-speech"
cache_dir = "E:/model/transformers/text-to-speech/"
mirror = "https://mirrors.tuna.tsinghua.edu.cn/hugging-face-models/"  # 清华大学镜像

tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir, mirror=mirror)
# tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)
model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir, mirror=mirror)

if __name__ == '__main__':
    pass