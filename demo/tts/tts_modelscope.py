from modelscope import AutoModel,AutoTokenizer
from modelscope.models import Model
from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
# from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 指定本地模型路径
model_path = "E:\\model\\text-to-speech\\iic\\speech_sambert-hifigan_tts_zh-cn_16k"
# model_path = "E:/model/text-to-speech"

# 加载模型配置
# model = AutoModel.from_pretrained(model_path)

# 加载分词器
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# tts = Model.from_pretrained(model_path)


# 创建 Pipeline 对象
tts = pipeline(
    task=Tasks.text_to_speech,
    model=model_path,
    # tokenizer=tokenizer
)

# 输入文本内容  angry
text = '<speak><emotion category="happy" intensity="2.0" >"文本转音频测试"</emotion></speak>'

#
output = tts(input=text,voice='zhitian_emo')

wav = output[OutputKeys.OUTPUT_WAV]

with open('test.wav','wb') as f:
    f.write(wav)


if __name__ == '__main__':
    pass