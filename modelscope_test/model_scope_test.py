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

from modelscope.models import Model
model = Model.from_pretrained('qwen/Qwen1.5-0.5B')

if __name__ == '__main__':
    pass
