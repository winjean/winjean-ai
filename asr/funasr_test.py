from modelscope.pipelines import pipeline
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr import AutoModel
from modelscope import snapshot_download
from modelscope.utils.constant import Tasks

# model_dir = "iic/SenseVoiceSmall"
model_dir = r'C:\Users\winjean\.cache\modelscope\hub\iic\SenseVoiceSmall'
model = AutoModel(
    model=model_dir,
    # trust_remote_code=True,
    # remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

# en
res = model.generate(
    input=r"../audio/output.mp3",
    cache={},
    language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,  #
    merge_length_s=15,
)
text = rich_transcription_postprocess(res[0]["text"])
print("=======", res)
print("=======", text)


# model_dir = snapshot_download('iic/SenseVoiceSmall')
#
# inference_pipeline = pipeline(
#     task=Tasks.auto_speech_recognition,
#     model=model_dir,
#     # model_revision="master",
#     # device="cuda:0",
# )
#
# res = inference_pipeline(r"../audio/output.mp3")
# print("=======", res)
# print("=======", rich_transcription_postprocess(res[0]["text"]))

