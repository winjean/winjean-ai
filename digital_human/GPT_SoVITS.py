from subprocess import Popen
import os
import json
import sys
import yaml
from config import python_exec,infer_device,is_half,exp_root
from tools.i18n.i18n import I18nAuto, scan_language_list
from tools import my_utils
import traceback
import psutil
import signal
import platform
import glob
import re
import shutil

SoVITS_weight_root = ["SoVITS_weights_v2", "SoVITS_weights"]
GPT_weight_root=["GPT_weights_v2","GPT_weights"]
version = "v1"
now_dir = os.getcwd()
tmp = os.path.join(now_dir, "TEMP")
language=sys.argv[-1] if sys.argv[-1] in scan_language_list() else "Auto"
i18n = I18nAuto(language=language)
set_gpu_numbers=set()
default_gpu_numbers = "1"
ps1abc=[]
system=platform.system()


def open1abc(inp_text, inp_wav_dir, exp_name,
             gpu_numbers1a, gpu_numbers1Ba, gpu_numbers1c, bert_pretrained_dir, ssl_pretrained_dir, pretrained_s2G_path):
    global ps1abc
    inp_text = my_utils.clean_path(inp_text)
    inp_wav_dir = my_utils.clean_path(inp_wav_dir)
    check_for_exists([inp_text,inp_wav_dir])

    if (ps1abc == []):
        opt_dir="%s/%s"%(exp_root,exp_name)
        try:
            #############################1a
            path_text="%s/2-name2text.txt" % opt_dir
            if(os.path.exists(path_text)==False or (os.path.exists(path_text)==True and len(open(path_text,"r",encoding="utf8").read().strip("\n").split("\n"))<2)):
                config={
                    "inp_text":inp_text,
                    "inp_wav_dir":inp_wav_dir,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "bert_pretrained_dir":bert_pretrained_dir,
                    "is_half": str(is_half)
                }
                gpu_names=gpu_numbers1a.split("-")
                all_parts=len(gpu_names)
                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/1-get-text.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                for p in ps1abc:p.wait()

                opt = []
                for i_part in range(all_parts):#txt_path="%s/2-name2text-%s.txt"%(opt_dir,i_part)
                    txt_path = "%s/2-name2text-%s.txt" % (opt_dir, i_part)
                    print("txt_path:",txt_path)
                    with open(txt_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(txt_path)
                with open(path_text, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                assert len("".join(opt)) > 0, "1Aa-文本获取进程失败"
            print("进度：1a-done")

            ps1abc=[]
            #############################1b
            config={
                "inp_text":inp_text,
                "inp_wav_dir":inp_wav_dir,
                "exp_name":exp_name,
                "opt_dir":opt_dir,
                "cnhubert_base_dir":ssl_pretrained_dir,
            }
            gpu_names=gpu_numbers1Ba.split("-")
            all_parts=len(gpu_names)
            for i_part in range(all_parts):
                config.update(
                    {
                        "i_part": str(i_part),
                        "all_parts": str(all_parts),
                        "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                    }
                )
                os.environ.update(config)
                cmd = '"%s" GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py'%python_exec
                print(cmd)
                p = Popen(cmd, shell=True)
                ps1abc.append(p)
            for p in ps1abc:p.wait()
            print("进度：1a1b-done")

            ps1abc=[]
            #############################1c
            path_semantic = "%s/6-name2semantic.tsv" % opt_dir
            if(os.path.exists(path_semantic)==False or (os.path.exists(path_semantic)==True and os.path.getsize(path_semantic)<31)):
                config={
                    "inp_text":inp_text,
                    "exp_name":exp_name,
                    "opt_dir":opt_dir,
                    "pretrained_s2G":pretrained_s2G_path,
                    "s2config_path":"GPT_SoVITS/configs/s2.json",
                }
                gpu_names=gpu_numbers1c.split("-")
                all_parts=len(gpu_names)

                for i_part in range(all_parts):
                    config.update(
                        {
                            "i_part": str(i_part),
                            "all_parts": str(all_parts),
                            "_CUDA_VISIBLE_DEVICES": fix_gpu_number(gpu_names[i_part]),
                        }
                    )
                    os.environ.update(config)
                    cmd = '"%s" GPT_SoVITS/prepare_datasets/3-get-semantic.py'%python_exec
                    print(cmd)
                    p = Popen(cmd, shell=True)
                    ps1abc.append(p)
                print("进度：1a1b-done, 1cing")
                for p in ps1abc:p.wait()

                opt = ["item_name\tsemantic_audio"]

                print("all_parts:", all_parts)
                for i_part in range(all_parts):
                    semantic_path = "%s/6-name2semantic-%s.tsv" % (opt_dir, i_part)
                    with open(semantic_path, "r",encoding="utf8") as f:
                        opt += f.read().strip("\n").split("\n")
                    os.remove(semantic_path)

                with open(path_semantic, "w",encoding="utf8") as f:
                    f.write("\n".join(opt) + "\n")
                print("进度：all-done")
            ps1abc = []
            print("一键三连进程结束")
        except:
            traceback.print_exc()
            close1abc()
            print("一键三连中途报错")
    else:
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务")


def close1abc():
    global ps1abc
    if (ps1abc != []):
        for p1abc in ps1abc:
            try:
                kill_process(p1abc.pid)
            except:
                traceback.print_exc()
        ps1abc=[]
    return "已终止所有一键三连进程", {"__type__": "update", "visible": True}, {"__type__": "update", "visible": False}

p_train_SoVITS=None

def open1Ba(batch_size,total_epoch,exp_name,text_low_lr_rate,
            if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers1Ba, pretrained_s2G, pretrained_s2D):
    global p_train_SoVITS

    if(p_train_SoVITS==None):
        with open("GPT_SoVITS/configs/s2.json")as f:
            data=f.read()
            data=json.loads(data)
        s2_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s2"%(s2_dir),exist_ok=True)
        check_for_exists([s2_dir],is_train=True)
        if(is_half==False):
            data["train"]["fp16_run"]=False
            batch_size=max(1,batch_size//2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["train"]["text_low_lr_rate"]=text_low_lr_rate
        data["train"]["pretrained_s2G"]=pretrained_s2G
        data["train"]["pretrained_s2D"]=pretrained_s2D
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["save_every_epoch"]=save_every_epoch
        data["train"]["gpu_numbers"]=gpu_numbers1Ba
        data["model"]["version"]=version
        data["data"]["exp_dir"]=data["s2_ckpt_dir"]=s2_dir
        data["save_weight_dir"]=SoVITS_weight_root[-int(version[-1])+2]
        data["name"]=exp_name
        data["version"]=version
        tmp_config_path="%s/tmp_s2.json"%tmp
        with open(tmp_config_path,"w")as f:f.write(json.dumps(data))

        cmd = '"%s" GPT_SoVITS/s2_train.py --config "%s"'%(python_exec,tmp_config_path)
        print("SoVITS训练开始")
        print(cmd)
        p_train_SoVITS = Popen(cmd, shell=True)
        p_train_SoVITS.wait()
        p_train_SoVITS=None
        print("SoVITS训练完成")
    else:
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务")


p_train_GPT=None

def open1Bb(batch_size, total_epoch, exp_name,
            if_dpo, if_save_latest, if_save_every_weights, save_every_epoch, gpu_numbers, pretrained_s1):
    global p_train_GPT
    if(p_train_GPT==None):
        with open("GPT_SoVITS/configs/s1longer.yaml"if version=="v1"else "GPT_SoVITS/configs/s1longer-v2.yaml")as f:
            data=f.read()
            data=yaml.load(data, Loader=yaml.FullLoader)
        s1_dir="%s/%s"%(exp_root,exp_name)
        os.makedirs("%s/logs_s1"%(s1_dir),exist_ok=True)
        check_for_exists([s1_dir],is_train=True)
        if(is_half==False):
            data["train"]["precision"]="32"
            batch_size = max(1, batch_size // 2)
        data["train"]["batch_size"]=batch_size
        data["train"]["epochs"]=total_epoch
        data["pretrained_s1"]=pretrained_s1
        data["train"]["save_every_n_epoch"]=save_every_epoch
        data["train"]["if_save_every_weights"]=if_save_every_weights
        data["train"]["if_save_latest"]=if_save_latest
        data["train"]["if_dpo"]=if_dpo
        data["train"]["half_weights_save_dir"]=GPT_weight_root[-int(version[-1])+2]
        data["train"]["exp_name"]=exp_name
        data["train_semantic_path"]="%s/6-name2semantic.tsv"%s1_dir
        data["train_phoneme_path"]="%s/2-name2text.txt"%s1_dir
        data["output_dir"]="%s/logs_s1"%s1_dir
        # data["version"]=version

        os.environ["_CUDA_VISIBLE_DEVICES"]=fix_gpu_numbers(gpu_numbers.replace("-",","))
        os.environ["hz"]="25hz"
        tmp_config_path="%s/tmp_s1.yaml"%tmp
        with open(tmp_config_path, "w") as f:f.write(yaml.dump(data, default_flow_style=False))
        # cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" --train_semantic_path "%s/6-name2semantic.tsv" --train_phoneme_path "%s/2-name2text.txt" --output_dir "%s/logs_s1"'%(python_exec,tmp_config_path,s1_dir,s1_dir,s1_dir)
        cmd = '"%s" GPT_SoVITS/s1_train.py --config_file "%s" '%(python_exec,tmp_config_path)
        print("GPT训练开始")
        print(cmd)
        p_train_GPT = Popen(cmd, shell=True)
        p_train_GPT.wait()
        p_train_GPT=None
        print("GPT训练完成")
    else:
        print("已有正在进行的一键三连任务，需先终止才能开启下一次任务")


def check_for_exists(file_list=None,is_train=False,is_dataset_processing=False):
    missing_files=[]
    if is_train == True and file_list:
        file_list.append(os.path.join(file_list[0], '2-name2text.txt'))
        file_list.append(os.path.join(file_list[0], '3-bert'))
        file_list.append(os.path.join(file_list[0], '4-cnhubert'))
        file_list.append(os.path.join(file_list[0], '5-wav32k'))
        file_list.append(os.path.join(file_list[0], '6-name2semantic.tsv'))
    for file in file_list:
        if os.path.exists(file):pass
        else:missing_files.append(file)
    # if missing_files:
        # if is_train:
            # for missing_file in missing_files:
                # if missing_file != '':
                    # gr.Warning(missing_file)
            # gr.Warning(i18n('以下文件或文件夹不存在:'))
        # else:
            # for missing_file in missing_files:
                # if missing_file != '':
                    # gr.Warning(missing_file)
            # if file_list[-1]==[''] and is_dataset_processing:
            #     pass
            # else:
                # gr.Warning(i18n('以下文件或文件夹不存在:'))


def fix_gpu_numbers(inputs):
    output = []
    try:
        for input in inputs.split(","):
            output.append(str(fix_gpu_number(input)))
        return ",".join(output)
    except:
        return inputs


def fix_gpu_number(input):#将越界的number强制改到界内
    try:
        if(int(input)not in set_gpu_numbers):
            return default_gpu_numbers
    except:
        return input
    return input


def kill_proc_tree(pid, including_parent=True):
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        # Process already terminated
        return

    children = parent.children(recursive=True)
    for child in children:
        try:
            os.kill(child.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass
    if including_parent:
        try:
            os.kill(parent.pid, signal.SIGTERM)  # or signal.SIGKILL
        except OSError:
            pass


def kill_process(pid):
    if(system=="Windows"):
        cmd = "taskkill /t /f /pid %s" % pid
        os.system(cmd)
    else:
        kill_proc_tree(pid)


def get_model(inp_text, inp_wav_dir, model_name):
    print("开始训练模型")

    # 一键三连
    open1abc(inp_text, inp_wav_dir, model_name, "1", "1", "1",
             "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
             "GPT_SoVITS/pretrained_models/chinese-hubert-base",
             "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth")

    # SoVITS训练
    open1Ba(20, 8, model_name, 0.4, True, True,
            4, "1",
            "GPT_SoVITS/pretrained_models/s2G488k.pth",
            "GPT_SoVITS/pretrained_models/s2G488k.pth".replace("s2G", "s2D"))

    # GPT训练
    open1Bb(25, 15, model_name, True, True, True,
            5, "1",
            "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")


def preprocessor(model_name: str, json_array: str) -> tuple[str, str]:
    cwd = os.getcwd()
    txt_path = os.path.join(cwd, "model_export", model_name, "asr_opt")
    wav_path = os.path.join(cwd, "model_export", model_name, "denoise_opt")
    os.makedirs(txt_path, exist_ok=True)
    os.makedirs(wav_path, exist_ok=True)

    # output/denoise_opt/001.mp3_0005144320_0005333440.wav|denoise_opt|ZH|我的豆，谢谢朋友们。

    # 生成list文本文件
    # 生成音频文件夹
    json_list = json.loads(json_array)
    with open('example.txt', 'w') as file:
        for item in json_list:
            # TODO 接收音频文件
            txt = item.get("txt")
            wav = item.get("wav")
            file.write(f"{wav}|denoise_opt|ZH|{txt}\n")

    return os.path.join(txt_path, "denoise_opt.list"), wav_path


def post_processor(model_name: str) -> str:
    cwd = os.getcwd()
    model_export_path = os.path.join(cwd, "model_export", model_name)
    os.makedirs(model_export_path, exist_ok=True)

    # 复制ckpt文件
    copy_ckpt_file(model_export_path, model_name)

    # 复制pth文件
    copy_pth_file(model_export_path, model_name)

    # 复制audio.wav音频文件
    copy_audio_file(model_export_path, model_name)

    # 生产 infer_config.json
    build_json_file(model_export_path)

    return model_export_path


def copy_ckpt_file(model_export_path: str, model_name: str):
    ckpt_file_path = os.path.join(os.getcwd(), "GPT_weights")
    files = glob.glob(os.path.join(ckpt_file_path, f"{model_name}*"))

    # 提取每个文件名中的数字部分
    file_numbers = []
    for file in files:
        match = re.search(rf"{re.escape(model_name)}-e(\d+)\.ckpt$", file)
        if match:
            file_numbers.append((int(match.group(1)), file))

    # 找出数字最大的文件
    if file_numbers:
        max_file = max(file_numbers, key=lambda x: x[0])
        src_file = os.path.join(ckpt_file_path, max_file[1])
        dst_file = os.path.join(model_export_path, model_name + ".ckpt")
        print("复制文件:", src_file, "到", dst_file)
        shutil.copy2(src_file, dst_file)


def copy_pth_file(model_export_path: str, model_name: str):
    ckpt_file_path = os.path.join(os.getcwd(), "SoVITS_weights")
    files = glob.glob(os.path.join(ckpt_file_path, f"{model_name}*"))

    # 提取每个文件名中的数字部分
    file_numbers = []
    for file in files:
        match = re.search(rf"{model_name}_e(\d+)_s(\d+)\.pth$", file)
        if match:
            file_numbers.append((int(match.group(1)), int(match.group(2)), file))

    # 找出数字最大的文件
    if file_numbers:
        max_file = max(file_numbers, key=lambda x: x[0] * 100 + x[1])
        src_file = os.path.join(ckpt_file_path, max_file[2])
        dst_file = os.path.join(model_export_path, model_name + ".pth")
        print("复制文件:", src_file, "到", dst_file)
        shutil.copy2(src_file, dst_file)


def build_json_file(model_export_path: str, model_name: str):
    data = {
        "gpt_path": model_name + ".ckpt",
        "sovits_path": model_name + ".pth",
        "software_version": "1.1",
        "简介": "这是一个配置文件适用于https://github.com/X-T-E-R/TTS-for-GPT-soVITS，是一个简单好用的前后端项目",
        "emotion_list": {
            "default": {
                "ref_wav_path": "audio.wav",
                # TODO 待完善
                "prompt_text": "",
                "prompt_language": "多语种混合"
            }
        }
    }

    # 构建 JSON 文件路径
    json_file_path = os.path.join(model_export_path, "infer_config.json")

    # 将数据写入 JSON 文件
    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


def copy_audio_file(model_export_path: str, model_name: str):
    ...


if __name__ == '__main__':
    import time
    import uuid

    start_time = time.time()
    pretrain_model_name = uuid.uuid4().hex

    # 预处理
    preprocessor("c5d8c709f08f47c28c3c622d785b9c15")

    input_text = "/home/ubuntu/Gaoming/GPT-SoVITS/output/asr_opt/denoise_opt.list"
    input_wav_dir = "output/denoise_opt"
    get_model(input_text, input_wav_dir, pretrain_model_name)

    # 生产模型后的后置处理
    post_processor("c5d8c709f08f47c28c3c622d785b9c15")

    print("完成模型：", pretrain_model_name)
    print("耗时：", time.time() - start_time)

