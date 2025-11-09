import os
import time
from huggingface_hub import snapshot_download
from datasets import load_dataset
import datasets
from tqdm import tqdm
import json

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

repo_id = "y-ren16/OVSpeech"
local_dir = "./dataset/OVSpeech"

repo_id = "Insects/ContextSpeech"
local_dir = "./dataset/ContextSpeech"

repo_type = "dataset"

def download_with_retry(retries=500):
    for attempt in range(retries):
        try:
            snapshot_download(
                repo_id=repo_id,
                resume_download=True,
                local_dir=local_dir,
                repo_type=repo_type,
                local_dir_use_symlinks=False,
            )
            print("下载成功")
            break
        except Exception as e:
            print(f"下载失败，正在重试 ({attempt + 1}/{retries})... 错误信息：{e}")
            time.sleep(10)  
    else:
        print("多次重试后仍然失败，请检查网络连接或联系支持。")

download_with_retry()
