import os
from tqdm import tqdm 
import json
import argparse
import pandas as pd

data_dir = "./dataset/OVSpeech"
df = pd.read_parquet(os.path.join(data_dir, "OVSpeech_test.parquet"))
jsonl_data_path = os.path.join(data_dir, "OVSpeech_test.jsonl")
df.to_json(jsonl_data_path, orient='records', lines=True)
output_dir = os.path.join(data_dir, "OVSpeech_test_message.jsonl")

if os.path.exists(output_dir):
    with open(output_dir, 'r') as f:
        test_meta_data = [json.loads(line.strip()) for line in f.readlines()]
    print(f"Loaded from {output_dir}")
else:
    print("Preparing test set with messages...")
    with open(jsonl_data_path, 'r') as f:
        test_meta_data = [json.loads(line.strip()) for line in f.readlines()]
    for item in tqdm(test_meta_data):
        transcription = item['transcription']
        open_vocabulary_instruct = item['open_vocabulary_instruct']
        audio_path = item['audio_path']
        item['ov_thinking_para_tts_messages'] = [
            {"role": "system", "content": "你是一个专业的配音演员，能根据要求来配音。要求合成语音具有高表现力，并且在情感，副语言等方面符合描述性的要求。"},
            {"role": "human", "content": [{"type": "text", "text": f"{open_vocabulary_instruct}“{transcription}”请先根据描述推理语音的情感，声学描述信息和副语言标签。"}]},
            {"role": "assistant", "content": None},
            {"role": "human", "content": [{"type": "text", "text": f"请根据刚刚推理得到的情感、声学信息和副语言标签合成语音：“{transcription}”"}]},
            {"role": "assistant", "content": "<tts_start>", "eot": False}, 
        ]
    with open(output_dir, 'w') as f:
        for item in test_meta_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to {output_dir}")

from stepaudio2 import StepAudio2Base, StepAudio2
from token2wav import Token2wav

def ov_thinking_para_tts_test(model, token2wav, messages, prompt_wav=None, audio_save_path=None, text_save_path=None):
    if os.path.exists(audio_save_path) and os.path.exists(text_save_path) and os.path.exists(audio_save_path.replace('.wav', '_emotion.txt')) and os.path.exists(audio_save_path.replace('.wav', '_acoustic_description.txt')) and os.path.exists(audio_save_path.replace('.wav', '_paralinguistic_tags.txt')):
        print(f"already exist, skip.")
        return
    messages_thinking = messages[:3]
    text_list = []
    assert prompt_wav is not None, "Please provide prompt_wav"
    assert audio_save_path is not None, "Please provide audio_save_path"
    tokens, text, _ = model(messages_thinking, max_new_tokens=1024, temperature=0.5, do_sample=True)
    try:
        text_json = json.loads(text.split('```')[1].strip().split('\n')[-1])
        emotion = text_json.get('emotion', '')
        acoustic_description = text_json.get('acoustic_description', '')
        para_tags = text_json.get('paralinguistic_tags', '')
        if emotion:
            emotion_save_path = audio_save_path.replace('.wav', '_emotion.txt')
            with open(emotion_save_path, 'w') as f:
                f.write(emotion)
        if acoustic_description:
            acoustic_description_save_path = audio_save_path.replace('.wav', '_acoustic_description.txt')
            with open(acoustic_description_save_path, 'w') as f:
                f.write(acoustic_description)
        if para_tags:
            para_tags_save_path = audio_save_path.replace('.wav', '_paralinguistic_tags.txt')
            with open(para_tags_save_path, 'w') as f:
                f.write(para_tags)
    except:
        print(f"Failed to parse thinking output: {text}")
        return
    text_list.append(text)
    messages[2]['content'] = text
    tokens, text, audio = model(messages, max_new_tokens=2048, temperature=0.7, do_sample=True)
    text_list.append(text)
    audio = [x for x in audio if x < 6561] # remove audio padding
    audio = token2wav(audio, prompt_wav=prompt_wav)
    with open(audio_save_path, 'wb') as f:
        f.write(audio)
    if text_save_path is not None:
        with open(text_save_path, 'w') as f:
            f.write('\n'.join(text_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=8, help='8: OV-TTS-Thinking-Para-Emotion')
    args = parser.parse_args()
    exp = args.exp
    token2wav = Token2wav('./checkpoints/OV-InstructTTS/token2wav')
    if exp == 8:
        save_dir8 = "./Output"
        os.makedirs(save_dir8, exist_ok=True)
        model_8 = StepAudio2('./checkpoints/OV-InstructTTS')
        for item in tqdm(test_meta_data):
            key = item['key']
            audio_path = item['audio_path'].replace("Datasets/ContextSpeech", "./dataset/ContextSpeech")
            # audio_path = "./assets/default_male.wav"
            prompt_wav = audio_path
            ov_thinking_para_tts_messages = item['ov_thinking_para_tts_messages'] 
            ov_thinking_tts_audio_save_path = os.path.join(save_dir8, f"{key}.wav")
            ov_thinking_tts_text_save_path = os.path.join(save_dir8, f"{key}.txt")
            ov_thinking_para_tts_test(model_8, token2wav, ov_thinking_para_tts_messages, prompt_wav=prompt_wav, audio_save_path=ov_thinking_tts_audio_save_path, text_save_path=ov_thinking_tts_text_save_path)