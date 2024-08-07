import os
import torch
import librosa
import hashlib

import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

from huggingface_hub import snapshot_download, hf_hub_download
from transformers import WhisperProcessor, WhisperModel, AutoTokenizer, set_seed
from .llasm import LlaaaLlamaForCausalLM
from .infer_tokenize import tokenize

# export HF_ENDPOINT=https://hf-mirror.com
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DEFAULT_AUDIO_PATCH_TOKEN = "<au_patch>"
DEFAULT_AUDIO_START_TOKEN = "<au_start>"
DEFAULT_AUDIO_END_TOKEN = "<au_end>"


class LLaSM2ModelLoader:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "llasm_model": (
                [
                    "LinkSoul/LLaSM-Cllama2",
                    "LinkSoul/LLaSM-Baichuan"
                ],
                {
                    "default": "LinkSoul/LLaSM-Cllama2"
                }),
            "llasm_audio_tower": (
                [
                    "openai/whisper-large-v2"
                ],
                {
                    "default": "openai/whisper-large-v2"
                }),
        },
        }

    RETURN_TYPES = ("LLASM2MODEL",)
    RETURN_NAMES = ("llasm_model",)
    FUNCTION = "load_model"
    CATEGORY = "llasm"

    def load_model(self, llasm_model, llasm_audio_tower):
        device = mm.get_torch_device()

        model_name = llasm_model.rsplit('/', 1)[-1]
        model_dir = (os.path.join(folder_paths.models_dir, "LLM", model_name))
        if not os.path.exists(model_dir):
            print(f"Downloading {llasm_model}")
            snapshot_download(repo_id=llasm_model, local_dir=model_dir, local_dir_use_symlinks=False)
            # huggingface-cli download --resume-download --local-dir-use-symlinks False LinkSoul/LLaSM-Cllama2 --local-dir LLaSM-Cllama2

        audio_tower_name = llasm_audio_tower.rsplit('/', 1)[-1]
        audio_tower_dir = (os.path.join(folder_paths.models_dir, "LLM", audio_tower_name))
        if not os.path.exists(audio_tower_dir):
            print(f"Downloading {llasm_audio_tower}")
            snapshot_download(repo_id=llasm_audio_tower, local_dir=audio_tower_dir, local_dir_use_symlinks=False)
            # huggingface-cli download --resume-download --local-dir-use-symlinks False openai/whisper-large-v2 --local-dir whisper-large-v2

        # 0.load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        # step0-1: add special token <au_patch>/<au_start>/<au_end>
        tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)

        # 1.load model
        model = LlaaaLlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=False).to(
            device)

        # 2.load audio processor
        audio_processor = WhisperProcessor.from_pretrained(audio_tower_dir, torch_dtype=torch.float16)

        # 3.load audio tower
        audio_tower = WhisperModel.from_pretrained(audio_tower_dir, torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=False).to(device)

        # step3-1: update audio_tower config for setting special tokens
        audio_config = audio_tower.config
        audio_config.audio_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_AUDIO_PATCH_TOKEN])[0]
        audio_config.audio_start_token, audio_config.audio_end_token = tokenizer.convert_tokens_to_ids(
            [DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN])
        model.get_model().audio_tower[0] = audio_tower

        llasm_model = {
            "model": model,
            "tokenizer": tokenizer,
            "audio_processor": audio_processor,
            # "audio_tower": audio_tower,
            "llm_type": model_name
        }
        return (llasm_model,)


class LLaSM2Interface:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "llasm_model": ("LLASM2MODEL",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 1, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0}),
                "do_sample": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 1, "min": 1, "max": 2 ** 32 - 1}),
                "sampling_rate": ("INT", {"default": 16000, "min": 1, "max": 16000}),
                "audio_token_len": ("INT", {"default": 64, "min": 1, "max": 128})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = "llasm"

    def process(self,
                audio,
                llasm_model,
                prompt,
                keep_model_loaded=False,
                max_new_tokens=1024,
                temperature=0.2,
                do_sample=True,
                sampling_rate=16000,
                audio_token_len=64,
                seed=0):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        # import pdb;pdb.set_trace()
        model = llasm_model["model"]
        tokenizer = llasm_model["tokenizer"]
        audio_processor = llasm_model["audio_processor"]
        # audio_tower = llasm_model["audio_tower"]
        llm_type = llasm_model['llm_type']

        if seed:
            set_seed(seed)

        # 4.preprocessing input audio
        audio_feat = audio_processor(audio['waveform'], sampling_rate=sampling_rate, return_tensors="pt").input_features
        audio_feat = audio_feat.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float16)

        # 5.tokenize
        qs = DEFAULT_AUDIO_START_TOKEN + DEFAULT_AUDIO_PATCH_TOKEN * audio_token_len + DEFAULT_AUDIO_END_TOKEN
        input_qs = {
            "conversations": [{
                "from": "human",
                "value": qs,
            }, {
                "from": "gpt",
                "value": ""
            }]
        }
        input_ids = torch.tensor([tokenize(input_qs, tokenizer, llm_type)]).to(device)

        # 6.inference
        output_ids = model.generate(
            input_ids,
            audios=audio_feat,
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens)

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # import pdb;pdb.set_trace()
        outputs = outputs.strip()
        if outputs.endswith("</s>"):
            outputs = outputs[:-len("</s>")]
        outputs = outputs.strip()

        if not keep_model_loaded:
            print("Offloading model...")
            model.to(offload_device)
            mm.soft_empty_cache()

        return (outputs,)


class LLaSMLoadAudio:
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.ogg', '.flac', '.aiff', '.aif')

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir)
            if (os.path.isfile(os.path.join(input_dir, f))
                and f.endswith(LLaSMLoadAudio.SUPPORTED_FORMATS)
            )
        ]
        return {"required": {"audio": (sorted(files), {"audio_upload": False})}}

    CATEGORY = "llasm"
    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "load"

    def load(self, audio):
        #import pdb;pdb.set_trace()
        audio = folder_paths.get_annotated_filepath(audio)
        waveform, _ = librosa.load(audio, sr=16000)
        audio = {"waveform": waveform, "sample_rate": 16000}
        return (audio,)

    # @classmethod
    # def IS_CHANGED(s, audio):
    #     image_path = folder_paths.get_annotated_filepath(audio)
    #     m = hashlib.sha256()
    #     with open(image_path, 'rb') as f:
    #         m.update(f.read())
    #     return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, audio):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True


NODE_CLASS_MAPPINGS = {
    "LLaSM2ModelLoader": LLaSM2ModelLoader,
    "LLaSM2Interface": LLaSM2Interface,
    "LLaSMLoadAudio": LLaSMLoadAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaSM2ModelLoader": "LLaSM Model Loader",
    "LLaSM2Interface": "LLaSM Interface",
    "LLaSMLoadAudio": "LLaSM Load Audio",
}
