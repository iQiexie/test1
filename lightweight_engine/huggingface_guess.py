"""
Упрощенная заглушка для huggingface_guess
"""

class ModelList:
    class Flux:
        pass
    
    class FluxSchnell:
        pass

model_list = ModelList()

class DiffusersConvert:
    @staticmethod
    def convert_vae_state_dict(state_dict):
        return state_dict

diffusers_convert = DiffusersConvert()

class GuessResult:
    def __init__(self):
        self.huggingface_repo = 'black-forest-labs/FLUX.1-dev'
        self.unet_config = {}
        self.supported_inference_dtypes = ['float16', 'bfloat16']
        self.unet_target = 'transformer'
        self.vae_target = 'vae'
        self.vae_key_prefix = ['first_stage_model.']
        self.text_encoder_key_prefix = ['cond_stage_model.']
        self.unet_key_prefix = ['model.diffusion_model.']
        self.clip_target = {}
        self.model_type = None
        self.ztsnr = False
    
    def process_vae_state_dict(self, sd):
        return sd
    
    def process_clip_state_dict(self, sd):
        return sd

def guess(state_dict):
    return GuessResult()