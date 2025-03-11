import traceback

import gradio as gr
from modules import scripts

from lib_multidiffusion.tiled_diffusion import TiledDiffusion


opTiledDiffusion = TiledDiffusion().apply


class MultiDiffusionForForge(scripts.Script):
    sorting_priority = 16

    def title(self):
        return "MultiDiffusion Integrated"

    def show(self, is_img2img):
        # make this extension visible in both txt2img and img2img tab.
        return scripts.AlwaysVisible

    def ui(self, *args, **kwargs):
        with gr.Accordion(open=False, label=self.title()):
            enabled = gr.Checkbox(label='Enabled', value=False)
            method = gr.Radio(label='Method', choices=['MultiDiffusion', 'Mixture of Diffusers'],
                              value='MultiDiffusion')
            tile_width = gr.Slider(label='Tile Width', minimum=16, maximum=8192, step=16, value=768)
            tile_height = gr.Slider(label='Tile Height', minimum=16, maximum=8192, step=16, value=768)
            tile_overlap = gr.Slider(label='Tile Overlap', minimum=0, maximum=2048, step=32, value=64)
            tile_batch_size = gr.Slider(label='Tile Batch Size', minimum=1, maximum=8192, step=1, value=4)

        return enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size

    def process_before_every_sampling(self, p, *script_args, **kwargs):
        # This will be called before every sampling.
        # If you use highres fix, this will be called twice.

        try:
            enabled, method, tile_width, tile_height, tile_overlap, tile_batch_size = script_args
        except ValueError as e:
            print(f"[too many values to unpack (expected 6)] {script_args=}")
            traceback.print_tb(e.__traceback__)
            raise e

        if not enabled:
            return

        unet = p.sd_model.forge_objects.unet

        unet = opTiledDiffusion(unet, method, tile_width, tile_height, tile_overlap, tile_batch_size)[0]

        p.sd_model.forge_objects.unet = unet

        # Below codes will add some logs to the texts below the image outputs on UI.
        # The extra_generation_params does not influence results.
        p.extra_generation_params.update(dict(
            multidiffusion_enabled=enabled,
            multidiffusion_method=method,
            multidiffusion_tile_width=tile_width,
            multidiffusion_tile_height=tile_height,
            multidiffusion_tile_overlap=tile_overlap,
            multidiffusion_tile_batch_size=tile_batch_size,
        ))

        return
