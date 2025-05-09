def register(options_templates, options_section, OptionInfo):
    options_templates.update(options_section((None, "Forge Hidden options"), {
        "forge_unet_storage_dtype": OptionInfo('Automatic (fp16 LoRA)'),
        "forge_inference_memory": OptionInfo(1024),
        "forge_async_loading": OptionInfo('Async'),
        "forge_pin_shared_memory": OptionInfo('CPU'),
        "forge_preset": OptionInfo('flux'),
        "forge_additional_modules": OptionInfo([]),
    }))
    options_templates.update(options_section(('ui_alternatives', "UI alternatives", "ui"), {
        "forge_canvas_plain": OptionInfo(False, "ForgeCanvas: use plain background").needs_reload_ui(),
        "forge_canvas_toolbar_always": OptionInfo(False, "ForgeCanvas: toolbar always visible").needs_reload_ui(),
    }))
