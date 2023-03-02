
import logging

import gradio as gr
from config import config, BaseConfig
from predict import inputs, outputs, predict

if __name__ == "__main__":
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s")
    config = BaseConfig()

    css = ".matplotlib.svelte-s9i01a img.svelte-s9i01a {width:var(--size-full) !important; max-height:var(--size-full) !important}"

    app = gr.Interface(
        predict,
        inputs=inputs,
        outputs=outputs,
        title="Emotion Classification Inference Service",
        description="Emotion Classification Inference Service for AI App Store",
        examples=config.example_dir, 
        css=css,
    )
    
    app.launch(
        server_name="0.0.0.0",
        server_port=config.port,
        enable_queue=True
    )
    
