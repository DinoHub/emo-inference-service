
from pydantic import BaseSettings, Field

class BaseConfig(BaseSettings):
    """Define any config here.

    See here for documentation:
    https://pydantic-docs.helpmanual.io/usage/settings/
    """
    # KNative assigns a $PORT environment variable to the container
    port: int = Field(default=8083, env="PORT",description="Gradio App Server Port")
    espnet_cfg_filepath: str = 'misc/espnet_configs/config.yaml'
    er_model_filepath: str = 'models/valid.ccc.ave_10best.pth'
    vad_model_filepath: str = 'models/model_vad2xy.pkl'
    
    example_dir: str = 'examples'

config = BaseConfig()