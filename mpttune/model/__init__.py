from .mpt.config import (
    MPT7B8bitConfig,MPT30B8bitConfig, MPT7BChat8bitConfig, MPT7BInstruct8bitConfig, MPT7BStorywriter4BitConfig, MPT7BStorywriter8bitConfig)


MODEL_CONFIGS = {
    MPT7B8bitConfig.name: MPT7B8bitConfig,
    MPT7BChat8bitConfig.name: MPT7BChat8bitConfig,
    MPT7BInstruct8bitConfig.name: MPT7BInstruct8bitConfig,
    MPT7BStorywriter8bitConfig.name: MPT7BStorywriter8bitConfig,
    MPT7BStorywriter4BitConfig.name: MPT7BStorywriter4BitConfig,
    MPT30B8bitConfig.name: MPT30B8bitConfig,
}


def load_model(model_name: str, weights, half=False, backend='triton',inference=False):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Invalid model name: {model_name}")

    model_config = MODEL_CONFIGS[model_name]

    if model_name in MODEL_CONFIGS:
        from .mpt.model import load_model
        if inference == False:
            model, tokenizer = load_model(model_config, weights, half=half, backend=backend,inference=False)
        elif inference == True:
            model, tokenizer = load_model(model_config, weights, half=half, backend=backend,inference=True)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model.eval()
    return model, tokenizer
