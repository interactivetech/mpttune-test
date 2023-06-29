class MPT7B8bitConfig:
    name = 'mpt-7b'
    # hf_config_name = "mosaicml/mpt-7b"
    hf_config_name = '/cstor/mendeza/hf_test/mpt-7b'
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 16
    groupsize = None
    max_seq_len = 2048
    attn_impl = 'torch'
    device_map = "auto"
class MPT30B8bitConfig:
    name = 'mpt-30b'
    # hf_config_name = "mosaicml/mpt-30b"
    hf_config_name = '/cstor/mendeza/hf_test/mpt-30b'
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 16
    groupsize = None
    max_seq_len = 2048
    attn_impl = 'torch'
    device_map = "auto"


class MPT70BConfig:
    '''
        model:
      name: mpt_causal_lm
      init_device: meta
      d_model: 8192
      n_heads: 64
      n_layers: 80
      expansion_ratio: 4
      max_seq_len: ${max_seq_len}
      vocab_size: 50368
      attn_config:
        attn_impl: triton
    '''
    name = 'mpt-70b'
    hf_config_name = None
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 16
    d_model=8192
    n_heads=64
    n_layers=80
    groupsize = None
    max_seq_len = 2048
    expansion_ratio=4
    attn_impl = 'torch'
    device_map = "auto"

class MPT7BInstruct8bitConfig:
    name = 'mpt-7b-instruct'
    hf_config_name = "mosaicml/mpt-7b-instruct"
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    attn_impl = 'torch'
    device_map = "auto"


class MPT7BChat8bitConfig:
    name = 'mpt-7b-chat'
    hf_config_name = "mosaicml/mpt-7b-chat"
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 8
    groupsize = None
    max_seq_len = 2048
    attn_impl = 'torch'
    device_map = "auto"


class MPT7BStorywriter8bitConfig:
    name = 'mpt-7b-storywriter'
    hf_config_name = "mosaicml/mpt-7b-storywriter"
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 8
    groupsize = None
    max_seq_len = 5 * 2048
    attn_impl = 'torch'
    device_map = "auto"


class MPT7BStorywriter4BitConfig:
    name = 'mpt-7b-storywriter-4bit'
    hf_config_name = "OccamRazor/mpt-7b-storywriter-4bit-128g"
    hf_tokenizer_config = "EleutherAI/gpt-neox-20b"
    bits = 4
    groupsize = 128
    max_seq_len = 5 * 2048
    attn_impl = 'torch'
    device_map = "auto"
