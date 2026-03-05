"""Port assignments and default sampling parameters for each model."""

PORTS = {
    "momask": 8081,
    "mdm": 8082,
    "mld": 8083,
    "t2m_gpt": 8084,
}

DEFAULT_PARAMS = {
    "momask": {
        "cond_scale": 4.0,
        "temperature": 1.0,
        "topkr": 0.9,
        "time_steps": 18,
        "gumbel_sample": False,
    },
    "mdm": {
        "guidance_param": 2.5,
        "num_repetitions": 1,
        "cond_scale": 4.0,
    },
    "mld": {
        "guidance_scale": 7.5,
        "sample_mean": False,
    },
    "t2m_gpt": {
        "temperature": 1.0,
        "if_categorial": True,
    },
}

FPS = 20
