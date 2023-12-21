import os


def getConfig():
    if os.path.isdir("/kaggle/input/otto-preprocessed-jsonl"):
        config={
        "dataset":"/kaggle/input/otto-preprocessed-jsonl",
        "train_dir":"/kaggle/working",
        "batch_size":128,
        "lr":0.001,
        "max_len":50,
        "hidden_size":50,
        "num_blocks":2,
        "num_epochs":201,
        "num_heads":1,
        "dropout_rate":0.5,
        "device":"cpu",
        "inference_only":False,
        "shuffle":"sessionwise",
        "state_dict_path":"/kaggle/working/latest.pth",
        "stats_file":"/kaggle/input/otto-preprocessed-jsonl/stats.json",
        "num_batch":1,
        "num_batch_negatives": 127,
        "num_uniform_negatives": 16384,
        "reject_uniform_session_items": False,
        "reject_in_batch_items": True,
        "sampling_style": "batchwise",
    }
    else:
        config={
            "dataset":"data/otto/jsonl_processed",
            "train_dir":"output",
            "batch_size":128,
            "lr":0.001,
            "max_len":50,
            "hidden_size":50,
            "num_blocks":2,
            "num_epochs":201,
            "num_heads":1,
            "dropout_rate":0.5,
            "device":"cpu",
            "inference_only":False,
            "shuffle":True,
            "state_dict_path":"output/latest.pth",
            "stats_file":"data/otto/jsonl_processed/stats.json",
            "num_batch":1,
            "num_batch_negatives": 127,
            "num_uniform_negatives": 16384,
            "reject_uniform_session_items": False,
            "reject_in_batch_items": True,
            "sampling_style": "sessionwise",
        }
    assert 0 <= config["num_batch_negatives"] < config['batch_size']
    return config