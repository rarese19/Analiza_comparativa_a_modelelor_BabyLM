import os
from glob import glob
import numpy as np
import pandas as pd
from devbench.comparison.stats_helper import *

TROG_DIR = "evaluation_data/devbench/evals/gram-trog/"
human_data_trog = pd.read_csv(TROG_DIR + "human.csv")

def compare_trog(model_data, human_data):
    opt_kl = get_opt_kl(human_data, model_data)    
    result_df = pd.DataFrame({
        'kl': opt_kl['objective'],
        'beta': opt_kl['solution'],
        'iterations': opt_kl['iterations']
    }, index=[0])
    return result_df

def get_scores(trog_file):
    model_name = os.path.splitext(os.path.basename(trog_file))[0]
    res = np.load(trog_file)
    res = pd.DataFrame(res.squeeze(),
                    columns = ["image1", "image2", "image3", "image4"])
    res['trial'] = np.arange(res.shape[0])+1
    res['correct'] = (res['image1'] > res['image2']) & \
                    (res['image1'] > res['image3']) & \
                    (res['image1'] > res['image4'])
    acc = res['correct'].mean()
    kls = compare_trog(res, human_data_trog)
    kls['model'] = os.path.splitext(os.path.basename(trog_file))[0].replace("trog_", "").replace("_epoch_256", "")
    kls['accuracy'] = acc
    other_res_trog = pd.concat([kls]).sort_values(["model"]).reset_index(drop=True)
    other_res_trog.to_csv(f"results/devbench/{model_name}/gram-trog_models.csv")
    return kls