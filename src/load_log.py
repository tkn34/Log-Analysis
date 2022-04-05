
from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from multiprocessing import Pool


# Compiling for optimization
re_sub_1 = re.compile(r"(:(?=\s))|((?<=\s):)")
re_sub_2 = re.compile(r"(\d+\.)+\d+")
re_sub_3 = re.compile(r"\d{2}:\d{2}:\d{2}")
re_sub_4 = re.compile(r"Mar|Apr|Dec|Jan|Feb|Nov|Oct|May|Jun|Jul|Aug|Sep")
re_sub_5 = re.compile(r":?(\w+:)+")
re_sub_6 = re.compile(r"\.|\(|\)|\<|\>|\/|\-|\=|\[|\]")
p = re.compile(r"[^(A-Za-z)]")

def remove_parameters(msg):
    # Removing parameters with Regex
    msg = re.sub(re_sub_1, "", msg)
    msg = re.sub(re_sub_2, "", msg)
    msg = re.sub(re_sub_3, "", msg)
    msg = re.sub(re_sub_4, "", msg)
    msg = re.sub(re_sub_5, "", msg)
    msg = re.sub(re_sub_6, " ", msg)
    L = msg.split()
    # Filtering strings that have non-letter tokens
    new_msg = [k for k in L if not p.search(k)]
    msg = " ".join(new_msg)
    return msg


def judge_log(L):
    if "-" == L[0]:
        log = " ".join(L[2:])
        label = 0
        category = "normal"
    else:
        log = " ".join(L[2:])
        label = 1
        category = L[0]
    return log, label, category



def load_bgl(config):
    """
    BGLオープンデータセット
        正常ログ: ログの先頭が「-」となっているログ
        異常ログ: ログの先頭が「-」となっていないログ(先頭が異常の種類が記載されている)
    1. judge_log
        ログの正常/異常の判定。異常の場合は、異常の種類も判定する。
    """
    # load log
    data_list = []
    with open(config.log_path, 'r', encoding='latin-1') as IN:
        for i, line in enumerate(IN):
            # load log(line)
            L = line.strip().split()
            # 正常/異常の分類
            log, label, category = judge_log(L)
            # パラメータワードの削除(日時、IP、などの固有値を削除する)
            log_after = remove_parameters(log)
            
            
            data_list.append([log, log_after, label, category])
            
    df = pd.DataFrame(data_list, columns=["log", "log(after)", "label", "category"])
    return df



