# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:08:07 2022

@author: 81901
"""
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


recid_regx = re.compile(r"^(\d+)")
separator = re.compile(r"(?:-.{1,3}){2} (.+)$")
msg_split_regx = re.compile(r"x'.+'")
severity = re.compile(r"(\w+)\s+(INFO|WARN|ERROR|FATAL)")

def process_line(line):
    line = line.strip()
    sep = separator.search(line)
    print(sep)
    if sep:
        print("==============================")
        print(line)
        print()
        print(sep)
        print()
        msg = sep.group(1).strip().split('   ')[-1].strip()
        print(msg)
        print()
        msg = msg_split_regx.split(msg)[-1].strip()
        print(msg)
        print()
        error_label = severity.search(line)
        print(error_label)
        print()
        recid = recid_regx.search(line)
        print(recid)
        print()
        if recid and error_label and len(msg) > 20:
            # recid = recid.group(1).strip() We may want to use it later
            general_label = error_label.group(2)
            label = error_label.group(1)
            if general_label == 'WARN':
                return ''
            if general_label == 'INFO':  # or label == 'WARN':
                label = 'unlabeled'
            msg = remove_parameters(msg)
            if msg:
                msg = ' '.join((label, msg))
                msg = ''.join((msg, '\n'))
                return msg
    
    return ''




def data_preprocessing(log_path, output_path):
    with open(output_path, "w", encoding='latin-1') as f:
        with open(log_path, 'r', encoding='latin-1') as IN:
            line_count = sum(1 for line in IN)
        with open(log_path, 'r', encoding='latin-1') as IN:
            for line in tqdm(IN, total=line_count):
                msg = process_line(line)
                f.write(msg)           

            
            
            
def load(log_path, ignore_unlabeled=False):
    unlabel_label = "unlabeled"
    x_data = []
    y_data = []
    label_dict = {}
    target_names = []
    with open(log_path, 'r', encoding='latin-1') as IN:
        line_count = sum(1 for line in IN)
    with open(log_path, 'r', encoding='latin-1') as IN:
        for line in tqdm(IN, total=line_count):
            L = line.strip().split()
            label = L[0]
            if label not in label_dict:
                if ignore_unlabeled and label == unlabel_label:
                    continue
                if label == unlabel_label:
                    label_dict[label] = -1.0
                elif label not in label_dict:
                    label_dict[label] = len(label_dict)
                    target_names.append(label)
            x_data.append(" ".join(L[1:]))
            y_data.append(label_dict[label])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    data = pd.DataFrame({"x_data": x_data, "y_data": y_data, "target_names": target_names})
    return data



# ================================================================================
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
    2. 
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



