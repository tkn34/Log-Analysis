# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 19:08:07 2022

@author: 81901
"""
from tqdm import tqdm
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, Counter


class FeatureExtraction:
    def __init__(self, x, config, mode="train",fe_type="tfilf"):
        self.config = config
        self.x = x
        self.mode = mode
        self.fe_type = fe_type # "tfilf" or "tfidf"
    
    def __call__(self):
        # Build Vocablary
        self.vocabulary = self._create_vocab()
        self._save_vocab()
        #print("Build Vocablary : Done")
        # Crete Vector
        self.vector = self._create_vector()
        #print("Crete Vector    : Done")
        # Create Feature
        self.feature, self.ixf_dict = self._create_feature()
        self._save_feature()
        #print("Create Feature  : Done")
        return self.feature, self.vocabulary
        
    def _create_vocab(self):
        if self.mode == "train":
            vocabulary = {}
            for line in self.x:
                token_list = line.strip().split()
                for token in token_list:
                    if token not in vocabulary:
                        vocabulary[token] = len(vocabulary)
        else:
            with open(self.config.output_dir + "vocablary.pickle", "rb") as f:
                vocabulary = pickle.load(f)
        return vocabulary
        
    def _save_vocab(self):
        if self.mode == "train":
            with open(self.config.output_dir + "vocablary.pickle", "wb") as f:
                pickle.dump(self.vocabulary, f)
        
    def _create_vector(self):
        result = []
        for line in self.x:
            temp = []
            token_list = line.strip().split()
            if token_list:
                for token in token_list:
                    if token not in self.vocabulary:
                        continue
                    else:
                        temp.append(self.vocabulary[token])
            result.append(temp)
        return np.array(result)        

    def _create_feature(self):
        if self.fe_type == "tfilf":
            print("tfilf")
            feature, ilf_dict = self._create_feature_tf_ilf()
        else:
            print("tfidf")
            feature, ilf_dict = self._create_feature_tf_idf()
        return feature, ilf_dict

    def _create_feature_tf_ilf(self):
        feature_vectors = []
        # Calculate lf
        token_index_ilf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_ilf_dict[token].add(location)
        # Calculate ilf
        ilf_dict = {}
        max_length = len(max(self.vector, key=len))
        for token in token_index_ilf_dict:
            ilf_dict[token] = np.log(float(max_length) / float(len(token_index_ilf_dict[token]) + 0.01))
        # Create feature
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * ilf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        return feature, ilf_dict

    def _create_feature_tf_idf(self):
        feature_vectors = []
        # Calculate lf
        token_index_idf_dict = defaultdict(set)
        for line in self.vector:
            for location, token in enumerate(line):
                token_index_idf_dict[token].add(location)
        # Calculate idf
        idf_dict = {}
        total_log_num = len(self.vector)
        for token in token_index_idf_dict:
            idf_dict[token] = np.log(float(total_log_num) / float(len(token_index_idf_dict[token]) + 0.01))
        # Create feature    
        tfinvf = []
        for line in self.vector:
            cur_tfinvf = np.zeros(len(self.vocabulary))
            count_dict = Counter(line)
            for token_index in line:
                cur_tfinvf[token_index] = (float(count_dict[token_index]) * idf_dict[token_index])
            tfinvf.append(cur_tfinvf)
        tfinvf = np.array(tfinvf)
        feature_vectors.append(tfinvf)
        feature = np.hstack(feature_vectors)
        return feature, idf_dict

    def _save_feature(self):
        if self.mode == "train":
            if self.fe_type == "tfilf":
                with open(self.config.output_dir + "tf_ilf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)
            else:
                with open(self.config.output_dir + "tf_idf.pickle", "wb") as f:
                    pickle.dump(self.ixf_dict, f)













