from utils.code_crawler import CodeCrawler

# Repo's to use as corpus data

# Scientific packages
import sklearn, skimage, numpy, scipy, sympy
import networkx

# Deep-learning / auto-diff packages
import tensorflow, keras, theano, torch, dynet, autograd, tangent, chainer

# I/O and viz packages
import pygame, seaborn, matplotlib, h5py, feather

# Data management packages
import dask, pandas, sqlalchemy


# WORKING: sklearn, numpy, scipy, sympy, mpl, pygame, sqlalchemy, skimage, h5py,
# dask, seaborn, tensorflow, keras, dynet, autograd, tangent, chainer, networkx,
# pandas, theano, torch

# NOT WORKING: N/A

# NO FUNCTIONS FOUND: feather

# modules = [networkx, pandas, theano, torch]
modules = [sklearn, numpy, scipy, sympy, matplotlib, pygame, sqlalchemy, skimage, h5py,
           dask, seaborn, tensorflow, keras, dynet, autograd, tangent, chainer,
           networkx, pandas, theano, torch]
# modules = [sklearn, numpy, scipy, sympy, matplotlib, pygame, sqlalchemy]

ccrawler = CodeCrawler(modules=modules)
print("Building function dict")
ccrawler.build_function_dict("../../data/")
print("Building code/comment dict")
ccrawler.build_code_comment_pairs("../../data/")
print("Getting output sentences")
ccrawler.get_sentence_output("code-sentences.output", "comm-sentences.output")
print("Making train/dev/test splits")
ccrawler.split_code_comment_data()
print("done.")
