import torch
import torch.nn as nn
# import torch.nn.functional as F
from utils.global_p import *
import numpy as np
import os
from sklearn.metrics import *
from itertools import product


import logging

class RankNet(nn.Module):
	append_id = True
	include_id = False
	include_user_features = False
	include_item_features = False
	include_context_features = False
	
	@staticmethod
	def evaluate_method(p, data, metrics):
		"""
		calculate evaluation metrics
		:param p: prediction values，np.array，generated by runner.predict()
		:param data: data dict，generated by DataProcessor
		:param metrics: metrics list，generated by runner.metrics，for example ['rmse', 'auc']
		:return:
		"""
		l = data['Y']
		evaluations = []
		threshold = None
		for metric in metrics:
			if metric == 'rmse':
				evaluations.append(np.sqrt(mean_squared_error(l, p)))
			elif metric == 'mae':
				evaluations.append(mean_absolute_error(l, p))
			elif metric == 'auc':
				evaluations.append(roc_auc_score(l, p))
			elif metric == 'f1':
				precision, recall, thresholds = precision_recall_curve(l, p)
				f1_score = 2 * recall * precision / (recall + precision + 1e-6)
				threshold = thresholds[np.argmax(f1_score)]
				evaluations.append(np.max(f1_score))
			elif metric == 'accuracy':
				evaluations.append(accuracy_score(l, np.where(p < threshold, 0, 1)))
			elif metric == 'logloss':
				evaluations.append(log_loss(l, p, eps=1e-6))
			elif metric == 'precision':
				evaluations.append(precision_score(l, np.around(p)))
			elif metric == 'recall':
				evaluations.append(recall_score(l, np.around(p)))
			else:
				pass  # temporary
		return evaluations
	
	@staticmethod
	def generate_pairs(pred, y):
		pred = torch.flatten(pred)
		y = torch.flatten(pred)

		# generate every pair of indices
		data_pairs = list(product(range(len(y)), repeat=2))

		true_pairs = y[data_pairs]
		pred_pairs = pred[data_pairs]

		# calculate reletaive relevance of pairs
		true_diffs = true_pairs[:, 0] - true_pairs[:, 1]
		pred_diffs = pred_pairs[:, 0] - pred_pairs[:, 1]

		return [pred_diffs, true_diffs]

	def __init__(self, user_num, item_num, u_vector_size, i_vector_size, model_path, smooth_coef=0.1, layers=None, loss_func='BCE'):
		super(RankNet, self).__init__()
		self.u_vector_size, self.i_vector_size = u_vector_size, i_vector_size
		assert self.u_vector_size == self.i_vector_size
		self.ui_vector_size = self.u_vector_size
		self.user_num = user_num
		self.item_num = item_num
		self.uid_embeddings = torch.nn.Embedding(self.user_num, self.ui_vector_size)
		self.iid_embeddings = torch.nn.Embedding(self.item_num, self.ui_vector_size)
		self.activation = torch.nn.Sigmoid()
		self.l2_embeddings = ['uid_embeddings', 'iid_embeddings']
		self.model_path = model_path
		self.optimizer = None

		self.smooth_coef = smooth_coef
		self.loss_func = loss_func.lower()

		pre_size = self.u_vector_size + self.i_vector_size

		self.prediction = nn.Sequential(
			nn.Linear( pre_size, 512),
			nn.Dropout(0.5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(512, 256),
			nn.Dropout(0.5),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Linear(256, 1),
		)
	
	def forward(self, feed_dict):
		out_dict = self.predict(feed_dict)
		pred = out_dict['prediction']
		y = feed_dict['Y']

		pred, y = self.generate_pairs(pred, y)

		if 'mse' in self.loss_func:
			loss = torch.nn.MSELoss(reduction='mean')(pred, y)
		elif 'bce' in self.loss_func:
			loss = torch.nn.BCEWithLogitsLoss(reduction='mean')(pred, y)
		elif 'hinge' in self.loss_func:
			loss = torch.nn.HingeEmbeddingLoss(reduction='mean')(out_dict['hinge_pred'], 1 - y * 2)
		elif 'focal' in self.loss_func:
			fpred = torch.clamp(pred, min=1e-6, max=(1 - 1e-6))
			loss = torch.mean(-y * torch.log(fpred) * torch.pow((1 - fpred), 2) - (1 - y) * torch.log(1 - fpred) * torch.pow(fpred, 2))
		elif 'maxr' in self.loss_func:
			loss = torch.mean(
				torch.max((y + self.smooth_coef) / (pred + self.smooth_coef),
						  (pred + self.smooth_coef) / (y + self.smooth_coef)))
		elif 'sumr' in self.loss_func:
			loss = torch.mean((y + pred + self.smooth_coef) / (self.smooth_coef + pred * y))
		elif 'logmin' in self.loss_func:
			loss = torch.mean(torch.log(torch.ones_like(pred) / (torch.min(y, pred) + self.smooth_coef) + self.smooth_coef) * (torch.min(y, pred) + y + pred))
		elif 'predpredlabel':
			loss = torch.mean(pred*(pred+y))
		else:
			print('NO PRE-DEFINED LOSS ERROR!')
			exit(0)
		out_dict['loss'] = loss
		return out_dict
	
	def predict(self, feed_dict):
		# unclear what these do later... check_list, embedding_l2 = [], []
		u_ids = feed_dict['X'][:, 0]
		i_ids = feed_dict['X'][:, 1]

		# user/item vectors shape: (batch_size, embedding_size)
		user_vectors = self.uid_embeddings(u_ids)
		item_vectors = self.iid_embeddings(i_ids)

		# concat iids with uids
		new_X = torch.cat((user_vectors, item_vectors), dim=1)
		# embedding_l2.append(item_vectors)

		hinge_pred = self.prediction(new_X) # I think this works???
		prediction = self.activation(hinge_pred)
		out_dict = {'prediction': prediction, 'hinge_pred': hinge_pred} # unclear what these are for... , 'check': check_list, 'embedding_l2': embedding_l2}
		return out_dict
	
	def l2(self):
		l2 = 0
		for p in self.parameters():
			l2 += (p ** 2).sum()
		return l2
	
	def save_model(self, model_path=None):
		"""
		save model
		"""
		if model_path is None:
			model_path = self.model_path
		dir_path = os.path.dirname(model_path)
		if not os.path.exists(dir_path):
			os.mkdir(dir_path)
		torch.save(self.state_dict(), model_path)
	
	def load_model(self, model_path=None):
		"""
		load model
		"""
		if model_path is None:
			model_path = self.model_path
		self.load_state_dict(torch.load(model_path))
		self.eval()