# # -*- coding: utf-8 -*-
# # __author__:Song Zhenqi
# # 2021-02-01
#
# from torch import optim
# from abc import ABC, abstractmethod
# import torch
#
#
# class BaseOptimizer(ABC):
#     def __init__(self,
#                  parameters,
#                  clip_norm,**kwargs):
#         self.clip_norm = clip_norm
#         self.optimer = self.create_optimizer(parameters, **kwargs)
#
#     @abstractmethod
#     def create_optimizer(self, *args, **kwargs):
#         return None
#
#     def __call__(self, *args, **kwargs):
#         if self.clip_norm:
#             torch.nn.utils.clip_grad_norm_()
#
#     def step(self):
#         self.optimer.step()
#
#
# class Adam(BaseOptimizer):
#     def __init__(self,
#                  parameters,
#                  clip_norm):
#         super(Adam, self).__init__(**kwargs)
#
#
#     def create_optimizer(self, *args, **kwargs):
