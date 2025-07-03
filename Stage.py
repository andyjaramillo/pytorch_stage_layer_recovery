import torch
import torch.nn as nn
from typing import Callable
class Stage(nn.Module):
    """iterates thoug"""
    def __init__(self, wrapped_module: nn.Module):
        super().__init__()
        self.wrapped_module = wrapped_module
        self.add_module('wrapped_module', wrapped_module)
        
    def forward(self, *args, **kwargs):
        try:
            return self.wrapped_module(*args, **kwargs)
        except Exception as e:
            print("made it", self.wrapped_module)
            return self.wrapped_module(*args, **kwargs)


class StagedModule:
    stages = []
    def __init__(self, module: nn.Module, restartMethod: Callable, epochs):
        module.__init__()
        self.module = module
        self.restartMethod = restartMethod
        self.epochs = epochs
        self.t = 0
    
    def run(self):
        try:
            while self.t < self.epochs:
                print(f"Epoch {self.t+1}\n-------------------------------")
                self.restartMethod()
                self.t += 1
                
        except Exception as e:
            print("made it boyyy", self.t)
            self.run()