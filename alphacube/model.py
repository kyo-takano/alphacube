"""
Model Loader

This module provides a function and classes for loading and using a pre-trained AlphaCube solver model.

Function:
    ``load_model(model_id, cache_dir)``: Load the pre-trained AlphaCube solver model.
    
Classes:
    - ``Model``: The MLP architecture for the AlphaCube solver.
    - ``LinearBlock``: A building block for the MLP architecture.
"""

import os
import torch
from . import logger, logargs

def load_model(
    model_id="small",
    cache_dir=os.path.expanduser("~/.cache/alphacube"),
):
    """
    Load the pre-trained AlphaCube solver model.
    
    Args:
        model_id (str): Identifier for the model variant to load ("small", "base", or "large").
        cache_dir (str): Directory to cache the downloaded model.
        
    Returns:
        torch.nn.Module: Loaded AlphaCube solver model.
    """
    model_path = os.path.join(cache_dir, model_id + ".zip")
    if not os.path.exists(model_path):
        import requests
        from rich.progress import Progress
        os.makedirs(cache_dir, exist_ok=True)
        model_url = os.path.join("https://storage.googleapis.com/alphacube/", model_id + ".zip")
        logger.info(f"[grey50]Downloading AlphaCube ({model_id}) from {model_url}", **logargs)
        with requests.get(model_url, stream=True) as r:
            total_size = int(r.headers.get("Content-Length"))
            with Progress() as progress:
                task = progress.add_task(f"[cyan]Downloading ...", total=total_size)
                with open(model_path, 'wb') as output:
                    for chunk in r.iter_content(chunk_size=8192):
                        output.write(chunk)
                        progress.update(task, completed=len(chunk))
        logger.info(f"[grey50]Saved to {model_path}", **logargs)
        try:
            state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        except Exception as e:
            os.remove(model_path)
            raise ValueError(f"The model file appears to be broken, most likely because of permission error (deleted):\n{e}")
    else:
        logger.info(f"[grey50]Loading AlphaCube solver from cache at {model_path}", **logargs)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    embed_dim, num_layers = {
        "small":    (1024, 7),
        "base":     (2048, 8),
        "large":    (4096, 8)
    }[model_id]
    model = Model(embed_dim, num_layers)
    model.load_state_dict(state_dict)
    return model

class Model(torch.nn.Module):
    """
    The MLP architecture for the compute-optimal Rubik's Cube solver introduced in the following paper:
    https://openreview.net/forum?id=bnBeNFB27b
    """
    def __init__(self, embed_dim=4096, num_hidden_layers=8, input_dim=324, output_dim=18):
        super(Model, self).__init__()
        self.one_hot = torch.nn.functional.one_hot
        self.layers = torch.nn.ModuleList(
            [LinearBlock(input_dim, embed_dim)] + \
            [LinearBlock(embed_dim, embed_dim) for i in range(num_hidden_layers-1)] + \
            [torch.nn.Linear(embed_dim, output_dim)]
        )
        self.softmax = torch.nn.functional.softmax

    def forward(self, inputs):
        """
        Forward pass of the model.
        
        Args:
            inputs (torch.Tensor): Input tensor representing the problem state.
        
        Returns:
            torch.Tensor: Predicted distribution over possible solutions.
        """
        x = self.one_hot(inputs, num_classes=6).to(torch.float).reshape(-1, 324)
        for layer in self.layers:
            x = layer(x)
        p_dist = self.softmax(x, dim=-1)
        return p_dist

class LinearBlock(torch.nn.Module):
    """
    A building block for the MLP architecture.
    
    This block consists of a linear layer followed by ReLU activation and batch normalization.
    """
    def __init__(self, input_prev, embed_dim):
        super(LinearBlock, self).__init__()
        self.fc = torch.nn.Linear(input_prev, embed_dim)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, inputs):
        """
        Forward pass of the linear block.
        
        Args:
            inputs (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor after linear transformation, ReLU activation, and batch normalization.
        """
        x = inputs
        x = self.fc(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

