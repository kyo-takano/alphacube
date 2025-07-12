"""
Model Loader

This module provides a function and classes for loading and using a pre-trained AlphaCube solver model.

Function:
    `load_model(model_id, cache_dir)`: Load the pre-trained AlphaCube solver model.

Classes:
    - `Model`: The MLP architecture for the AlphaCube solver.
    - `LinearBlock`: A building block for the MLP architecture.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from .utils import logger, logger_args, dtype, cache_dir


def load_model(
    model_id="small",
    cache_dir=cache_dir,
):
    """
    Load the pre-trained AlphaCube solver model.

    Args:
        model_id (str): Identifier for the model variant to load ("small", "base", or "large").
        cache_dir (str): Directory to cache the downloaded model.

    Returns:
        nn.Module: Loaded AlphaCube solver model.
    """
    model_path = os.path.join(cache_dir, model_id + ".zip")
    if not os.path.exists(model_path):
        import requests
        from rich.progress import Progress

        os.makedirs(cache_dir, exist_ok=True)
        model_url = os.path.join("https://storage.googleapis.com/alphacube/", model_id + ".zip")
        logger.info(f"[grey50]Downloading AlphaCube ({model_id}) from {model_url}", **logger_args)
        with requests.get(model_url, stream=True) as r:
            total_size = int(r.headers.get("Content-Length"))
            with Progress() as progress:
                task = progress.add_task(
                    f"[cyan]Downloading [bold]`{model_id}`[/bold]", total=total_size
                )
                with open(model_path, "wb") as output:
                    for chunk in r.iter_content(chunk_size=8192):
                        output.write(chunk)
                        progress.update(task, advance=len(chunk))
        logger.info(f"[grey50]Saved to {model_path}", **logger_args)
    else:
        logger.info(f"[grey50]Loading AlphaCube solver from cache at {model_path}", **logger_args)
    try:
        state_dict = torch.load(model_path, weights_only=True, map_location=torch.device("cpu"))
    except Exception as e:
        os.remove(model_path)
        raise ValueError(
            f"Failed to load the model from '{model_path}'. The file is likely corrupt. "
            f"The cached model has been deleted to enable a fresh download on the next run. Original error: {e}"
        )

    if model_id in {"small", "base", "large"}:
        module = Model_v1
        embed_dim, num_layers = {
            "small": (1024, 7),
            "base": (2048, 8),
            "large": (4096, 8),
        }[model_id]
    else:
        module = Model
        embed_dim, num_layers = (
            state_dict["layers.0.fc.weight"].shape[0],
            sum(1 for k in state_dict.keys() if k.endswith(".fc.weight")),
        )

    model = module(embed_dim, num_layers)
    model.load_state_dict(state_dict)
    model.to(dtype)
    return model


class Model_v1(nn.Module):
    """
    The MLP architecture for the compute-optimal Rubik's Cube solver introduced in the following paper:
    https://openreview.net/forum?id=bnBeNFB27b
    """

    def __init__(self, embed_dim=4096, num_hidden_layers=8, input_dim=324, output_dim=18):
        super(Model_v1, self).__init__()
        self.layers = nn.ModuleList(
            [LinearBlock(input_dim, embed_dim)]
            + [LinearBlock(embed_dim, embed_dim) for i in range(num_hidden_layers - 1)]
            + [nn.Linear(embed_dim, output_dim)]
        )

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor representing the problem state.

        Returns:
            torch.Tensor: Predicted distribution over possible solutions.
        """
        x = F.one_hot(inputs, 6).reshape(-1, 324).to(dtype)
        for layer in self.layers:
            x = layer(x)
        logits = x
        return logits


class Model(nn.Module):
    """
    An architecture better than `Model`.

    **Changes**:
    - Remove ReLU activation from the first layer (`embedding`), which had the dying ReLU problem.
    - Following the recent convention, the `embedding` layer does *not* count as one hidden layer.
    """

    def __init__(self, hidden_size=4096, num_hidden_layers=8, input_dim=324, output_dim=18):
        super(Model, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_size, bias=False)
        self.layers = nn.ModuleList(
            [LinearBlock(hidden_size, hidden_size) for i in range(num_hidden_layers)]
        )
        self.head = nn.Linear(hidden_size, output_dim, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize all weight such that the activation variances are approximately 1.0, with bias terms & the output layer weights being zeros
        """
        nn.init.normal_(self.embedding.weight, std=1 / 54**0.5)  # There'll be 54 ones in a sample
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.fc.weight)
            if layer.fc.bias is not None:
                nn.init.zeros_(layer.fc.bias)
        nn.init.zeros_(self.head.weight)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor representing the problem state.

        Returns:
            torch.Tensor: Predicted distribution over possible solutions.
        """
        x = F.one_hot(inputs, 6).reshape(-1, 324).to(dtype)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        logits = self.head(x)
        return logits


class LinearBlock(nn.Module):
    """
    A building block for the MLP architecture.

    This block consists of a linear layer followed by ReLU activation and batch normalization.
    """

    def __init__(self, input_prev, embed_dim):
        super(LinearBlock, self).__init__()
        self.fc = nn.Linear(input_prev, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

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
        x = F.relu(x)
        x = self.bn(x)
        return x
