# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import numpy as np
import torch
from torch import nn


class SimpleCNN(nn.Module):
    r"""A Simple 3-Conv CNN followed by a fully connected layer.

    Takes RGB and/or depth observations and produces an embedding vector.
    """

    def __init__(
        self,
        observation_space,
        output_size,
    ):
        super().__init__()

        # Number of RGB channels
        if "rgb" in observation_space.spaces:
            self._n_input_rgb = observation_space.spaces["rgb"].shape[2]
        else:
            self._n_input_rgb = 0

        # Number of depth channels
        if "depth" in observation_space.spaces:
            self._n_input_depth = observation_space.spaces["depth"].shape[2]
        else:
            self._n_input_depth = 0

        # CNN configuration
        self._cnn_layers_kernel_size = [
            (8, 8),
            (4, 4),
            (3, 3),
        ]

        self._cnn_layers_stride = [
            (4, 4),
            (2, 2),
            (1, 1),
        ]

        # Determine input image dimensions
        if self._n_input_rgb > 0:
            cnn_dims = np.array(
                observation_space.spaces["rgb"].shape[:2],
                dtype=np.float32,
            )
        elif self._n_input_depth > 0:
            cnn_dims = np.array(
                observation_space.spaces["depth"].shape[:2],
                dtype=np.float32,
            )
        else:
            cnn_dims = np.array([1, 1], dtype=np.float32)

        # Build CNN
        if self.is_blind:
            self.cnn = nn.Sequential()
        else:
            for kernel_size, stride in zip(
                self._cnn_layers_kernel_size,
                self._cnn_layers_stride,
            ):
                cnn_dims = self._conv_output_dim(
                    dimension=cnn_dims,
                    padding=np.array([0, 0], dtype=np.float32),
                    dilation=np.array([1, 1], dtype=np.float32),
                    kernel_size=np.array(kernel_size, dtype=np.float32),
                    stride=np.array(stride, dtype=np.float32),
                )

            self.cnn = nn.Sequential(
                # Convolution 1
                nn.Conv2d(
                    in_channels=self._n_input_rgb + self._n_input_depth,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[0],
                    stride=self._cnn_layers_stride[0],
                ),
                nn.ReLU(True),

                # Convolution 2
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=self._cnn_layers_kernel_size[1],
                    stride=self._cnn_layers_stride[1],
                ),
                nn.ReLU(True),

                # Convolution 3
                nn.Conv2d(
                    in_channels=64,
                    out_channels=32,
                    kernel_size=self._cnn_layers_kernel_size[2],
                    stride=self._cnn_layers_stride[2],
                ),

                # Convert feature maps into a vector
                nn.Flatten(),

                # Fully connected layer
                nn.Linear(
                    32 * int(cnn_dims[0]) * int(cnn_dims[1]),
                    output_size,
                ),
                nn.ReLU(True),
            )

        self.layer_init()

    def _conv_output_dim(
        self,
        dimension,
        padding,
        dilation,
        kernel_size,
        stride,
    ):
        """Calculate output height and width after convolution."""

        assert len(dimension) == 2

        out_dimension = []

        for i in range(len(dimension)):
            output = int(
                np.floor(
                    (
                        dimension[i]
                        + 2 * padding[i]
                        - dilation[i] * (kernel_size[i] - 1)
                        - 1
                    )
                    / stride[i]
                    + 1
                )
            )

            out_dimension.append(output)

        return tuple(out_dimension)

    def layer_init(self):
        """Initialize CNN weights using Kaiming initialization."""

        for layer in self.cnn:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight,
                    nn.init.calculate_gain("relu"),
                )

                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    @property
    def is_blind(self):
        """Return True if there is no RGB or depth input."""

        return (
            self._n_input_rgb + self._n_input_depth == 0
        )

    def forward(self, observations: Dict[str, torch.Tensor]):
        """Process RGB/depth observations through the CNN."""

        cnn_input = []

        # Process RGB
        if self._n_input_rgb > 0:
            rgb_observations = observations["rgb"]

            # [B, H, W, C] -> [B, C, H, W]
            rgb_observations = rgb_observations.permute(
                0, 3, 1, 2
            )

            # Normalize RGB pixels from [0, 255] to [0, 1]
            rgb_observations = (
                rgb_observations.float() / 255.0
            )

            cnn_input.append(rgb_observations)

        # Process depth
        if self._n_input_depth > 0:
            depth_observations = observations["depth"]

            # [B, H, W, C] -> [B, C, H, W]
            depth_observations = depth_observations.permute(
                0, 3, 1, 2
            )

            cnn_input.append(depth_observations)

        # Combine RGB and depth channels
        cnn_inputs = torch.cat(
            cnn_input,
            dim=1,
        )

        # Produce visual embedding
        return self.cnn(cnn_inputs)