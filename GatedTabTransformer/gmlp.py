import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    """
    A residual block that adds the input to the output of a given layer function.

    This block implements a residual connection, which helps in mitigating the vanishing 
    gradient problem and allows for easier training of deeper neural networks.

    Args:
        layer_fn (callable): A layer function to be wrapped with a residual connection.
    """
    def __init__(self, layer_fn):
        super().__init__()
        self.layer_fn = layer_fn

    def forward(self, x, **kwargs):
        """
        Forward pass of the residual block.

        Args:
            x (torch.Tensor): Input tensor
            **kwargs: Additional arguments to be passed to the layer function

        Returns:
            torch.Tensor: Output tensor with the residual connection applied
        """
        return self.layer_fn(x, **kwargs) + x


class LayerNormWrapper(nn.Module):
    """
    A wrapper that applies Layer Normalization before passing the input to a given function.

    Args:
        dim (int): The dimension of the input to normalize
        fn (callable): The function to apply after normalization
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """
        Forward pass applying layer normalization.

        Args:
            x (torch.Tensor): Input tensor
            **kwargs: Additional arguments to be passed to the function

        Returns:
            torch.Tensor: Normalized and transformed tensor
        """
        return self.fn(self.norm_layer(x), **kwargs)


class GatingUnit(nn.Module):
    """
    A gating unit that modulates the input using a learned weight matrix.

    Args:
        input_dim (int): Dimensionality of the input
        seq_len (int): Length of the sequence
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        activation (nn.Module, optional): Activation function to apply. Defaults to Identity.
        heads (int, optional): Number of attention heads. Defaults to 1.
        init_range (float, optional): Range for weight initialization. Defaults to 1e-3.
        use_circulant (bool, optional): Whether to use circulant matrix. Defaults to False.
    """
    def __init__(self, input_dim, seq_len, causal=False, activation=nn.Identity(), heads=1, init_range=1e-3, use_circulant=False):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.activation_fn = activation
        dim_half = input_dim // 2

        self.norm = nn.LayerNorm(dim_half)

        # Initialize circulant parameters if specified
        if use_circulant:
            self.circulant_params_x = nn.Parameter(torch.ones(heads, seq_len))
            self.circulant_params_y = nn.Parameter(torch.ones(heads, seq_len))

        self.use_circulant = use_circulant
        weight_shape = (heads, seq_len) if use_circulant else (heads, seq_len, seq_len)
        self.weights = nn.Parameter(torch.zeros(weight_shape))
        nn.init.uniform_(self.weights, -init_range / seq_len, init_range / seq_len)

        self.biases = nn.Parameter(torch.ones(heads, seq_len))

    def forward(self, x, additional_gate=None):
        """
        Forward pass of the gating unit.

        Args:
            x (torch.Tensor): Input tensor
            additional_gate (torch.Tensor, optional): Additional gating input. Defaults to None.

        Returns:
            torch.Tensor: Gated output tensor
        """
        # Split input into residual and gate components
        residual, gate_input = x.chunk(2, dim=-1)
        gate_input = self.norm(gate_input)

        weight_matrix, biases = self.weights, self.biases

        # Handle circulant matrix if specified
        if self.use_circulant:
            dim_seq = weight_matrix.shape[-1]
            weight_matrix = F.pad(weight_matrix, (0, dim_seq), value=0)
            weight_matrix = weight_matrix.repeat_interleave(dim_seq, dim=1)[:, :-dim_seq].reshape(self.heads, dim_seq, 2 * dim_seq - 1)
            weight_matrix = weight_matrix[:, :, dim_seq - 1:]
            weight_matrix *= self.circulant_params_x.unsqueeze(-1) * self.circulant_params_y.unsqueeze(-2)

        # Apply causal masking if specified
        if self.causal:
            weight_matrix = weight_matrix[:, :x.shape[1], :x.shape[1]]
            biases = biases[:, :x.shape[1]]
            causal_mask = torch.triu(torch.ones_like(weight_matrix), diagonal=1).bool()
            weight_matrix = weight_matrix.masked_fill(causal_mask, 0.0)

        # Reshape and apply gating
        batch_size, seq_len, _ = gate_input.size()
        gate_input = gate_input.view(batch_size, seq_len, self.heads, -1).permute(0, 2, 1, 3)

        gated_output = torch.matmul(weight_matrix, gate_input.permute(1, 0, 2, 3))
        gated_output = gated_output.permute(1, 0, 2, 3) + biases.unsqueeze(1).unsqueeze(-1)
        gated_output = gated_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)

        # Add additional gate if provided
        if additional_gate is not None:
            gated_output += additional_gate

        return self.activation_fn(gated_output) * residual


class gMLPBlock(nn.Module):
    """
    A gMLP (Gated Multilayer Perceptron) block.

    Args:
        dim (int): Input dimension
        ff_dim (int): Feedforward dimension
        seq_len (int): Sequence length
        heads (int, optional): Number of attention heads. Defaults to 1.
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        activation (nn.Module, optional): Activation function. Defaults to Identity.
        use_circulant (bool, optional): Whether to use circulant matrix. Defaults to False.
    """
    def __init__(self, dim, ff_dim, seq_len, heads=1, causal=False, activation=nn.Identity(), use_circulant=False):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU()
        )
        self.gating_unit = GatingUnit(ff_dim, seq_len, causal, activation, heads, use_circulant=use_circulant)
        self.output_proj = nn.Linear(ff_dim // 2, dim)

    def forward(self, x):
        """
        Forward pass of the gMLP block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Transformed tensor
        """
        x = self.input_proj(x)
        x = self.gating_unit(x)
        return self.output_proj(x)


class gMLP(nn.Module):
    """
    Gated Multilayer Perceptron (gMLP) model for sequence classification.

    Args:
        patch_size (int): Size of input patches
        seq_len (int): Total sequence length
        num_classes (int): Number of output classes
        model_dim (int): Model's embedding dimension
        depth (int): Number of gMLP layers
        num_heads (int, optional): Number of attention heads. Defaults to 1.
        ff_multiplier (int, optional): Multiplier for feedforward dimension. Defaults to 4.
        survival_prob (float, optional): Probability of layer survival during training. Defaults to 1.0.
    """
    def __init__(self, patch_size, seq_len, num_classes, model_dim, depth, num_heads=1, ff_multiplier=4, survival_prob=1.0):
        super().__init__()
        assert model_dim % num_heads == 0, "Model dimension must be divisible by the number of heads"
        num_patches = seq_len // patch_size
        ff_dim = model_dim * ff_multiplier

        # Patch embedding layer
        self.patch_embedding = nn.Sequential(
            nn.Unflatten(1, (num_patches, patch_size)),
            nn.Linear(patch_size, model_dim)
        )

        # Stacked gMLP layers with residual connections and layer normalization
        self.layers = nn.ModuleList([
            ResidualBlock(
                LayerNormWrapper(
                    model_dim,
                    gMLPBlock(dim=model_dim, ff_dim=ff_dim, seq_len=num_patches, heads=num_heads)
                )
            )
            for _ in range(depth)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, num_classes)
        )
        self.survival_prob = survival_prob

    def forward(self, x):
        """
        Forward pass of the gMLP model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Class prediction
        """
        # Patch embedding
        x = self.patch_embedding(x)

        # Stochastic depth (layer dropout) during training
        if self.training and self.survival_prob < 1.0:
            layers = [layer for layer in self.layers if torch.rand(1).item() < self.survival_prob]
        else:
            layers = self.layers

        # Apply gMLP layers
        for layer in layers:
            x = layer(x)

        # Global average pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)