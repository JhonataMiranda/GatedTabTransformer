import torch
from torch import nn
import torch.nn.functional as F

class SkipConnection(nn.Module):
    """
    Implements a residual (skip) connection in neural networks.
    
    This module adds the input to the output of a given module, which helps in:
    - Mitigating vanishing gradient problem
    - Enabling easier training of deeper networks
    - Allowing information to flow more directly through the network
    
    Args:
        module (nn.Module): The module to be wrapped with a skip connection
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs, **kwargs):
        """
        Forward pass that applies the module and adds the original inputs.
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments passed to the module
        
        Returns:
            torch.Tensor: Result of module(inputs) + inputs
        """
        return self.module(inputs, **kwargs) + inputs

class NormalizeAndApply(nn.Module):
    """
    Applies layer normalization before passing inputs to a module.
    
    This helps in:
    - Stabilizing the learning process
    - Normalizing inputs across different layers
    - Improving gradient flow
    
    Args:
        dimension (int): Dimension of the input features
        module (nn.Module): Module to be applied after normalization
    """
    def __init__(self, dimension, module):
        super().__init__()
        # Layer normalization applied across the specified dimension
        self.normalization = nn.LayerNorm(dimension)
        self.module = module

    def forward(self, inputs, **kwargs):
        """
        Normalizes inputs before applying the module.
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments passed to the module
        
        Returns:
            torch.Tensor: Normalized and transformed inputs
        """
        return self.module(self.normalization(inputs), **kwargs)

class GatedGEGLU(nn.Module):
    """
    Implements a Gated Gaussian Error Linear Unit (GEGLU) activation.
    
    This activation function:
    - Splits the input tensor into two parts
    - Applies GELU activation to the second part
    - Multiplies the first part with the activated second part
    
    Provides a more flexible activation mechanism compared to standard ReLU
    """
    def forward(self, inputs):
        """
        Applies gated GELU activation.
        
        Args:
            inputs: Input tensor split into two parts
        
        Returns:
            torch.Tensor: Gated and activated tensor
        """
        # Split input tensor into two parts along the last dimension
        a, b = inputs.chunk(2, dim=-1)
        # Multiply first part with GELU-activated second part
        return a * F.gelu(b)

class DenseFeedForward(nn.Module):
    """
    Implements a dense feed-forward neural network layer with gated activation.
    
    Key characteristics:
    - Expands input dimension
    - Uses gated activation
    - Applies dropout for regularization
    
    Args:
        dimension (int): Input feature dimension
        expansion (int, optional): Expansion factor for intermediate layer. Defaults to 4.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, dimension, expansion=4, dropout_rate=0.):
        super().__init__()
        # Pipeline of transformations: expansion, gated activation, dropout, projection
        self.pipeline = nn.Sequential(
            nn.Linear(dimension, dimension * expansion * 2),  # Expand input
            GatedGEGLU(),  # Apply gated activation
            nn.Dropout(dropout_rate),  # Regularization
            nn.Linear(dimension * expansion, dimension)  # Project back to original dimension
        )

    def forward(self, inputs, **kwargs):
        """
        Applies the feed-forward transformation.
        
        Args:
            inputs: Input tensor
            **kwargs: Additional arguments (not used)
        
        Returns:
            torch.Tensor: Transformed inputs
        """
        return self.pipeline(inputs)

class MultiHeadSelfAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention mechanism.
    
    Key components:
    - Projects inputs to queries, keys, and values
    - Splits into multiple attention heads
    - Computes attention scores
    - Applies scaled dot-product attention
    
    Args:
        dimension (int): Input feature dimension
        num_heads (int, optional): Number of attention heads. Defaults to 8.
        head_dim (int, optional): Dimension of each attention head. Defaults to 16.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.
    """
    def __init__(self, dimension, num_heads=8, head_dim=16, dropout_rate=0.):
        super().__init__()
        # Compute inner dimension based on heads and head dimension
        inner_dimension = head_dim * num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Scaling factor to stabilize gradients
        self.scale_factor = head_dim ** -0.5

        # Projections for query, key, value
        self.qkv_projection = nn.Linear(dimension, inner_dimension * 3, bias=False)
        # Output projection to map back to original dimension
        self.output_projection = nn.Linear(inner_dimension, dimension)
        # Dropout for regularization
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        """
        Computes multi-head self-attention.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, dimension)
        
        Returns:
            torch.Tensor: Attention-weighted representation
        """
        # Get input tensor dimensions
        batch_size, seq_length, _ = inputs.size()
        h, d = self.num_heads, self.head_dim

        # Project inputs to queries, keys, and values
        qkv = self.qkv_projection(inputs)
        # Split into separate Q, K, V tensors
        q, k, v = torch.split(qkv, [h * d, h * d, h * d], dim=-1)

        # Reshape and transpose for multi-head attention
        q = q.view(batch_size, seq_length, h, d).permute(0, 2, 1, 3)  
        k = k.view(batch_size, seq_length, h, d).permute(0, 2, 1, 3)  
        v = v.view(batch_size, seq_length, h, d).permute(0, 2, 1, 3)  

        # Compute attention scores with scaling
        similarity = torch.matmul(q, k.transpose(-2, -1)) * self.scale_factor
        # Apply softmax to get attention weights
        attention = F.softmax(similarity, dim=-1)
        # Apply dropout for regularization
        attention = self.dropout_layer(attention)

        # Compute weighted sum of values
        result = torch.matmul(attention, v)

        # Reshape back to original dimensions
        result = result.permute(0, 2, 1, 3).contiguous()
        result = result.view(batch_size, seq_length, h * d)

        # Final projection
        return self.output_projection(result)

class TransformerEncoder(nn.Module):
    """
    Implements a Transformer Encoder architecture.
    
    Consists of:
    - Token embedding layer
    - Multiple encoder layers with self-attention and feed-forward networks
    
    Args:
        vocab_size (int): Size of the vocabulary
        dimension (int): Embedding and hidden layer dimension
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        head_dim (int): Dimension of each attention head
        attn_dropout (float): Dropout rate for attention layers
        ff_dropout (float): Dropout rate for feed-forward layers
    """
    def __init__(self, vocab_size, dimension, num_layers, num_heads, head_dim, attn_dropout, ff_dropout):
        super().__init__()
        # Embedding layer to convert token indices to dense vectors
        self.token_embedding = nn.Embedding(vocab_size, dimension)
        
        # Create encoder layers
        self.layers = nn.ModuleList()

        # Construct multiple encoder layers
        for _ in range(num_layers):
            # Each layer consists of a self-attention block and a feed-forward block
            # Both blocks use skip connections and layer normalization
            self.layers.append(nn.ModuleList([
                SkipConnection(NormalizeAndApply(dimension, MultiHeadSelfAttention(dimension, num_heads, head_dim, attn_dropout))),
                SkipConnection(NormalizeAndApply(dimension, DenseFeedForward(dimension, dropout_rate=ff_dropout))),
            ]))

    def forward(self, inputs):
        """
        Forward pass through the Transformer Encoder.
        
        Args:
            inputs: Input tensor of token indices
        
        Returns:
            torch.Tensor: Encoded representation of input tokens
        """
        # Convert token indices to embeddings
        x = self.token_embedding(inputs)
        
        # Pass through each encoder layer
        for attention_layer, feedforward_layer in self.layers:
            # Apply self-attention
            x = attention_layer(x)
            # Apply feed-forward transformation
            x = feedforward_layer(x)
        
        return x