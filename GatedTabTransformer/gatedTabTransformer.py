import torch
import torch.nn.functional as F
from torch import nn
from GatedTabTransformer.gmlp import gMLP
from GatedTabTransformer.transformerEncoder import TransformerEncoder


class GatedTabTransformer(nn.Module):
    """
    A hybrid neural network for tabular data processing combining Transformer and gMLP architectures.

    This model is designed to handle mixed-type input data with categorical and numerical features,
    using a Transformer encoder for categorical features and a gated MLP for final classification.

    Key Features:
    - Handles categorical features through embedding and Transformer encoding
    - Supports optional normalization of numerical features
    - Flexible architecture with configurable encoder and MLP parameters
    - Supports various output dimensions (binary, multi-class classification)

    Args:
        category_sizes (list): List of vocabulary sizes for each categorical feature
        num_numerical_features (int): Number of numerical features in the input
        encoder_dim (int): Dimensionality of the Transformer encoder
        encoder_depth (int): Number of layers in the Transformer encoder
        encoder_heads (int): Number of attention heads in the Transformer encoder
        encoder_head_dim (int, optional): Dimensionality of each attention head. Defaults to 8.
        output_dim (int, optional): Number of output classes. Defaults to 1 (binary classification).
        mlp_layers (int, optional): Number of layers in the final gMLP. Defaults to 6.
        special_token_count (int, optional): Number of special tokens to add. Defaults to 2.
        numerical_mean_std (torch.Tensor, optional): Pre-computed mean and std for numerical features. 
            Should have shape (num_numerical_features, 2). Defaults to None.
        attention_dropout (float, optional): Dropout rate for attention layers. Defaults to 0.
        feedforward_dropout (float, optional): Dropout rate for feedforward layers. Defaults to 0.
        mlp_hidden_dim (int, optional): Hidden dimension for the final MLP. Defaults to 64.

    Raises:
        AssertionError: If category sizes are invalid or input dimensions are incorrect
    """
    def __init__(
        self,
        *,
        category_sizes,
        num_numerical_features,
        encoder_dim,
        encoder_depth,
        encoder_heads,
        encoder_head_dim=8,
        output_dim=1,
        mlp_layers=6,
        special_token_count=2,
        numerical_mean_std=None,
        attention_dropout=0.,
        feedforward_dropout=0.,
        mlp_hidden_dim=64,
    ):
        super().__init__()
        # Validate that all category sizes are positive
        assert all(cat_size > 0 for cat_size in category_sizes), "Each category size must be greater than zero."

        # Calculate total number of categorical embeddings
        self.num_categories = len(category_sizes)
        self.total_category_embeddings = sum(category_sizes)

        # Add special tokens to the total token count
        self.special_token_count = special_token_count
        total_tokens = self.total_category_embeddings + special_token_count

        # Create embedding offsets to ensure unique token indices for each category
        offset = F.pad(torch.tensor(category_sizes), (1, 0), value=special_token_count)
        offset = offset.cumsum(dim=-1)[:-1]
        self.register_buffer('embedding_offset', offset)

        # Validate and register numerical feature normalization parameters
        if numerical_mean_std is not None:
            assert numerical_mean_std.shape == (num_numerical_features, 2), (
                f"numerical_mean_std should have shape ({num_numerical_features}, 2), representing mean and std."
            )
        self.register_buffer('numerical_mean_std', numerical_mean_std)

        # Numerical feature normalization layer
        self.numerical_layer_norm = nn.LayerNorm(num_numerical_features)
        self.num_numerical_features = num_numerical_features

        # Transformer encoder for categorical features
        self.encoder = TransformerEncoder(
            vocab_size=total_tokens,
            dimension=encoder_dim,
            num_layers=encoder_depth,
            num_heads=encoder_heads,
            head_dim=encoder_head_dim,
            attn_dropout=attention_dropout,
            ff_dropout=feedforward_dropout
        )

        # Final classifier using gated MLP
        combined_feature_size = (encoder_dim * self.num_categories) + num_numerical_features
        self.classifier = gMLP(
            patch_size=1,
            seq_len=combined_feature_size,
            num_classes=output_dim,
            model_dim=mlp_hidden_dim,
            depth=mlp_layers
        )

    def forward(self, categorical_inputs, numerical_inputs=None):
        """
        Forward pass of the Gated Tab Transformer.

        Args:
            categorical_inputs (torch.Tensor): Categorical input tensor 
                with shape (batch_size, num_categories)
            numerical_inputs (torch.Tensor, optional): Numerical input tensor 
                with shape (batch_size, num_numerical_features). Defaults to None.

        Returns:
            torch.Tensor: Model predictions with shape (batch_size, output_dim)

        Raises:
            AssertionError: If input dimensions are incorrect
        """
        # Validate categorical input dimensions
        assert categorical_inputs.shape[-1] == self.num_categories, (
            f"Expected {self.num_categories} categorical inputs."
        )

        # Add embedding offsets to ensure unique token indices
        categorical_inputs += self.embedding_offset
        
        # Encode categorical inputs using Transformer
        encoded_categ = self.encoder(categorical_inputs)

        # Flatten the transformer outputs
        flattened_categ = encoded_categ.flatten(1)

        # Process numerical features
        if self.num_numerical_features > 0:
            # Validate numerical input dimensions
            assert numerical_inputs is not None, "Numerical inputs are required"
            assert numerical_inputs.shape[1] == self.num_numerical_features, (
                f"Expected {self.num_numerical_features} numerical inputs."
            )

            # Normalize numerical features if mean and std are provided
            if self.numerical_mean_std is not None:
                mean, std = self.numerical_mean_std.unbind(dim=-1)
                numerical_inputs = (numerical_inputs - mean) / std

            # Apply layer normalization to numerical features
            normalized_numerical = self.numerical_layer_norm(numerical_inputs)

            # Concatenate categorical and numerical features
            combined_inputs = torch.cat((flattened_categ, normalized_numerical), dim=-1)
        else:
            combined_inputs = flattened_categ

        # Pass combined features through MLP for final prediction
        return self.classifier(combined_inputs)