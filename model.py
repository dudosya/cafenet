import torch
import torch.nn as nn
import torchvision



class CentroidAttention(nn.Module):
    def __init__(self, num_classes, feature_dim, attention_dim, qkv_bias=False, attn_drop=0., proj_drop=0.):
        """
        Initializes the Centroid Attention module.

        Args:
            num_classes (int): Number of classes.
            feature_dim (int): Dimension of input features.
            attention_dim (int): Dimension for attention computations.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            attn_drop (float): Dropout rate on attention weights.
            proj_drop (float): Dropout rate on output projections.
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.scale = attention_dim ** -0.5  # Scaling factor for attention scores
        self.num_classes = num_classes

        # Centroids (as buffers, not learnable parameters)
        self.register_buffer('centers', torch.zeros(num_classes, feature_dim))
        self.register_buffer('center_values', torch.zeros(num_classes, feature_dim))
        self.register_buffer('center_counts', torch.zeros(num_classes, 1))

        # Query, Key, and Value projections
        self.query = nn.Linear(feature_dim, attention_dim, bias=qkv_bias)
        self.key = nn.Linear(feature_dim, attention_dim, bias=qkv_bias)
        self.value = nn.Linear(feature_dim, attention_dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)

        if self.feature_dim != self.attention_dim:
            self.proj = nn.Linear(self.attention_dim, self.feature_dim)
            self.proj_drop = nn.Dropout(proj_drop)
        else:
            self.proj = nn.Identity()
            self.proj_drop = nn.Identity()

    def update_center(self, features, labels):
        """
        Updates the class centroids based on the current batch features.

        Args:
            features (Tensor): Feature embeddings from the batch [batch_size, feature_dim].
            labels (Tensor): Corresponding labels [batch_size].

        Returns:
            centers (Tensor): Updated centroids.
        """
        if labels is not None:
            with torch.no_grad():
                for i in torch.unique(labels):
                    idx = (labels == i)
                    if idx.sum() > 0:
                        class_features = features[idx]
                        class_sum = class_features.sum(dim=0)
                        self.center_values[i] += class_sum
                        self.center_counts[i] += idx.sum()

                        # Update the centroid
                        self.centers[i] = self.center_values[i] / self.center_counts[i]
        else:
            self.center_values.zero_()
            self.center_counts.zero_()

        return self.centers

    def forward(self, features, labels=None):
        # Update Centroids
        centers = self.update_center(features, labels)

        # Compute attention scores
        q = self.query(features)  # [batch_size, attention_dim]
        k = self.key(centers)     # [num_classes, attention_dim]
        scores = torch.matmul(q, k.t()) * self.scale  # [batch_size, num_classes]
        scores = torch.softmax(scores, dim=-1)
        scores = self.attn_drop(scores)

        # Compute attention-weighted values
        v = self.value(centers)  # [num_classes, attention_dim]
        attention_values = torch.matmul(scores, v)  # [batch_size, attention_dim]
        attention_values = self.proj(attention_values)
        attention_values = self.proj_drop(attention_values)

        # Concatenate features and attention values
        output = torch.cat([features, attention_values], dim=-1)

        return output



class CaFeNet(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the CaFeNet model.

        Args:
            num_classes (int): Number of classes.
        """
        super().__init__()
        # Load pre-trained EfficientNet-B0 backbone
        self.feature_extractor = torchvision.models.efficientnet_b0(pretrained=True)
        # Remove the last classification layer
        self.feature_extractor.classifier = nn.Identity()
        # Feature dimension after EfficientNet-B0
        self.feature_dim = 1280  # EfficientNet-B0 outputs 1280 features

        self.num_classes = num_classes

        # Centroid Attention module
        self.attention_dim = self.feature_dim  # Can be adjusted
        self.attn = CentroidAttention(num_classes, self.feature_dim, self.attention_dim)

        # Classification head
        # Since we concatenate features and attention values
        self.classifier = nn.Linear(self.feature_dim + self.feature_dim, num_classes)

    def forward(self, x, labels=None):
        # Extract features using EfficientNet-B0 backbone
        features = self.feature_extractor(x)

        # Apply Centroid-aware Feature Recalibration
        features = self.attn(features, labels)

        # Classification layer
        logits = self.classifier(features)

        return logits, features
