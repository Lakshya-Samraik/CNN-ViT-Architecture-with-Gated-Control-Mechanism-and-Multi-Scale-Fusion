class PatchEmbedding(nn.Module):
    """Convert CNN feature maps to patch embeddings for ViT"""
    
    def __init__(self, in_channels=768, embed_dim=768, patch_size=1):
        super().__init__()
        # Use 1x1 conv to convert feature maps to embeddings
        # This preserves spatial information while creating embeddings
        self.projection = nn.Conv2d(in_channels, embed_dim, 
                                  kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # Input: [B, C, H, W] -> Output: [B, N, embed_dim] where N = H*W
        x = self.projection(x)  # [B, embed_dim, H', W']
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        return x
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4., dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MHSA(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = nn.Sequential(
                                    nn.Linear(dim, int(dim * mlp_ratio)),
                                    nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(int(dim * mlp_ratio), dim),
                                    nn.Dropout(dropout))
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
class MHSA(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Feature extraction layers - designed to output 768 channels for ViT
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5 - Output feature map with 768 channels to match ViT embedding dimension
            nn.Conv2d(512, 768, kernel_size=3, padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        
    def forward_features(self, x):
        """Extract feature maps for ViT processing"""
        return self.features(x)  # Returns [Batch, 768, H, W]
    
    def forward(self, x):
        """Standard forward pass for standalone CNN"""
        x = self.features(x)
        # Global average pooling for classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x
class VisionTransformer(nn.Module):
    def __init__(self, num_classes=2, embed_dim=768, depth=6, heads=8, 
                 mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbedding(in_channels=768, embed_dim=embed_dim)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim,heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly"""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        
    def forward(self, x):
        # x is CNN features: [B, 768, H, W]
        B, C, H, W = x.shape
        
        # Convert to patch embeddings
        x = self.patch_embed(x)  # [B, N, embed_dim] where N = H*W
        N = x.shape[1]
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, N+1, embed_dim]
        
        # Add learnable position embeddings
        pos_embed = nn.Parameter(torch.zeros(1, N + 1, self.embed_dim)).to(x.device)
        nn.init.trunc_normal_(pos_embed, std=0.02)
        x = x + pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Classification head (use only cls token)
        cls_output = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_output)
        
        return logits
    
class CNNViTHybrid(nn.Module):
    """Properly integrated CNN-ViT Hybrid Model"""
    
    def __init__(self, num_classes=2, embed_dim=768, depth=6, heads=8):
        super().__init__()
        
        # CNN backbone for feature extraction
        self.cnn = ConvNet()
        
        # Vision Transformer for final classification
        self.vit = VisionTransformer(
            num_classes=num_classes,
            embed_dim=embed_dim, 
            depth=depth,
            heads=heads)
    
    def forward(self, x):
        # Extract CNN features first
        cnn_features = self.cnn.forward_features(x)  # [B, 768, H, W]
        
        # Process through ViT
        output = self.vit(cnn_features)
        
        return output
