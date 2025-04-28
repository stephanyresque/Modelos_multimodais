import torch
import torch.nn as nn
import torchvision.models as models

# Encoder para Imagem
class ImageEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super(ImageEncoder, self).__init__()
        
        self.base_model = models.resnet18(weights=None)  
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, output_dim)
    
    def forward(self, x):
        return self.base_model(x)

# Encoder para Texto
class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super(TextEncoder, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        embedded = self.embedding(x)  
        embedded = embedded.permute(0, 2, 1)  
        pooled = self.pool(embedded)  
        pooled = pooled.squeeze(2)    
        return pooled
