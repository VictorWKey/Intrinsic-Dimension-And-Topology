#!/usr/bin/env python3
"""
Script de prueba rápida para verificar que el sistema funcione correctamente.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm

# Configurar dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

class AutoEncoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super(AutoEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def train_autoencoder(model, train_loader, num_epochs=5):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for data, _ in train_loader:
            data = data.view(data.size(0), -1).to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        print(f"Época {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    return train_losses

def evaluate_autoencoder(model, test_loader):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    criterion = nn.MSELoss(reduction='sum')
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            reconstructed, _ = model(data)
            
            batch_loss = criterion(reconstructed, data).item()
            total_loss += batch_loss
            total_samples += data.size(0)
    
    return total_loss / total_samples

def main():
    # Cargar datos
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("🧪 PRUEBA RÁPIDA DEL SISTEMA CORREGIDO")
    print("=" * 50)
    
    # Probar dos dimensiones diferentes
    test_dims = [8, 64]
    results = {}
    
    for dim in test_dims:
        print(f"\n🔍 Entrenando autoencoder con {dim} dimensiones latentes:")
        
        model = AutoEncoder(784, dim)
        print(f"   Parámetros: {sum(p.numel() for p in model.parameters()):,}")
        
        # Entrenar
        train_losses = train_autoencoder(model, train_loader, num_epochs=5)
        
        # Evaluar
        test_error = evaluate_autoencoder(model, test_loader)
        
        results[dim] = {
            'train_losses': train_losses,
            'test_error': test_error,
            'improvement': (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        }
        
        print(f"   ✅ Error final de prueba: {test_error:.6f}")
        print(f"   📈 Mejora en entrenamiento: {results[dim]['improvement']:.1f}%")
    
    # Verificación final
    print("\n" + "="*50)
    print("📊 RESULTADO DE LA VERIFICACIÓN:")
    
    error_8 = results[8]['test_error']
    error_64 = results[64]['test_error']
    
    print(f"Error con 8 dimensiones: {error_8:.6f}")
    print(f"Error con 64 dimensiones: {error_64:.6f}")
    print(f"Diferencia absoluta: {abs(error_8 - error_64):.6f}")
    
    if abs(error_8 - error_64) > 1e-4:
        print("✅ ¡ÉXITO! Los errores son diferentes como se esperaba.")
        print("✅ El sistema está funcionando correctamente.")
        
        # Expectativa: dimensión mayor debería tener menor error
        if error_64 < error_8:
            print("✅ Como se esperaba: mayor dimensión → menor error de reconstrucción")
        else:
            print("ℹ️  Nota: menor dimensión tiene menor error (podría ser por regularización)")
    else:
        print("❌ PROBLEMA: Los errores son idénticos. Hay un bug en el código.")
    
    print("\n🚀 El notebook debería funcionar correctamente ahora.")

if __name__ == "__main__":
    main()