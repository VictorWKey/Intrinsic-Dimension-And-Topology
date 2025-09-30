import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
from tqdm import tqdm

# Importar módulos locales
from src.models import AutoEncoder, VariationalAutoEncoder, vae_loss
from src.evaluation import (
    reconstruction_error, evaluate_intrinsic_dimension,
    information_retention_score, explained_variance_ratio
)
from src.visualization import (
    setup_plots, plot_reconstruction_comparison, create_comprehensive_report,
    plot_latent_distribution, plot_training_progress
)


class IntrinsicDimensionFinder:
    """
    Clase principal para encontrar la dimensión intrínseca usando autoencoders.
    """
    
    def __init__(self, data_dir='./data', results_dir='./results', device=None):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Crear directorios
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Configurar transformaciones para MNIST
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalizar a [-1, 1]
        ])
        
        # Cargar datos
        self.train_loader, self.test_loader = self._load_data()
        
    def _load_data(self):
        """Carga el dataset MNIST."""
        train_dataset = datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        test_dataset = datasets.MNIST(
            root=self.data_dir, train=False, transform=self.transform
        )
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        return train_loader, test_loader
    
    def train_autoencoder(self, model, num_epochs=50, learning_rate=1e-3):
        """
        Entrena un autoencoder estándar.
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        train_losses = []
        
        for epoch in tqdm(range(num_epochs), desc="Entrenando Autoencoder"):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.view(data.size(0), -1).to(self.device)
                
                optimizer.zero_grad()
                reconstructed, _ = model(data)
                loss = criterion(reconstructed, data)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Época [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        
        return train_losses
    
    def train_vae(self, model, num_epochs=50, learning_rate=1e-3, beta=1.0):
        """
        Entrena un Variational Autoencoder.
        """
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        
        for epoch in tqdm(range(num_epochs), desc="Entrenando VAE"):
            model.train()
            epoch_loss = 0.0
            
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.view(data.size(0), -1).to(self.device)
                
                optimizer.zero_grad()
                reconstructed, mu, logvar, _ = model(data)
                loss = vae_loss(reconstructed, data, mu, logvar, beta)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(self.train_loader)
            train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f'Época [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
        
        return train_losses
    
    def evaluate_model(self, model, model_type='autoencoder'):
        """
        Evalúa un modelo entrenado.
        """
        model.eval()
        total_error = 0.0
        latent_representations = []
        original_data = []
        reconstructed_data = []
        labels = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data = data.view(data.size(0), -1).to(self.device)
                
                if model_type == 'vae':
                    reconstructed, mu, logvar, latent = model(data)
                    latent_representations.append(mu)  # Usar mu para VAE
                else:
                    reconstructed, latent = model(data)
                    latent_representations.append(latent)
                
                original_data.append(data.cpu())
                reconstructed_data.append(reconstructed.cpu())
                labels.append(target)
                
                # Calcular error de reconstrucción
                error = reconstruction_error(data, reconstructed)
                total_error += error
        
        avg_error = total_error / len(self.test_loader)
        
        # Concatenar todos los datos
        latent_representations = torch.cat(latent_representations, dim=0)
        original_data = torch.cat(original_data, dim=0)
        reconstructed_data = torch.cat(reconstructed_data, dim=0)
        labels = torch.cat(labels, dim=0)
        
        return {
            'error': avg_error,
            'latent_repr': latent_representations,
            'original': original_data,
            'reconstructed': reconstructed_data,
            'labels': labels
        }
    
    def find_intrinsic_dimension(self, latent_dims=[2, 4, 8, 16, 32, 64, 128], 
                                num_epochs=30, model_type='autoencoder'):
        """
        Encuentra la dimensión intrínseca probando diferentes dimensiones latentes.
        """
        results = {}
        
        print(f"Buscando dimensión intrínseca con {model_type.upper()}...")
        print(f"Dimensiones a probar: {latent_dims}")
        
        for latent_dim in latent_dims:
            print(f"\n--- Probando dimensión latente: {latent_dim} ---")
            
            # Crear modelo
            if model_type == 'vae':
                model = VariationalAutoEncoder(input_dim=784, latent_dim=latent_dim)
                train_losses = self.train_vae(model, num_epochs)
            else:
                model = AutoEncoder(input_dim=784, latent_dim=latent_dim)
                train_losses = self.train_autoencoder(model, num_epochs)
            
            # Evaluar modelo
            eval_results = self.evaluate_model(model, model_type)
            eval_results['model'] = model
            eval_results['train_losses'] = train_losses
            
            results[latent_dim] = eval_results
            
            print(f"Error de reconstrucción: {eval_results['error']:.6f}")
        
        return results
    
    def generate_comprehensive_analysis(self, results, model_type='autoencoder'):
        """
        Genera un análisis completo y visualizaciones.
        """
        print("\n=== ANÁLISIS DE DIMENSIÓN INTRÍNSECA ===")
        
        # Análisis cuantitativo
        analysis = evaluate_intrinsic_dimension(results)
        
        print(f"\nResultados del análisis:")
        print(f"Método del codo: {analysis['elbow_method']} dimensiones")
        print(f"Rate-distortion: {analysis['rate_distortion']} dimensiones")
        print(f"Umbral de estabilidad: {analysis['stability_threshold']} dimensiones")
        print(f"Mejor eficiencia: {analysis['best_compression_efficiency']} dimensiones")
        print(f"RECOMENDACIÓN FINAL: {analysis['recommended_dimension']} dimensiones")
        
        # Crear directorio para resultados específicos
        model_results_dir = os.path.join(self.results_dir, f'{model_type}_results')
        os.makedirs(model_results_dir, exist_ok=True)
        
        # Generar reporte visual completo
        summary_df = create_comprehensive_report(results, analysis, model_results_dir)
        
        # Visualizaciones adicionales con la dimensión recomendada
        recommended_dim = analysis['recommended_dimension']
        if recommended_dim in results:
            recommended_results = results[recommended_dim]
            
            # Comparación de reconstrucción
            plot_reconstruction_comparison(
                recommended_results['original'][:10],
                recommended_results['reconstructed'][:10],
                save_path=os.path.join(model_results_dir, 'reconstruction_comparison.png')
            )
            
            # Distribución del espacio latente
            plot_latent_distribution(
                recommended_results['latent_repr'][:1000],  # Subset para visualización
                labels=recommended_results['labels'][:1000],
                method='tsne',
                save_path=os.path.join(model_results_dir, 'latent_distribution_tsne.png')
            )
            
            plot_latent_distribution(
                recommended_results['latent_repr'][:1000],
                labels=recommended_results['labels'][:1000],
                method='pca',
                save_path=os.path.join(model_results_dir, 'latent_distribution_pca.png')
            )
            
            # Progreso de entrenamiento
            plot_training_progress(
                recommended_results['train_losses'],
                save_path=os.path.join(model_results_dir, 'training_progress.png')
            )
        
        return analysis, summary_df


def main():
    """Función principal que ejecuta todo el análisis."""
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Inicializar buscador de dimensión intrínseca
    finder = IntrinsicDimensionFinder(device=device)
    
    # Configurar dimensiones a probar
    latent_dimensions = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    
    # === ANÁLISIS CON AUTOENCODER ESTÁNDAR ===
    print("\n" + "="*60)
    print("ANÁLISIS CON AUTOENCODER ESTÁNDAR")
    print("="*60)
    
    ae_results = finder.find_intrinsic_dimension(
        latent_dims=latent_dimensions,
        num_epochs=25,
        model_type='autoencoder'
    )
    
    ae_analysis, ae_summary = finder.generate_comprehensive_analysis(
        ae_results, 'autoencoder'
    )
    
    # === ANÁLISIS CON VAE (OPCIONAL) ===
    print("\n" + "="*60)
    print("ANÁLISIS CON VARIATIONAL AUTOENCODER")
    print("="*60)
    
    vae_results = finder.find_intrinsic_dimension(
        latent_dims=latent_dimensions,
        num_epochs=25,
        model_type='vae'
    )
    
    vae_analysis, vae_summary = finder.generate_comprehensive_analysis(
        vae_results, 'vae'
    )
    
    # === COMPARACIÓN FINAL ===
    print("\n" + "="*60)
    print("COMPARACIÓN FINAL")
    print("="*60)
    
    print(f"Autoencoder estándar - Dimensión recomendada: {ae_analysis['recommended_dimension']}")
    print(f"VAE - Dimensión recomendada: {vae_analysis['recommended_dimension']}")
    
    # Determinar consenso
    consensus_dim = int(np.median([
        ae_analysis['recommended_dimension'],
        vae_analysis['recommended_dimension']
    ]))
    
    print(f"\nCONCLUSIÓN: La dimensión intrínseca estimada es {consensus_dim} dimensiones")
    print(f"Esto representa una compresión de {784/consensus_dim:.1f}x desde las 784 dimensiones originales")
    
    # Guardar resultados finales
    final_results = {
        'autoencoder_analysis': ae_analysis,
        'vae_analysis': vae_analysis,
        'consensus_dimension': consensus_dim,
        'compression_ratio': 784/consensus_dim
    }
    
    # Guardar como archivo de texto para referencia
    with open(os.path.join(finder.results_dir, 'final_results.txt'), 'w') as f:
        f.write("ANÁLISIS DE DIMENSIÓN INTRÍNSECA - RESULTADOS FINALES\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset: MNIST (28x28 = 784 dimensiones)\n")
        f.write(f"Modelos evaluados: Autoencoder estándar, VAE\n")
        f.write(f"Dimensiones probadas: {latent_dimensions}\n\n")
        f.write(f"Autoencoder - Dimensión recomendada: {ae_analysis['recommended_dimension']}\n")
        f.write(f"VAE - Dimensión recomendada: {vae_analysis['recommended_dimension']}\n")
        f.write(f"CONSENSO: {consensus_dim} dimensiones\n")
        f.write(f"Ratio de compresión: {784/consensus_dim:.1f}x\n")
    
    print(f"\nTodos los resultados guardados en: {finder.results_dir}")
    print("¡Análisis completo!")


if __name__ == "__main__":
    main()