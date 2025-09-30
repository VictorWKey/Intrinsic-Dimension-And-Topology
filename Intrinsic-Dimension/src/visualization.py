import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os


def setup_plots():
    """Configuración global para plots consistentes."""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12


def plot_reconstruction_comparison(original, reconstructed, n_samples=10, save_path=None):
    """
    Compara imágenes originales vs reconstruidas.
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 4))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(original[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstruida
        axes[1, i].imshow(reconstructed[i].cpu().numpy().reshape(28, 28), cmap='gray')
        axes[1, i].set_title('Reconstruida')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_error_vs_dimension(results_dict, save_path=None):
    """
    Gráfica de error de reconstrucción vs dimensión latente.
    """
    latent_dims = list(results_dict.keys())
    errors = [results_dict[dim]['error'] for dim in latent_dims]
    
    plt.figure(figsize=(10, 6))
    plt.plot(latent_dims, errors, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Dimensión Latente')
    plt.ylabel('Error de Reconstrucción (MSE)')
    plt.title('Error de Reconstrucción vs Dimensión del Espacio Latente')
    plt.grid(True, alpha=0.3)
    
    # Marcar el punto con menor error
    min_error_idx = np.argmin(errors)
    plt.plot(latent_dims[min_error_idx], errors[min_error_idx], 'ro', markersize=12, 
             label=f'Menor error: dim={latent_dims[min_error_idx]}')
    
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_elbow_analysis(results_dict, save_path=None):
    """
    Visualiza el análisis del método del codo.
    """
    latent_dims = list(results_dict.keys())
    errors = [results_dict[dim]['error'] for dim in latent_dims]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Gráfica principal
    ax1.plot(latent_dims, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Dimensión Latente')
    ax1.set_ylabel('Error de Reconstrucción')
    ax1.set_title('Método del Codo - Error vs Dimensión')
    ax1.grid(True, alpha=0.3)
    
    # Segunda derivada para encontrar el codo
    if len(errors) >= 3:
        first_diff = np.diff(errors)
        second_diff = np.diff(first_diff)
        
        ax2.plot(latent_dims[2:], second_diff, 'ro-', linewidth=2)
        ax2.set_xlabel('Dimensión Latente')
        ax2.set_ylabel('Segunda Derivada del Error')
        ax2.set_title('Segunda Derivada - Detección del Codo')
        ax2.grid(True, alpha=0.3)
        
        # Marcar el codo
        elbow_idx = np.argmin(second_diff)
        elbow_dim = latent_dims[elbow_idx + 2]
        ax1.axvline(x=elbow_dim, color='red', linestyle='--', alpha=0.7, 
                   label=f'Codo detectado: {elbow_dim}')
        ax1.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_latent_distribution(latent_representations, labels=None, method='tsne', save_path=None):
    """
    Visualiza la distribución del espacio latente en 2D.
    """
    latent_data = latent_representations.cpu().numpy()
    
    if latent_data.shape[1] > 2:
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:  # PCA
            reducer = PCA(n_components=2)
        
        latent_2d = reducer.fit_transform(latent_data)
    else:
        latent_2d = latent_data
    
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], 
                            c=labels.cpu().numpy(), cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'Distribución del Espacio Latente ({method.upper()}) - Coloreado por Clases')
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
        plt.title(f'Distribución del Espacio Latente ({method.upper()})')
    
    plt.xlabel(f'{method.upper()} Componente 1')
    plt.ylabel(f'{method.upper()} Componente 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_compression_efficiency(results_dict, save_path=None):
    """
    Visualiza la eficiencia de compresión vs calidad.
    """
    latent_dims = list(results_dict.keys())
    errors = [results_dict[dim]['error'] for dim in latent_dims]
    compression_ratios = [784 / dim for dim in latent_dims]  # MNIST: 28x28 = 784
    
    # Eficiencia = Ratio de compresión / (1 + error normalizado)
    normalized_errors = np.array(errors) / max(errors)
    efficiency = np.array(compression_ratios) / (1 + normalized_errors)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot: Compresión vs Error
    scatter = ax1.scatter(compression_ratios, errors, c=latent_dims, 
                         cmap='viridis', s=100, alpha=0.7)
    ax1.set_xlabel('Ratio de Compresión (Original/Latente)')
    ax1.set_ylabel('Error de Reconstrucción')
    ax1.set_title('Compresión vs Calidad de Reconstrucción')
    ax1.grid(True, alpha=0.3)
    
    # Añadir etiquetas con dimensiones
    for i, dim in enumerate(latent_dims):
        ax1.annotate(f'{dim}', (compression_ratios[i], errors[i]), 
                    xytext=(5, 5), textcoords='offset points')
    
    # Eficiencia vs Dimensión
    ax2.plot(latent_dims, efficiency, 'go-', linewidth=2, markersize=8)
    ax2.set_xlabel('Dimensión Latente')
    ax2.set_ylabel('Eficiencia de Compresión')
    ax2.set_title('Eficiencia de Compresión por Dimensión')
    ax2.grid(True, alpha=0.3)
    
    # Marcar la dimensión más eficiente
    max_eff_idx = np.argmax(efficiency)
    ax2.plot(latent_dims[max_eff_idx], efficiency[max_eff_idx], 'ro', 
            markersize=12, label=f'Máxima eficiencia: dim={latent_dims[max_eff_idx]}')
    ax2.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_summary_table(analysis_results, save_path=None):
    """
    Crea una tabla resumen con todas las métricas de evaluación.
    """
    data = {
        'Método': ['Método del Codo', 'Rate-Distortion', 'Umbral de Estabilidad', 
                  'Máxima Eficiencia', 'Recomendación Final'],
        'Dimensión Sugerida': [
            analysis_results['elbow_method'],
            analysis_results['rate_distortion'],
            analysis_results['stability_threshold'],
            analysis_results['best_compression_efficiency'],
            analysis_results['recommended_dimension']
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Crear tabla visual
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)
    
    # Destacar la recomendación final
    table[(len(df), 0)].set_facecolor('#FFE6E6')
    table[(len(df), 1)].set_facecolor('#FFE6E6')
    
    plt.title('Resumen de Análisis de Dimensión Intrínseca', 
              fontsize=16, fontweight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return df


def plot_training_progress(train_losses, val_losses=None, save_path=None):
    """
    Visualiza el progreso del entrenamiento.
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Loss de Entrenamiento', linewidth=2)
    
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', label='Loss de Validación', linewidth=2)
    
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Progreso del Entrenamiento del Autoencoder')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_comprehensive_report(results_dict, analysis_results, save_dir):
    """
    Genera un reporte visual completo con todas las gráficas.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Configurar estilo
    setup_plots()
    
    # 1. Error vs Dimensión
    plot_error_vs_dimension(results_dict, 
                           save_path=os.path.join(save_dir, 'error_vs_dimension.png'))
    
    # 2. Análisis del codo
    plot_elbow_analysis(results_dict, 
                       save_path=os.path.join(save_dir, 'elbow_analysis.png'))
    
    # 3. Eficiencia de compresión
    plot_compression_efficiency(results_dict, 
                               save_path=os.path.join(save_dir, 'compression_efficiency.png'))
    
    # 4. Tabla resumen
    df_summary = create_summary_table(analysis_results, 
                                     save_path=os.path.join(save_dir, 'summary_table.png'))
    
    # Guardar tabla como CSV también
    df_summary.to_csv(os.path.join(save_dir, 'summary_table.csv'), index=False)
    
    return df_summary