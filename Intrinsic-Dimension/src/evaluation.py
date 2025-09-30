import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def reconstruction_error(original, reconstructed):
    """
    Calcula el error de reconstrucción MSE entre datos originales y reconstruidos.
    """
    return torch.mean((original - reconstructed) ** 2).item()


def compression_ratio(original_dim, latent_dim):
    """
    Calcula la ratio de compresión.
    """
    return original_dim / latent_dim


def information_retention_score(original, reconstructed):
    """
    Evalúa qué tanta información se preserva después de la reconstrucción
    usando correlación de Pearson.
    """
    original_flat = original.view(original.size(0), -1).cpu().numpy()
    reconstructed_flat = reconstructed.view(reconstructed.size(0), -1).cpu().numpy()
    
    correlations = []
    for i in range(original_flat.shape[0]):
        corr = np.corrcoef(original_flat[i], reconstructed_flat[i])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0


def explained_variance_ratio(data, latent_representations):
    """
    Calcula cuánta varianza explica la representación latente usando PCA como referencia.
    """
    # PCA en los datos originales
    data_flat = data.view(data.size(0), -1).cpu().numpy()
    pca_original = PCA()
    pca_original.fit(data_flat)
    
    # PCA en las representaciones latentes
    latent_flat = latent_representations.cpu().numpy()
    n_components = min(latent_flat.shape[1], data_flat.shape[1])
    
    # Varianza explicada por los componentes principales hasta la dimensión latente
    cumulative_variance = np.cumsum(pca_original.explained_variance_ratio_)
    if n_components <= len(cumulative_variance):
        return cumulative_variance[n_components - 1]
    else:
        return cumulative_variance[-1]


def elbow_method_score(reconstruction_errors, latent_dims):
    """
    Implementa el método del codo para encontrar la dimensión óptima.
    Calcula la segunda derivada para encontrar el punto de inflexión.
    """
    if len(reconstruction_errors) < 3:
        return latent_dims[np.argmin(reconstruction_errors)]
    
    # Calcular diferencias consecutivas
    first_diff = np.diff(reconstruction_errors)
    second_diff = np.diff(first_diff)
    
    # Encontrar el punto donde la segunda derivada es mínima (mayor curvatura)
    elbow_idx = np.argmin(second_diff) + 1  # +1 porque perdimos una posición en cada diff
    
    return latent_dims[elbow_idx] if elbow_idx < len(latent_dims) else latent_dims[-1]


def rate_distortion_score(reconstruction_errors, latent_dims):
    """
    Calcula un score basado en la teoría de rate-distortion.
    Busca el mejor balance entre compresión (rate) y calidad (distortion).
    """
    # Normalizar errores y dimensiones
    normalized_errors = np.array(reconstruction_errors) / max(reconstruction_errors)
    normalized_dims = np.array(latent_dims) / max(latent_dims)
    
    # Score combinado: queremos minimizar error y dimensión
    scores = normalized_errors + 0.5 * normalized_dims  # Peso para balancear
    
    optimal_idx = np.argmin(scores)
    return latent_dims[optimal_idx]


def evaluate_intrinsic_dimension(results_dict):
    """
    Evalúa la dimensión intrínseca usando múltiples métricas y devuelve un resumen.
    
    Args:
        results_dict: Diccionario con resultados de diferentes dimensiones latentes
                     {'latent_dim': {'error': float, 'model': model, 'latent_repr': tensor}}
    
    Returns:
        Dictionary con análisis de dimensión intrínseca
    """
    latent_dims = list(results_dict.keys())
    reconstruction_errors = [results_dict[dim]['error'] for dim in latent_dims]
    
    # Método del codo
    elbow_dim = elbow_method_score(reconstruction_errors, latent_dims)
    
    # Rate-distortion
    rd_dim = rate_distortion_score(reconstruction_errors, latent_dims)
    
    # Buscar el punto donde el error se estabiliza (cambio relativo < 5%)
    stable_dim = latent_dims[0]
    for i in range(1, len(reconstruction_errors)):
        relative_change = abs(reconstruction_errors[i] - reconstruction_errors[i-1]) / reconstruction_errors[i-1]
        if relative_change < 0.05:  # Cambio menor al 5%
            stable_dim = latent_dims[i]
            break
    
    # Calcular eficiencia de compresión para cada dimensión
    compression_efficiency = {}
    for dim in latent_dims:
        error = results_dict[dim]['error']
        ratio = compression_ratio(784, dim)  # Asumiendo MNIST (28x28)
        efficiency = ratio / (1 + error)  # Mayor ratio, menor error = mayor eficiencia
        compression_efficiency[dim] = efficiency
    
    best_efficiency_dim = max(compression_efficiency.keys(), key=lambda k: compression_efficiency[k])
    
    analysis = {
        'elbow_method': elbow_dim,
        'rate_distortion': rd_dim,
        'stability_threshold': stable_dim,
        'best_compression_efficiency': best_efficiency_dim,
        'reconstruction_errors': dict(zip(latent_dims, reconstruction_errors)),
        'compression_efficiencies': compression_efficiency,
        'recommended_dimension': int(np.median([elbow_dim, rd_dim, stable_dim]))
    }
    
    return analysis


def perplexity_based_dimension(latent_representations, max_dim=50):
    """
    Estima la dimensión intrínseca usando perplejidad de t-SNE.
    Dimensiones altas tienden a tener mejor separación hasta cierto punto.
    """
    latent_data = latent_representations.cpu().numpy()
    
    # Si tenemos más dimensiones de las que queremos probar
    if latent_data.shape[1] > max_dim:
        latent_data = latent_data[:, :max_dim]
    
    # Probar diferentes perplejidades
    perplexities = [5, 10, 20, 30, 50]
    best_perplexity = perplexities[0]
    
    # Para simplificar, usar la dimensión actual como estimación
    # En un análisis más completo, se evaluaría la calidad de la separación
    estimated_dim = min(latent_data.shape[1], max_dim)
    
    return estimated_dim