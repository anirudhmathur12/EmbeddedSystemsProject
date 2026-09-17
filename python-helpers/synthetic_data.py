import numpy as np
import pandas as pd

def generate_data(n_samples=5000, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate base standard normal distributions for the features
    z1 = np.random.randn(n_samples)
    z2 = np.random.randn(n_samples)
    z3 = np.random.randn(n_samples)
    z4 = np.random.randn(n_samples)
    
    # 1. Define the underlying nonlinear relationship
    # A mix of circular boundaries (z1^2 + z2^2), trigonometric waviness (sin), and a linear component
    signal = (z1**2) + (z2**2) - 1.5 * np.sin(z3 * np.pi) + 0.5 * z4
    
    # 2. Add Gaussian noise to create overlap and prevent 100% accuracy
    noise = np.random.randn(n_samples) * 1.2
    noisy_signal = signal + noise
    
    # 3. Threshold the signal to create a binary target (using median to ensure rough balance)
    threshold = np.median(noisy_signal)
    target = (noisy_signal > threshold).astype(int)
    
    # 4. Map the base distributions to continuous floating-point values at different scales
    # feature_1: scale ~ [0, 1]
    f1 = np.clip(0.5 + z1 * 0.15, 0.0, 1.0)
    
    # feature_2: scale ~ [0, 100]
    f2 = np.clip(50 + z2 * 20, 0.0, 100.0)
    
    # feature_3: scale ~ [10, 50]
    f3 = np.clip(30 + z3 * 8, 10.0, 50.0)
    
    # feature_4: scale ~ [0, 1000]
    f4 = np.clip(500 + z4 * 200, 0.0, 1000.0)
    
    # Compile into a DataFrame
    df = pd.DataFrame({
        'feature_1': f1,
        'feature_2': f2,
        'feature_3': f3,
        'feature_4': f4,
        'target': target
    })
    
    # Randomize the rows
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    dataset = generate_data()
    dataset.to_csv('dataset.csv', index=False)
    print("dataset.csv generated successfully with 5,000 samples.")
