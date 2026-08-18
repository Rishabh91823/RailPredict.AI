import numpy as np
import pandas as pd
import time

def generate_1cr_dataset():
    print("Starting generation of 1,00,00,000 (1 Crore) record dataset...")
    start_time = time.time()
    
    np.random.seed(42)
    n_samples = 10_000_000  

    route_tier = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.3, 0.2]).astype(np.uint8)
    days_left = np.random.randint(1, 121, size=n_samples, dtype=np.int16)
    waitlist_num = np.random.randint(1, 301, size=n_samples, dtype=np.int16)
    train_type = np.random.randint(0, 3, size=n_samples, dtype=np.uint8) 
    travel_class = np.random.randint(0, 4, size=n_samples, dtype=np.uint8) 
    quota = np.random.randint(0, 2, size=n_samples, dtype=np.uint8) 
    is_festival = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]).astype(np.uint8)

   
    base_score = 92.0 - (waitlist_num * 0.32) - (days_left * 0.12)
    tier_effect = np.where(route_tier == 1, -15.0, np.where(route_tier == 2, -5.0, 5.0))
    festival_effect = np.where(is_festival == 1, -22.0, 5.0)
    quota_effect = np.where(quota == 1, -12.0, 4.0)
    noise = np.random.normal(0, 3.5, n_samples)

    final_probability = base_score + tier_effect + festival_effect + quota_effect + noise
    final_probability = np.clip(final_probability, 1.0, 99.0)
    
 
    confirmed = (final_probability > 50.0).astype(np.uint8)

    df = pd.DataFrame({
        'route_tier': route_tier,
        'days_left': days_left,
        'waitlist_num': waitlist_num,
        'train_type': train_type,
        'class': travel_class,
        'quota': quota,
        'is_festival': is_festival,
        'confirmed': confirmed
    })

    print(f"Dataset generated in {time.time() - start_time:.2f} seconds.")
    print("Saving to 'railway_dataset_1cr.csv'...")
    
    df.to_csv('railway_dataset_1cr.csv', index=False)
    print("Saved 'railway_dataset_1cr.csv' successfully!")

if __name__ == '__main__':
    generate_1cr_dataset()
