import pandas as pd
import os
from utils import load_data, clean

def save_random_samples(data, folder_path, num_samples=5):
    for i in range(num_samples):
        random_row = data.sample(n=1, random_state=i)
        output_path = os.path.join(folder_path, f'ECG-patient-{i+1}.csv')
        random_row.to_csv(output_path, index=False)
        print(f"Random row saved as {output_path}")

def main():
    folder_path = os.environ.get('FOLDER_PATH')
    file_name = os.environ.get('DATASET_FILE')

    if not folder_path or not file_name:
        print("Please set the FOLDER_PATH and DATASET_FILE environment variables.")
        return

    data = load_data(folder_path, file_name)
    cleaned_data = clean()

    # Split the data to get the test set
    from sklearn.model_selection import train_test_split
    _, test_data = train_test_split(cleaned_data, test_size=0.2, stratify=cleaned_data['type'], random_state=42)

    # Remove the 'type' column from test data
    test_data = test_data.drop(columns=['type'])

    save_random_samples(test_data, folder_path)

if __name__ == "__main__":
    main()
