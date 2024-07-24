import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta

class FishingPredictor:
    def __init__(self):
        self.data = None
        self.model = None

    def generate_sample_data(self, num_samples=10000):
        # Generate fictional but reasonable data
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', end='2024-07-24', periods=num_samples)
        latitudes = np.random.uniform(-60, 90, num_samples)
        longitudes = np.random.uniform(-180, 180, num_samples)
        species = np.random.choice(['cod', 'tuna', 'salmon', 'mackerel'], num_samples)
        catch_amount = np.random.exponential(scale=100, size=num_samples)
        area_size = np.random.uniform(10, 1000, num_samples)
        water_temp = np.random.normal(15, 5, num_samples)
        
        self.data = pd.DataFrame({
            'date': dates,
            'latitude': latitudes,
            'longitude': longitudes,
            'species': species,
            'catch_amount': catch_amount,
            'area_size': area_size,
            'water_temp': water_temp
        })

    def preprocess_data(self):
        # Convert date to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        
        # Extract month as a feature
        self.data['month'] = self.data['date'].dt.month
        
        # Encode categorical variables
        self.data = pd.get_dummies(self.data, columns=['species'])

        print("Data preprocessed successfully.")

    def engineer_features(self):
        # Calculate fishing pressure (example feature)
        self.data['fishing_pressure'] = self.data['catch_amount'] / self.data['area_size']
        
        # Create target variable based on fishing pressure and water temperature
        pressure_threshold = self.data['fishing_pressure'].quantile(0.7)
        temp_threshold = self.data['water_temp'].quantile(0.3)
        
        conditions = [
            (self.data['fishing_pressure'] > pressure_threshold) & (self.data['water_temp'] < temp_threshold),
            (self.data['fishing_pressure'] > pressure_threshold) | (self.data['water_temp'] < temp_threshold),
            (self.data['fishing_pressure'] <= pressure_threshold) & (self.data['water_temp'] >= temp_threshold)
        ]
        choices = [0, 1, 2]  # 0: Prohibited, 1: Limited, 2: Permitted
        
        self.data['fishing_status'] = np.select(conditions, choices, default=2)

        print("Features engineered successfully.")

    def train_model(self):
        # Prepare features and target
        features = ['latitude', 'longitude', 'month', 'fishing_pressure', 'water_temp'] + [col for col in self.data.columns if col.startswith('species_')]
        X = self.data[features]
        y = self.data['fishing_status']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained successfully. Accuracy: {accuracy:.2f}")
        print(classification_report(y_test, y_pred))

    def predict_fishing_status(self, new_data):
        # Make predictions on new data
        predictions = self.model.predict(new_data)
        return predictions

class FishingMapGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("OceanGuardian: Fishing Permission Map")
        self.master.geometry("1000x800")

        self.predictor = FishingPredictor()
        self.predictor.generate_sample_data()
        self.predictor.preprocess_data()
        self.predictor.engineer_features()
        self.predictor.train_model()
        
        self.create_widgets()

    def create_widgets(self):
        self.frame = ttk.Frame(self.master, padding="10")
        self.frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.create_map()

        update_button = ttk.Button(self.frame, text="Update Map", command=self.update_map)
        update_button.grid(row=1, column=0, pady=10)

    def create_map(self):
        fig, self.ax = plt.subplots(figsize=(12, 8))
        canvas = FigureCanvasTkAgg(fig, master=self.frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0)

        self.m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90, 
                         llcrnrlon=-180, urcrnrlon=180, resolution='c', ax=self.ax)
        self.m.drawcoastlines()
        self.m.fillcontinents(color='coral', lake_color='aqua')
        self.m.drawmapboundary(fill_color='aqua')

        self.scatter = None

    def update_map(self):
        if self.scatter:
            self.scatter.remove()

        # Generate grid of lat/lon points
        lats = np.linspace(-60, 90, 100)
        lons = np.linspace(-180, 180, 100)
        lat_grid, lon_grid = np.meshgrid(lats, lons)

        # Prepare data for prediction
        current_date = datetime.now()
        prediction_data = pd.DataFrame({
            'latitude': lat_grid.flatten(),
            'longitude': lon_grid.flatten(),
            'month': np.full(lat_grid.size, current_date.month),
            'fishing_pressure': np.random.uniform(0, 200, lat_grid.size),
            'water_temp': np.random.normal(15, 5, lat_grid.size),
            'species_cod': np.random.randint(0, 2, lat_grid.size),
            'species_tuna': np.random.randint(0, 2, lat_grid.size),
            'species_salmon': np.random.randint(0, 2, lat_grid.size),
            'species_mackerel': np.random.randint(0, 2, lat_grid.size)
        })

        # Get predictions
        predictions = self.predictor.predict_fishing_status(prediction_data)
        predictions = predictions.reshape(lat_grid.shape)

        x, y = self.m(lon_grid, lat_grid)

        self.scatter = self.m.pcolormesh(x, y, predictions, cmap='RdYlGn', alpha=0.6, vmin=0, vmax=2)
        
        cbar = self.m.colorbar(self.scatter, location='bottom', pad="10%")
        cbar.set_ticks([0.33, 1, 1.67])
        cbar.set_ticklabels(['Prohibited', 'Limited', 'Permitted'])

        self.ax.set_title(f'Fishing Permission Map (Updated: {current_date.strftime("%Y-%m-%d")})')
        self.ax.figure.canvas.draw()

    def run(self):
        self.master.mainloop()

# Usage
if __name__ == "__main__":
    root = tk.Tk()
    app = FishingMapGUI(root)
    app.run()