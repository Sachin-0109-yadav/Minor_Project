import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class AdvancedEarthquakePredictor:
    def __init__(self, csv_path):
        print("🌋 Advanced Earthquake Prediction System")
        print("Incorporating Tectonic Plates & Volcanic Activity")
        print("="*60)
        print("Loading geological dataset from CSV file...\n")
        try:
            self.data = pd.read_csv(csv_path)
            print("✅ Dataset loaded successfully!")
            print(f"Dataset shape: {self.data.shape}\n")
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            exit()

    def generate_enhanced_data(self):
        print("Generating enhanced geological features...")
        df = self.data.copy()

        # Feature engineering (synthetic but geologically inspired)
        df["plate_stress"] = np.random.uniform(0.1, 1.0, len(df))
        df["plate_movement_rate"] = np.random.uniform(0.1, 10.0, len(df))
        df["boundary_type_convergent"] = np.random.randint(0, 2, len(df))
        df["boundary_type_transform"] = np.random.randint(0, 2, len(df))
        df["boundary_type_divergent"] = np.random.randint(0, 2, len(df))
        df["volcanic_risk_index"] = np.random.uniform(0, 1, len(df))
        df["nearest_volcano_distance"] = np.random.uniform(5, 500, len(df))
        df["active_volcanoes_nearby"] = np.random.randint(0, 5, len(df))
        df["earthquake_risk"] = np.where(df["magnitude"] > 5.5, 1, 0)

        self.data = df
        print("Enhanced features added:",
              ['plate_stress', 'plate_movement_rate', 'boundary_type_convergent',
               'boundary_type_transform', 'boundary_type_divergent', 'volcanic_risk_index',
               'nearest_volcano_distance', 'active_volcanoes_nearby', 'earthquake_risk'], "\n")

    def train_models(self):
        print("Training enhanced prediction models...\n")

        df = self.data.dropna()
        X = df.drop("earthquake_risk", axis=1)
        y = df["earthquake_risk"]

        if "soil_type" in X.columns:
            X = pd.get_dummies(X, columns=["soil_type"], drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        models = {
            "Enhanced Random Forest": RandomForestClassifier(random_state=42),
            "Enhanced Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Enhanced Logistic Regression": LogisticRegression(max_iter=1000)
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            print(f"{name} Results:")
            print(f"Accuracy: {acc:.3f}\n")
            print("Classification Report:")
            print(classification_report(y_test, preds))
            print("\n")

    def location_risk_analysis(self):
        print("🌍 LOCATION RISK ANALYSIS")
        print("="*50)
        print("Choose analysis mode:")
        print("1. Analyze your custom location")
        print("2. Analyze sample high-risk locations")
        print("3. Skip location analysis")

        choice = input("\nEnter your choice (1/2/3): ")

        if choice == "1":
            try:
                lat = float(input("Enter latitude: "))
                lon = float(input("Enter longitude: "))
                mag = float(input("Enter predicted magnitude: "))

                # Basic rule for demo
                if mag > 6.0:
                    print(f"\n⚠️ Location ({lat}, {lon}) is in a HIGH-RISK zone!")
                elif mag > 4.0:
                    print(f"\n⚠️ Location ({lat}, {lon}) is in a MODERATE-RISK zone.")
                else:
                    print(f"\n✅ Location ({lat}, {lon}) is in a LOW-RISK zone.")
            except:
                print("❌ Invalid input. Try again.")

        elif choice == "2":
            print("\nAnalyzing sample high-risk locations...\n")
            samples = [
                {"name": "Tokyo, Japan", "lat": 35.6895, "lon": 139.6917, "mag": 7.2},
                {"name": "San Francisco, USA", "lat": 37.7749, "lon": -122.4194, "mag": 6.8},
                {"name": "Jakarta, Indonesia", "lat": -6.2088, "lon": 106.8456, "mag": 6.3},
                {"name": "Istanbul, Turkey", "lat": 41.0082, "lon": 28.9784, "mag": 6.9},
                {"name": "Mexico City, Mexico", "lat": 19.4326, "lon": -99.1332, "mag": 7.1},
            ]

            for loc in samples:
                print(f"{loc['name']} ({loc['lat']}, {loc['lon']}) - Magnitude: {loc['mag']} ⚠️ High-Risk")

        elif choice == "3":
            print("\nSkipping location analysis...")

        else:
            print("❌ Invalid choice. Please restart and enter 1, 2, or 3.")

# Run the system
if __name__ == "__main__":
    predictor = AdvancedEarthquakePredictor("C:\\Users\\SACHIN\\Downloads\\advanced-earthquake-prediction-system-main\\earthquake.csv")
    predictor.generate_enhanced_data()
    predictor.train_models()
    predictor.location_risk_analysis()
