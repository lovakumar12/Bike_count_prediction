# Bike Rental Prediction System

## Overview

The **Bike Rental Prediction System** is a machine learning project that predicts bike rental demand based on historical data. The system uses various environmental and seasonal factors to estimate the number of bikes that will be rented on a given day. The goal is to help bike rental services optimize availability and improve customer satisfaction.

This project is deployed as a Streamlit web application. You can access the live deployment here:

🔗 **[Live Deployment](https://bikecountprediction-4rwwuxetyvdcgdkhvvhxru.streamlit.app/)**

---

## Project Structure

The project follows a modular structure for ease of maintenance and scalability:

```
Bike_Rental_Prediction/
├── src/
│   └── mlProject/
│       ├── __init__.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   └── model_trainer.py
│       ├── config/
│       │   ├── __init__.py
│       │   └── configuration.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── prediction.py
│       │   └── training.py
│       ├── entity/
│       │   ├── __init__.py
│       │   └── config_entity.py
│       ├── constants/
│       │   └── __init__.py
│       └── utils/
│           ├── __init__.py
│           └── common.py
├── config/
│   └── config.yaml
├── params.yaml
├── schema.yaml
├── app.py
├── main.py
├── requirements.txt
├── setup.py
├── research/
│   └── trials.ipynb
└── templates/
    └── index.html
```

---

## Features

- **Data Ingestion**: Automatically loads and preprocesses the input dataset.
- **Data Transformation**: Cleans the data and applies necessary transformations.
- **Model Training**: Trains a regression model to predict bike rentals.
- **Prediction Pipeline**: Generates predictions based on user inputs.
- **Web Interface**: Interactive Streamlit app for easy model interaction.

---

## Installation

### Prerequisites

- Python 3.8+
- Git

### Clone the Repository

```bash
git clone https://github.com/yourusername/bike_rental_prediction.git
cd bike_rental_prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Run the Application

To launch the Streamlit app locally, use the following command:

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Configuration

- **`config/config.yaml`**: Contains configuration settings for data ingestion and model training.
- **`params.yaml`**: Stores model parameters and hyperparameters.

---

## Dataset

The dataset used in this project is `SeoulBikeData.csv`, which contains the following features:

- **Date**
- **Temperature**
- **Humidity**
- **Wind speed**
- **Visibility**
- **Dew point temperature**
- **Solar radiation**
- **Rainfall**
- **Snowfall**
- **Seasons**
- **Holiday**
- **Functioning day**

---

## Deployment

The project is deployed using Streamlit. You can access the live app at:

🔗 **[Bike Rental Prediction App](https://bikecountprediction-4rwwuxetyvdcgdkhvvhxru.streamlit.app/)**

---

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Add new feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact

For any inquiries or feedback, please reach out to:

📧 **your-email@example.com**
