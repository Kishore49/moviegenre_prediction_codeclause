# Moviegenre_prediction_codeclause


# Movie Genre Prediction

## Overview
This project aims to predict the genres of movies based on their synopses using machine learning techniques. The dataset used contains movie synopses and their corresponding genres. The main objective is to preprocess the text data, perform feature extraction, and train a classifier to predict the genres of new movie synopses.

## Table of Contents
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Results](#results)

## Project Structure
```
├── data
│   ├── train.csv
│   ├── test.csv
├── notebooks
│   ├── Movie_Genre_Prediction.ipynb
├── src
│   ├── preprocessing.py
│   ├── training.py
│   ├── evaluation.py
├── README.md
├── requirements.txt
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/movie-genre-prediction.git
   cd movie-genre-prediction
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Data
- The dataset consists of two CSV files: `train.csv` and `test.csv`.
- `train.csv`: Contains movie synopses and their corresponding genres for training the model.
- `test.csv`: Contains movie synopses without genres for which predictions are to be made.

## Preprocessing
The text preprocessing steps include:
- Removing URLs, punctuation, and non-alphabetic characters.
- Converting text to lowercase.
- Removing stopwords.
- Lemmatizing the text to reduce words to their base form.

## Model Training
- The `Multinomial Naive Bayes` classifier is used for training.
- Text data is converted to numerical features using `TF-IDF Vectorizer`.

## Hyperparameter Tuning
- Hyperparameters for the Naive Bayes classifier are tuned using `GridSearchCV`.
- The optimal `alpha` value is selected based on cross-validation performance.

## Evaluation
- The model's performance is evaluated using accuracy and classification report metrics.
- Cross-validation is used to ensure the robustness of the model.

## Results
- The trained model is used to predict the genres of the movies in the test dataset.
- The predictions are saved in `predicted_genre.csv`.

## How to Run
1. **Run the Jupyter Notebook:**
   - Open `Movie_Genre_Prediction.ipynb` in a Jupyter Notebook environment.
   - Execute the cells sequentially to preprocess the data, train the model, tune hyperparameters, and evaluate the results.

2. **Run the Python scripts:**
   - Execute the preprocessing, training, and evaluation scripts as needed.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any feature requests, bug fixes, or improvements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
