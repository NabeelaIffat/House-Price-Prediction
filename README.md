# House Price Prediction Using Regression Algorithms

A beginner-friendly machine learning project using the Kaggle House Prices dataset. It compares Linear Regression, Decision Tree Regression, and Random Forest Regression, saves the best model, and provides a Streamlit web interface.

## 1. Dataset

Download `train.csv` from the Kaggle competition:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

Create a folder named `data` in this project and place `train.csv` inside it:

```text
house_price_prediction_project/
├── data/
│   └── train.csv
├── app.py
├── train_model.py
└── requirements.txt
```

## 2. Run in VS Code

Install Python 3 and the Python extension in VS Code. Open this project folder, then open Terminal > New Terminal.

Windows:
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Streamlit will show a local address (usually http://localhost:8501). Open it in your browser and enter house details to get a prediction.

## 3. Algorithms and evaluation

The script trains three regression algorithms and evaluates them on a held-out 20% validation split. It reports MAE, RMSE, and R-squared. The model with the lowest RMSE is saved as `house_price_model.joblib`.

## 4. Publish a submission link

1. Create a GitHub repository and upload the project files. Do not upload your `.venv` folder.
2. Add `data/train.csv` only if your instructor permits redistribution of the dataset; otherwise keep the data local and explain in this README how to download it from Kaggle.
3. Deploy the GitHub repository on Streamlit Community Cloud: https://share.streamlit.io/
4. Set the app's main file path to `app.py`. Make sure `train_model.py` has been run and the generated `house_price_model.joblib` and `metrics.json` are included in the deployment repository. Alternatively, add a build step that trains the model during deployment.
5. Submit both your GitHub repository URL (source code) and Streamlit app URL (working demo).

## Project note

The Kaggle competition dataset describes homes in Ames, Iowa, USA. The model is an educational demonstration and should not be treated as a real-world valuation tool.
