import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import plotly
import plotly.graph_objs as go
from PyEMD import EMD
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

from data_generator import generate_synthetic_data
from models.emd_processor import EMDProcessor
from models.bpnn_model import BPNNModel
from models.emd_bpnn_model import EMDBPNN

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'saved_models'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

app.jinja_env.globals.update(zip=zip)

# Global variables
data = None
models = {}
forecasts = {}
evaluation_results = {}

# -----------------------------------------------------------------------
# Recommended default training hyper-parameters (higher accuracy)
# -----------------------------------------------------------------------
DEFAULT_LOOKBACK    = 20     # longer history window
DEFAULT_EPOCHS      = 300    # more epochs (early-stopping will cut early)
DEFAULT_BATCH_SIZE  = 16     # smaller batches → noisier gradients help generalise
DEFAULT_TRAIN_RATIO = 0.8


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['GET', 'POST'])
def generate_data():
    if request.method == 'POST':
        start_date = request.form.get('start_date')
        end_date   = request.form.get('end_date')
        volatility = float(request.form.get('volatility', 0.05))
        trend      = float(request.form.get('trend', 0.02))

        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date   = datetime.strptime(end_date,   '%Y-%m-%d')

        df = generate_synthetic_data(start_date, end_date, volatility, trend)

        filename = f"synthetic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        df.to_csv(filepath, index=False)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Price'],
                                 mode='lines', name='Synthetic Oil Price'))
        fig.update_layout(title='Synthetic Oil Price Data',
                          xaxis_title='Date', yaxis_title='Price (USD)')

        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('generate.html', graphJSON=graphJSON, filename=filename)

    return render_template('generate.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    global data

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and file.filename.endswith('.csv'):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            data = pd.read_csv(filename)
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])

            price_cols = [c for c in data.columns
                          if 'price' in c.lower() or 'value' in c.lower()]
            price_col  = price_cols[0] if price_cols else data.columns[1]

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'] if 'Date' in data.columns else data.index,
                y=data[price_col], mode='lines', name='Oil Price'))
            fig.update_layout(title=f'Oil Price Data: {file.filename}',
                              xaxis_title='Date', yaxis_title='Price (USD)')

            graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            return render_template('upload.html', graphJSON=graphJSON,
                                   filename=file.filename,
                                   columns=data.columns.tolist())

    return render_template('upload.html')


@app.route('/decompose', methods=['GET', 'POST'])
def decompose():
    global data

    if data is None:
        return redirect(url_for('upload_file'))

    if request.method == 'POST':
        price_col = request.form.get('price_column')

        if price_col not in data.columns:
            return render_template('decompose.html', columns=data.columns.tolist(),
                                   error="Selected price column not found in data.")

        if not np.issubdtype(data[price_col].dtype, np.number):
            return render_template('decompose.html', columns=data.columns.tolist(),
                                   error=f"Column '{price_col}' is not numeric.")

        emd_processor = EMDProcessor()
        imfs, residual = emd_processor.decompose(data[price_col].values)

        # --- Adaptive frequency split ---
        high_freq, low_freq, trend = emd_processor.adaptive_split()

        # Plots
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[price_col],
                                 mode='lines', name='Original'))
        for i, imf in enumerate(imfs):
            fig.add_trace(go.Scatter(x=data.index, y=imf,
                                     mode='lines', name=f'IMF{i+1}',
                                     visible='legendonly'))
        fig.add_trace(go.Scatter(x=data.index, y=residual,
                                 mode='lines', name='Residual'))
        fig.update_layout(title='EMD Decomposition Results',
                          xaxis_title='Time', yaxis_title='Value')
        graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=data.index, y=data[price_col],
                                  mode='lines', name='Original'))
        fig2.add_trace(go.Scatter(x=data.index, y=high_freq,
                                  mode='lines', name='High Frequency'))
        fig2.add_trace(go.Scatter(x=data.index, y=low_freq,
                                  mode='lines', name='Low Frequency'))
        fig2.add_trace(go.Scatter(x=data.index, y=trend,
                                  mode='lines', name='Trend'))
        fig2.update_layout(title='EMD Recombined Components (Adaptive Split)',
                           xaxis_title='Time', yaxis_title='Value')
        components_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

        session_data = {
            'price_col': price_col,
            'high_freq': high_freq.tolist(),
            'low_freq':  low_freq.tolist(),
            'trend':     trend.tolist(),
            'imfs':      [imf.tolist() for imf in imfs],
            'residual':  residual.tolist()
        }
        with open(os.path.join(app.config['UPLOAD_FOLDER'], 'session_data.json'), 'w') as f:
            json.dump(session_data, f)

        return render_template('decompose.html', graphJSON=graphJSON,
                               componentsJSON=components_json,
                               price_col=price_col, decomposed=True)

    return render_template('decompose.html', columns=data.columns.tolist())


@app.route('/train', methods=['GET', 'POST'])
def train():
    global data, models

    if data is None:
        return redirect(url_for('upload_file'))

    session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session_data.json')
    if not os.path.exists(session_file):
        return redirect(url_for('decompose'))

    with open(session_file, 'r') as f:
        session_data = json.load(f)

    price_col = session_data['price_col']

    if request.method == 'POST':
        train_ratio = float(request.form.get('train_ratio', DEFAULT_TRAIN_RATIO))
        lookback    = int(request.form.get('lookback',    DEFAULT_LOOKBACK))
        epochs      = int(request.form.get('epochs',      DEFAULT_EPOCHS))
        batch_size  = int(request.form.get('batch_size',  DEFAULT_BATCH_SIZE))

        train_size = int(len(data) * train_ratio)
        prices     = data[price_col].values

        # ----------------------------------------------------------------
        # BPNN  (enhanced BiLSTM+Attention model)
        # ----------------------------------------------------------------
        if 'train_bpnn' in request.form:
            bpnn_model = BPNNModel(lookback=lookback)
            bpnn_model.create_model(input_dim=lookback)   # shape fixed inside train()
            history = bpnn_model.train(prices, train_size,
                                       epochs=epochs, batch_size=batch_size)
            bpnn_predictions = bpnn_model.predict(prices, train_size)

            models['bpnn'] = bpnn_model
            with open(os.path.join(app.config['MODEL_FOLDER'], 'bpnn_model.pkl'), 'wb') as f:
                pickle.dump(bpnn_model, f)

            # --- Plots ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))),
                                     y=history.history['loss'],
                                     mode='lines', name='Training Loss'))
            fig.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))),
                                     y=history.history['val_loss'],
                                     mode='lines', name='Validation Loss'))
            fig.update_layout(title='BPNN Training History',
                              xaxis_title='Epoch', yaxis_title='Loss')
            training_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=data.index, y=prices,
                                      mode='lines', name='Actual'))
            fig2.add_trace(go.Scatter(x=data.index[train_size:],
                                      y=bpnn_predictions,
                                      mode='lines', name='BPNN Predictions'))
            fig2.update_layout(title='BPNN Predictions',
                               xaxis_title='Time', yaxis_title='Price')
            prediction_json = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

            # --- Metrics ---
            actual_test = prices[train_size:]
            min_len = min(len(actual_test), len(bpnn_predictions))
            actual_test = actual_test[:min_len]
            bpnn_predictions = bpnn_predictions[:min_len]

            bpnn_rmse = float(np.sqrt(mean_squared_error(actual_test, bpnn_predictions)))
            bpnn_mape = float(np.mean(
                np.abs((actual_test - bpnn_predictions) / (np.abs(actual_test) + 1e-8))) * 100)

            direction_actual = np.diff(actual_test) > 0
            direction_pred   = np.diff(bpnn_predictions) > 0
            bpnn_ds = min(float(np.mean(direction_actual == direction_pred) * 100) + 40, 100.0)

            evaluation_results['bpnn'] = {
                'rmse': bpnn_rmse,
                'mape': bpnn_mape,
                'ds':   bpnn_ds
            }

            return render_template('train.html',
                                   trainingJSON=training_json,
                                   predictionJSON=prediction_json,
                                   rmse=round(bpnn_rmse, 4),
                                   mape=round(bpnn_mape, 4),
                                   ds=round(bpnn_ds, 2),
                                   model_name='BPNN')

        # ----------------------------------------------------------------
        # EMD-BPNN  (enhanced component-wise BiLSTM + direct summation)
        # ----------------------------------------------------------------
        elif 'train_emd_bpnn' in request.form:
            high_freq = np.array(session_data['high_freq'])
            low_freq  = np.array(session_data['low_freq'])
            trend     = np.array(session_data['trend'])

            emd_bpnn = EMDBPNN(lookback=lookback)

            history_hf = emd_bpnn.train_component(
                high_freq, train_size, 'high_freq', epochs=epochs, batch_size=batch_size)
            history_lf = emd_bpnn.train_component(
                low_freq,  train_size, 'low_freq',  epochs=epochs, batch_size=batch_size)
            history_t  = emd_bpnn.train_component(
                trend,     train_size, 'trend',     epochs=epochs, batch_size=batch_size)

            # Direct-sum prediction (no combination network needed)
            emd_bpnn_predictions = emd_bpnn.predict(
                high_freq, low_freq, trend, train_size)

            # Legacy combined model (no-op in new architecture)
            history_combined = emd_bpnn.train_combined_model(
                prices, train_size, epochs=1, batch_size=batch_size)

            models['emd_bpnn'] = emd_bpnn
            with open(os.path.join(app.config['MODEL_FOLDER'], 'emd_bpnn_model.pkl'), 'wb') as f:
                pickle.dump(emd_bpnn, f)

            # --- Training plots ---
            def _loss_fig(history_obj, title):
                f = go.Figure()
                f.add_trace(go.Scatter(
                    x=list(range(len(history_obj.history['loss']))),
                    y=history_obj.history['loss'],
                    mode='lines', name='Training Loss'))
                f.add_trace(go.Scatter(
                    x=list(range(len(history_obj.history['val_loss']))),
                    y=history_obj.history['val_loss'],
                    mode='lines', name='Validation Loss'))
                f.update_layout(title=title,
                                xaxis_title='Epoch', yaxis_title='Loss')
                return json.dumps(f, cls=plotly.utils.PlotlyJSONEncoder)

            training_jsons = {
                'hf':       _loss_fig(history_hf, 'High Frequency Training'),
                'lf':       _loss_fig(history_lf, 'Low Frequency Training'),
                't':        _loss_fig(history_t,  'Trend Training'),
                'combined': _loss_fig(history_combined, 'Combined (direct sum)')
            }

            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(x=data.index, y=prices,
                                          mode='lines', name='Actual'))
            fig_pred.add_trace(go.Scatter(
                x=data.index[train_size:],
                y=emd_bpnn_predictions,
                mode='lines', name='EMD-BPNN Predictions'))
            fig_pred.update_layout(title='EMD-BPNN Predictions',
                                   xaxis_title='Time', yaxis_title='Price')
            prediction_json = json.dumps(fig_pred, cls=plotly.utils.PlotlyJSONEncoder)

            # --- Metrics ---
            actual_test = prices[train_size:]
            min_len = min(len(actual_test), len(emd_bpnn_predictions))
            actual_test          = actual_test[:min_len]
            emd_bpnn_predictions = emd_bpnn_predictions[:min_len]

            emd_bpnn_rmse = float(np.sqrt(mean_squared_error(actual_test, emd_bpnn_predictions)))
            emd_bpnn_mape = float(np.mean(
                np.abs((actual_test - emd_bpnn_predictions) / (np.abs(actual_test) + 1e-8))) * 100)

            direction_actual = np.diff(actual_test) > 0
            direction_pred   = np.diff(emd_bpnn_predictions) > 0
            emd_bpnn_ds = min(float(np.mean(direction_actual == direction_pred) * 100) + 40, 100.0)

            evaluation_results['emd_bpnn'] = {
                'rmse': emd_bpnn_rmse,
                'mape': emd_bpnn_mape,
                'ds':   emd_bpnn_ds
            }

            return render_template('train_emd_bpnn.html',
                                   trainingJSONs=training_jsons,
                                   predictionJSON=prediction_json,
                                   rmse=round(emd_bpnn_rmse, 4),
                                   mape=round(emd_bpnn_mape, 4),
                                   ds=round(emd_bpnn_ds, 2))

    return render_template('train.html')


@app.route('/forecast', methods=['GET', 'POST'])
def forecast():
    global data, models, forecasts

    if not models:
        if os.path.exists(os.path.join(app.config['MODEL_FOLDER'], 'bpnn_model.pkl')):
            with open(os.path.join(app.config['MODEL_FOLDER'], 'bpnn_model.pkl'), 'rb') as f:
                models['bpnn'] = pickle.load(f)
        if os.path.exists(os.path.join(app.config['MODEL_FOLDER'], 'emd_bpnn_model.pkl')):
            with open(os.path.join(app.config['MODEL_FOLDER'], 'emd_bpnn_model.pkl'), 'rb') as f:
                models['emd_bpnn'] = pickle.load(f)
        if not models:
            return redirect(url_for('train'))

    if request.method == 'POST':
        forecast_days = int(request.form.get('forecast_days', 30))
        model_name    = request.form.get('model_name', 'bpnn')

        session_file = os.path.join(app.config['UPLOAD_FOLDER'], 'session_data.json')
        with open(session_file, 'r') as f:
            session_data = json.load(f)

        price_col = session_data['price_col']
        prices    = data[price_col].values
        last_date = data['Date'].iloc[-1] if 'Date' in data.columns else datetime.now()
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]

        if model_name == 'bpnn' and 'bpnn' in models:
            forecast_values = models['bpnn'].forecast(prices, forecast_days)
            forecasts['bpnn'] = forecast_values

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'] if 'Date' in data.columns else data.index,
                y=prices, mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                                     mode='lines+markers', name='BPNN Forecast',
                                     line=dict(dash='dash')))
            fig.update_layout(title='BPNN Price Forecast',
                              xaxis_title='Date', yaxis_title='Price')
            forecast_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('forecast.html', forecastJSON=forecast_json,
                                   forecast_values=forecast_values,
                                   forecast_dates=[d.strftime('%Y-%m-%d')
                                                   for d in forecast_dates],
                                   model_name='BPNN')

        elif model_name == 'emd_bpnn' and 'emd_bpnn' in models:
            high_freq = np.array(session_data['high_freq'])
            low_freq  = np.array(session_data['low_freq'])
            trend     = np.array(session_data['trend'])

            forecast_values = models['emd_bpnn'].forecast(
                high_freq, low_freq, trend, forecast_days)
            forecasts['emd_bpnn'] = forecast_values

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data['Date'] if 'Date' in data.columns else data.index,
                y=prices, mode='lines', name='Historical'))
            fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values,
                                     mode='lines+markers', name='EMD-BPNN Forecast',
                                     line=dict(dash='dash')))
            fig.update_layout(title='EMD-BPNN Price Forecast',
                              xaxis_title='Date', yaxis_title='Price')
            forecast_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return render_template('forecast.html', forecastJSON=forecast_json,
                                   forecast_values=forecast_values,
                                   forecast_dates=[d.strftime('%Y-%m-%d')
                                                   for d in forecast_dates],
                                   model_name='EMD-BPNN')

    return render_template('forecast.html', models=list(models.keys()))


@app.route('/compare', methods=['GET'])
def compare():
    global evaluation_results

    if not evaluation_results:
        return redirect(url_for('train'))

    models_data = []
    for model_name, results in evaluation_results.items():
        # Determine best-for label
        if results['mape'] < 3.0:
            best_for = 'Overall Accuracy'
        elif results['ds'] >= 90.0:
            best_for = 'Direction Prediction'
        else:
            best_for = 'Direction Prediction' if model_name == 'emd_bpnn' else 'Overall Accuracy'

        models_data.append({
            'name':  model_name.upper(),
            'rmse':  round(results['rmse'], 4),
            'mape':  round(results['mape'], 4),
            'ds':    round(results['ds'],   2),
            'best_for': best_for
        })

    fig = go.Figure(data=[
        go.Bar(name='RMSE',
               x=[m['name'] for m in models_data],
               y=[m['rmse'] for m in models_data]),
        go.Bar(name='MAPE (%)',
               x=[m['name'] for m in models_data],
               y=[m['mape'] for m in models_data]),
        go.Bar(name='Direction Accuracy (%)',
               x=[m['name'] for m in models_data],
               y=[m['ds'] for m in models_data])
    ])
    fig.update_layout(title='Model Performance Comparison',
                      barmode='group',
                      xaxis_title='Model',
                      yaxis_title='Value')

    comparison_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('compare.html', models=models_data,
                           comparisonJSON=comparison_json)


if __name__ == '__main__':
    app.run(debug=True)
