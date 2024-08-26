import streamlit as st
import ccxt
import pandas as pd
import numpy as np
import ta
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Function to get available exchanges that don't require API credentials
def get_available_exchanges():
    available_exchanges = []
    exchanges_to_test = [
        'kraken',     # Kraken allows public access to market data
        'digifinex',  # Digifinex allows public access to market data
        # Add more exchanges here as needed
    ]

    for exchange_name in exchanges_to_test:
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class()
            # Load markets
            exchange.load_markets()
            available_exchanges.append(exchange_name)
        except Exception as e:
            print(f"Error initializing {exchange_name}: {e}")
    
    return available_exchanges

# Function to fetch symbols for a given exchange
def fetch_symbols(exchange_name):
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        exchange.load_markets()
        return list(exchange.symbols)
    except Exception as e:
        print(f"Error fetching symbols for {exchange_name}: {e}")
        return []

# Function to fetch historical data
def fetch_data(exchange_name, symbol, timeframe):
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
        df = exchange.fetch_ohlcv(symbol, timeframe, exchange.parse8601('2018-01-01T00:00:00Z'))
        df = pd.DataFrame(df, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

# Streamlit app layout
st.title('Cryptocurrency Trading Signal Generator')

# Fetch available exchanges
exchange_list = get_available_exchanges()
if not exchange_list:
    st.write("No exchanges with available symbols found.")
else:
    # Exchange selection
    selected_exchange = st.selectbox('Select Exchange:', exchange_list)
    exchange = getattr(ccxt, selected_exchange)()

    # Fetch symbols for the selected exchange
    symbols = fetch_symbols(selected_exchange)
    if symbols:
        symbol = st.selectbox('Select Coin Pair:', symbols)
    else:
        st.write("No symbols available. Please select a different exchange.")
        st.stop()

    # Time frame selection
    time_frame = st.selectbox('Select Time Frame:', ['1d', '4h', '1h', '15m', '5m'])

    # Fetch historical data
    df = fetch_data(selected_exchange, symbol, time_frame)
    if df.empty:
        st.write("No data available for the selected coin pair and time frame.")
        st.stop()

    # Add indicators based on user input
    def add_indicators(df):
        if st.checkbox('SMA (5)', value=True):
            df['sma5'] = ta.trend.sma_indicator(df['close'], window=5)
        if st.checkbox('SMA (20)', value=True):
            df['sma20'] = ta.trend.sma_indicator(df['close'], window=20)
        if st.checkbox('RSI', value=True):
            df['rsi'] = ta.momentum.rsi(df['close'])
        if st.checkbox('CCI', value=True):
            df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
        if st.checkbox('ADX', value=True):
            df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
        if st.checkbox('MACD', value=True):
            df['macd'] = ta.trend.macd(df['close'])
            df['macd_signal'] = ta.trend.macd_signal(df['close'])
            df['macd_diff'] = ta.trend.macd_diff(df['close'])
        if st.checkbox('Bollinger Bands', value=True):
            df['bb_high'], df['bb_mid'], df['bb_low'] = (
                ta.volatility.bollinger_hband(df['close']),
                ta.volatility.bollinger_mavg(df['close']),
                ta.volatility.bollinger_lband(df['close'])
            )
        if st.checkbox('VWAP', value=True):
            df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        if st.checkbox('ROC', value=True):
            df['roc'] = ta.momentum.roc(df['close'], window=12)
        return df

    df = add_indicators(df)

    # Prepare features and target
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]

    # Fill missing values and remove any remaining NaNs
    df[features] = df[features].ffill().bfill()
    df = df.dropna()

    X = df[features]
    y = df['target']

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize models
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)

    # Train and evaluate models
    models = [rf_model, xgb_model]
    model_names = ['Random Forest', 'XGBoost']

    model_performances = {}

    for model, name in zip(models, model_names):
        scores = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}
        all_y_test = []
        all_y_pred = []

        for train_index, test_index in tscv.split(X_resampled):
            X_train, X_test = X_resampled[train_index], X_resampled[test_index]
            y_train, y_test = y_resampled[train_index], y_resampled[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            scores['accuracy'].append(accuracy_score(y_test, y_pred))
            scores['precision'].append(precision_score(y_test, y_pred))
            scores['recall'].append(recall_score(y_test, y_pred))
            scores['f1'].append(f1_score(y_test, y_pred))

            all_y_test.extend(y_test)
            all_y_pred.extend(y_pred)

        model_performances[name] = {
            'scores': scores,
            'confusion_matrix': confusion_matrix(all_y_test, all_y_pred)
        }

    # Create performance table
    performance_data = []
    for name, performance in model_performances.items():
        model_data = {'Model': name}
        for metric, values in performance['scores'].items():
            model_data[metric.capitalize()] = f"{np.mean(values):.4f} (±{np.std(values):.4f})"
        performance_data.append(model_data)

    performance_df = pd.DataFrame(performance_data)

    # Display model performance table
    st.write("Model Performance:")
    st.table(performance_df.set_index('Model'))

    # Display confusion matrices
    st.write("Confusion Matrices:")
    for name, performance in model_performances.items():
        st.write(f"\n{name} Confusion Matrix:")
        cm_df = pd.DataFrame(performance['confusion_matrix'], 
                             columns=['Predicted Negative', 'Predicted Positive'],
                             index=['Actual Negative', 'Actual Positive'])
        st.table(cm_df)

    # Model selection
    selected_model = st.selectbox('Select Model for Signal Generation:', model_names)

    if selected_model == 'Random Forest':
        best_model = rf_model
    else:
        best_model = xgb_model

    best_model.fit(X_resampled, y_resampled)

    # Generate signals
    df['probability'] = best_model.predict_proba(X_scaled)[:, 1]

    # Implement lookback period, weighted signal, and confirmation signal
    def generate_improved_signal(df, lookback_period=5, confirmation_threshold=3):
        df['raw_signal'] = 0
        df.loc[df['probability'] > 0.6, 'raw_signal'] = 1  # Long
        df.loc[df['probability'] < 0.4, 'raw_signal'] = -1  # Short

        # Calculate weighted signal
        weights = np.linspace(1, 2, lookback_period)
        weights = weights / weights.sum()
        df['weighted_signal'] = df['raw_signal'].rolling(window=lookback_period).apply(lambda x: np.dot(x, weights[::-1]), raw=True)

        # Apply confirmation signal
        df['confirmed_signal'] = 0
        long_confirmation = (df['weighted_signal'] > 0).rolling(window=confirmation_threshold).sum() == confirmation_threshold
        short_confirmation = (df['weighted_signal'] < 0).rolling(window=confirmation_threshold).sum() == confirmation_threshold

        df.loc[long_confirmation, 'confirmed_signal'] = 1
        df.loc[short_confirmation, 'confirmed_signal'] = -1

        return df

    df = generate_improved_signal(df)

    # Convert numerical signals to labels
    def convert_signal_to_label(signal):
        if signal == 1:
            return 'Buy'
        elif signal == -1:
            return 'Sell'
        else:
            return 'Exit (No Position)'

    df['signal_label'] = df['confirmed_signal'].apply(convert_signal_to_label)

    # Implement strategy
    def implement_strategy(df):
        in_position = None
        entry_price = None

        for i in range(1, len(df)):
            current_signal = df['confirmed_signal'].iloc[i]
            previous_signal = df['confirmed_signal'].iloc[i-1]

            if in_position is None and current_signal in [1, -1]:
                entry_price = df['close'].iloc[i]
                df.loc[df.index[i], 'entry_price'] = entry_price
                in_position = current_signal
            elif in_position == 1:
                if current_signal == -1 or (current_signal == 0 and previous_signal == 1):
                    df.loc[df.index[i], 'exit_price'] = df['open'].iloc[i]
                    in_position = None
            elif in_position == -1:
                if current_signal == 1 or (current_signal == 0 and previous_signal == -1):
                    df.loc[df.index[i], 'exit_price'] = df['open'].iloc[i]
                    in_position = None

        return df

    df = implement_strategy(df)

    # Display the final DataFrame with signals and trade details, sorted from the most recent
    st.write("Generated Signals and Trade Details:")
    st.dataframe(df[['open','high','low','close', 'raw_signal', 'weighted_signal', 'confirmed_signal', 'signal_label', 'entry_price', 'exit_price']].sort_index(ascending=False).head(30), width=1500)

    # Calculate and display strategy performance
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['confirmed_signal'].shift(1) * df['returns']

    cumulative_returns = (1 + df['returns']).cumprod() - 1
    cumulative_strategy_returns = (1 + df['strategy_returns']).cumprod() - 1

    st.write("\nStrategy Performance:")
    st.write(f"Total Return: {cumulative_strategy_returns.iloc[-1]:.2%}")
    st.write(f"Sharpe Ratio: {np.sqrt(252) * df['strategy_returns'].mean() / df['strategy_returns'].std():.2f}")

    # Plot cumulative returns
    st.line_chart(pd.DataFrame({
        'Buy and Hold': cumulative_returns,
        'Strategy': cumulative_strategy_returns
    }))