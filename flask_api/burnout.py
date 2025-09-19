from __future__ import annotations

import pandas as pd
import xgboost as xgb
from textblob import TextBlob


class BurnoutPredictor:
    def __init__(self) -> None:
        self.model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
        self.user_history_df: pd.DataFrame | None = None

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').set_index('timestamp')
        df['sentiment'] = df['journal_entry'].apply(lambda text: TextBlob(text).sentiment.polarity)
        df['mood_7d_avg'] = df['mood'].rolling(window='7D').mean()
        df['energy_7d_avg'] = df['energy'].rolling(window='7D').mean()
        df['days_since_last_entry'] = df.index.to_series().diff().dt.days.fillna(0)
        return df.dropna()

    def train(self, user_history: list) -> None:
        df = pd.DataFrame(user_history)
        self.user_history_df = df
        features_df = self._create_features(df)
        if features_df.empty:
            return
        labels = (features_df['mood'] <= 2).astype(int)
        features = features_df[['mood_7d_avg', 'energy_7d_avg', 'days_since_last_entry']]
        self.model.fit(features, labels)

    def predict_risk_for_today(self, user_history_df: pd.DataFrame | None = None) -> tuple[float, str]:
        if user_history_df is None:
            user_history_df = self.user_history_df
        if user_history_df is None:
            return 0.0, "No user history available."
        features_df = self._create_features(user_history_df)
        if features_df.empty:
            return 0.0, "Not enough data."
        latest = features_df[['mood_7d_avg', 'energy_7d_avg', 'days_since_last_entry']].iloc[[-1]]
        prob = float(self.model.predict_proba(latest)[0][1])
        reason = (
            f"Based on 7-day avg mood ({latest['mood_7d_avg'].values[0]:.2f}) and "
            f"energy ({latest['energy_7d_avg'].values[0]:.2f})."
        )
        return prob, reason


