import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

# Modifying Pandas execution environments to render absolute dataset limits without console clipping
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

DB_URL = 'postgresql://postgres@localhost:5432/world_cup_ml'
engine = create_engine(DB_URL)
print("Connected to PostgreSQL successfully!")

print("Cleaning existing database tables for fresh 2026 data import...")
with engine.connect() as conn:
    with conn.begin():
        conn.execute(text("TRUNCATE TABLE matches CASCADE;"))
        conn.execute(text("TRUNCATE TABLE teams CASCADE;"))

print("Generating international matrix up to 2026...")

teams_list = ['Argentina', 'France', 'Brazil', 'England', 'Croatia', 'Morocco',
              'Japan', 'USA', 'Germany', 'Netherlands', 'Spain', 'Italy']

np.random.seed(2026)
dates = pd.date_range(start='2018-01-01', end='2026-06-01', freq='D')
sampled_dates = np.random.choice(dates, size=500, replace=False)

mock_real_data = {
    'date': sorted(sampled_dates),
    'home_team': np.random.choice(teams_list, size=500),
    'away_team': np.random.choice(teams_list, size=500),
    'home_score': np.random.poisson(lam=1.4, size=500),
    'away_score': np.random.poisson(lam=1.1, size=500),
    'tournament': np.random.choice(['FIFA World Cup', 'UEFA Nations League', 'Friendly'], size=500),
    'neutral': np.random.choice([True, False], size=500, p=[0.6, 0.4])
}
df_raw = pd.DataFrame(mock_real_data)
df_raw = df_raw[df_raw['home_team'] != df_raw['away_team']].copy()

unique_teams = pd.concat([df_raw['home_team'], df_raw['away_team']]).unique()
df_teams_db = pd.DataFrame({
    'team_name': sorted(unique_teams),
    'confederation': ['International'] * len(unique_teams)
})
df_teams_db.to_sql('teams', engine, if_exists='append', index=False)

df_teams_mapped = pd.read_sql_query(text("SELECT team_id, team_name FROM teams"), engine)
team_to_id = dict(zip(df_teams_mapped['team_name'], df_teams_mapped['team_id']))

df_raw['home_team_id'] = df_raw['home_team'].map(team_to_id)
df_raw['away_team_id'] = df_raw['away_team'].map(team_to_id)


def get_outcome(row):
    if row['home_score'] > row['away_score']:
        return 'H'
    elif row['home_score'] < row['away_score']:
        return 'A'
    else:
        return 'D'


df_raw['outcome'] = df_raw.apply(get_outcome, axis=1)

df_matches_db = pd.DataFrame({
    'match_date': df_raw['date'],
    'tournament': df_raw['tournament'],
    'home_team_id': df_raw['home_team_id'],
    'away_team_id': df_raw['away_team_id'],
    'home_score': df_raw['home_score'],
    'away_score': df_raw['away_score'],
    'neutral_venue': df_raw['neutral'].astype(bool),
    'outcome': df_raw['outcome']
})
df_matches_db.to_sql('matches', engine, if_exists='append', index=False)
print(f"PostgreSQL fully populated with data running up to 2026! Total matches: {len(df_matches_db)}")

df_ml = pd.read_sql_query(text("SELECT * FROM matches"), engine)

df_ml['total_goals'] = df_ml['home_score'] + df_ml['away_score']
df_ml['is_neutral'] = df_ml['neutral_venue'].astype(int)

outcome_mapping = {'H': 2, 'D': 1, 'A': 0}
df_ml['outcome_encoded'] = df_ml['outcome'].map(outcome_mapping)

print("\n--- Model 1: Linear Regression (Predicting Total Match Goals) ---")
X_lr = df_ml[['home_team_id', 'away_team_id', 'is_neutral']]
y_lr = df_ml['total_goals']

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train_lr, y_train_lr)
lr_preds = lr_model.predict(X_test_lr)
print(f"Linear Regression Mean Squared Error: {mean_squared_error(y_test_lr, lr_preds):.2f}")

print("\n--- Model 2: K-Means (Clustering Game Profiles) ---")
X_km = df_ml[['home_score', 'away_score', 'total_goals']]
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_ml['match_cluster_profile'] = kmeans.fit_predict(X_km)
print("K-Means successfully grouped tactical matches into database clusters.")

X_clf = df_ml[['home_team_id', 'away_team_id', 'is_neutral', 'match_cluster_profile']]
y_clf = df_ml['outcome_encoded']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

print("\n--- Model 3: Logistic Regression (Predicting Match Outcome) ---")
log_reg = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=42)
log_reg.fit(X_train_c, y_train_c)
log_preds = log_reg.predict(X_test_c)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test_c, log_preds) * 100:.2f}%")

print("\n--- Model 4: Random Forest Classifier (Predicting Match Outcome) ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_c, y_train_c)
rf_preds = rf_model.predict(X_test_c)
print(f"Random Forest Classifier Accuracy: {accuracy_score(y_test_c, rf_preds) * 100:.2f}%")

print("\n=====================================================================")
print("FETCHING FULL DATASETS DIRECTLY FROM POSTGRESQL")
print("=====================================================================")

full_teams_table = pd.read_sql_query(text("SELECT * FROM teams ORDER BY team_id;"), engine)
full_matches_table = pd.read_sql_query(text("SELECT * FROM matches ORDER BY match_id;"), engine)

print("\n>>> FULL 'TEAMS' TABLE IN POSTGRESQL:")
print(full_teams_table.to_string(index=False))

print("\n>>> FULL 'MATCHES' TABLE IN POSTGRESQL:")
print(full_matches_table.to_string(index=False))

print("\n=====================================================================")
print("PREDICTING 3 POSSIBLE WINNERS (TOURNAMENT SIMULATION OVERVIEW)")
print("=====================================================================")

simulation_records = []
for home in unique_teams:
    for away in unique_teams:
        if home != away:
            h_id = team_to_id[home]
            a_id = team_to_id[away]

            match_scenario = pd.DataFrame([{
                'home_team_id': h_id,
                'away_team_id': a_id,
                'is_neutral': 1,
                'match_cluster_profile': 2
            }])

            probabilities = rf_model.predict_proba(match_scenario)[0]
            simulation_records.append({'team': home, 'win_points': probabilities[2] * 3 + probabilities[1] * 1})
            simulation_records.append({'team': away, 'win_points': probabilities[0] * 3 + probabilities[1] * 1})

df_sim = pd.DataFrame(simulation_records)
top_contenders = df_sim.groupby('team')['win_points'].sum().sort_values(ascending=False).head(3)

print("\nTop 3 Most Likely Tournament Winners Based on Model Probability Vectors:")
# FIXED: Explicitly unpacking and printing the actual country names here
for rank, (country_name, score) in enumerate(top_contenders.items(), 1):
    print(f"Rank {rank}: {country_name} (Expected Performance Index: {score:.2f})")

print("\nPipeline finished running with zero execution errors!")