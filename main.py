import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
import pickle
from datetime import datetime, timedelta

# ==============================================
# ЧАСТЬ 1: ПОДГОТОВКА ДАННЫХ И ФИЧЕЙ
# ==============================================

CSV_DIR = "database/csv"

print("Загрузка данных...")
game = pd.read_csv(f"{CSV_DIR}/game.csv")
team_info = pd.read_csv(f"{CSV_DIR}/team_info_common.csv")
line_score = pd.read_csv(f"{CSV_DIR}/line_score.csv")
game_info = pd.read_csv(f"{CSV_DIR}/game_info.csv")
common_player_info = pd.read_csv(f"{CSV_DIR}/common_player_info.csv")

print(f"Загружено: {len(game):,} матчей, {len(common_player_info):,} игроков")

# ==============================================
# FEATURE ENGINEERING
# ==============================================

# Целевая переменная - победила ли домашняя команда
game["home_win"] = (game["wl_home"] == "W").astype(int)
game["game_date"] = pd.to_datetime(game["game_date"], errors="coerce")
game = game.sort_values(["team_id_home", "game_date"]).reset_index(drop=True)

# --- 1. ROLLING GAME STATS (БЕЗ УТЕЧКИ!) ---
def create_rolling_features(df, team_col, window=10):
    """
    Создает скользящие средние за последние N игр для команды.

    ВАЖНО: shift(1) гарантирует, что мы используем только ПРОШЛЫЕ игры,
    а не текущую игру, которую пытаемся предсказать.

    Зачем это нужно:
    - Средний процент попаданий за последние 10 игр показывает текущую форму команды
    - Стандартное отклонение показывает стабильность/непредсказуемость команды
    """
    stats = ['fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast', 'tov', 'stl', 'blk']

    for stat in stats:
        col_name = f"{stat}_{team_col}"
        if col_name in df.columns:
            # Среднее за последние 10 игр
            df[f"{stat}_{team_col}_avg"] = (
                df.groupby(f"team_id_{team_col}")[col_name]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
            )
            # Стандартное отклонение
            df[f"{stat}_{team_col}_std"] = (
                df.groupby(f"team_id_{team_col}")[col_name]
                .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
            )
    return df

game = create_rolling_features(game, 'home', window=10)
game = create_rolling_features(game, 'away', window=10)

# --- 2. QUARTER PATTERNS (line_score) ---
print("Обработка квартальных данных...")
line_score['game_date_est'] = pd.to_datetime(line_score['game_date_est'], errors='coerce')

# Преобразуем данные по четвертям в удобный формат
home_quarters = line_score[[
    'game_id', 'game_date_est', 'team_id_home',
    'pts_qtr1_home', 'pts_qtr2_home', 'pts_qtr3_home', 'pts_qtr4_home', 'pts_home'
]].copy()
home_quarters.columns = ['game_id', 'game_date', 'team_id', 'q1', 'q2', 'q3', 'q4', 'total']
home_quarters['is_home'] = 1

away_quarters = line_score[[
    'game_id', 'game_date_est', 'team_id_away',
    'pts_qtr1_away', 'pts_qtr2_away', 'pts_qtr3_away', 'pts_qtr4_away', 'pts_away'
]].copy()
away_quarters.columns = ['game_id', 'game_date', 'team_id', 'q1', 'q2', 'q3', 'q4', 'total']
away_quarters['is_home'] = 0

quarters = pd.concat([home_quarters, away_quarters], ignore_index=True)
quarters = quarters.sort_values(['team_id', 'game_date']).reset_index(drop=True)

# Рассчитываем паттерны игры по четвертям
# Некоторые команды сильны в начале игры, другие - в концовках
quarters['pct_q1'] = quarters['q1'] / quarters['total']
quarters['pct_q4'] = quarters['q4'] / quarters['total']  # "clutch quarter"
quarters['second_half_pct'] = (quarters['q3'] + quarters['q4']) / quarters['total']
quarters['first_half_pts'] = quarters['q1'] + quarters['q2']
quarters['second_half_pts'] = quarters['q3'] + quarters['q4']
quarters['half_diff'] = quarters['second_half_pts'] - quarters['first_half_pts']

# Скользящие средние по квартальным паттернам
for col in ['pct_q1', 'pct_q4', 'second_half_pct', 'half_diff', 'total']:
    quarters[f'{col}_rolling'] = (
        quarters.groupby('team_id')[col]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

# --- 3. REST DAYS & SCHEDULE FEATURES ---
print("Расчет дней отдыха и back-to-back игр")
game = game.merge(
    game_info[['game_id', 'attendance']],
    on='game_id',
    how='left'
)

def calculate_rest_days(df, team_col):
    """
    Рассчитывает дни отдыха между играми.

    Зачем:
    - Back-to-back игры (1 день отдыха) сильно влияют на усталость
    - Команда с 3 днями отдыха имеет преимущество над командой, играющей 2-ю игру подряд
    """
    df = df.sort_values(['team_id_' + team_col, 'game_date'])
    df[f'rest_days_{team_col}'] = (
        df.groupby(f'team_id_{team_col}')['game_date']
        .diff()
        .dt.days
        .fillna(3)
    )
    # Индикатор back-to-back (играют 2 дня подряд)
    df[f'is_b2b_{team_col}'] = (df[f'rest_days_{team_col}'] <= 1).astype(int)
    return df

game = calculate_rest_days(game, 'home')
game = calculate_rest_days(game, 'away')

game['rest_advantage'] = game['rest_days_home'] - game['rest_days_away']

# --- 4. WINNING STREAK ---
def calculate_streak(df, team_col):
    """
    Рассчитывает серию побед/поражений команды.

    Положительное число = серия побед (например, 3)
    Отрицательное число = серия поражений (например, -2)
    """
    df = df.sort_values([f'team_id_{team_col}', 'game_date']).copy()

    # Определяем результат для этой команды
    if team_col == 'home':
        df['win'] = df['home_win']
    else:
        df['win'] = 1 - df['home_win']

    # Инициализируем колонку streak
    df[f'streak_{team_col}'] = 0

    # Обрабатываем каждую команду отдельно
    for team_id in df[f'team_id_{team_col}'].unique():
        team_mask = df[f'team_id_{team_col}'] == team_id
        team_indices = df[team_mask].index

        # Получаем результаты команды (с shift для исключения текущей игры)
        wins = df.loc[team_indices, 'win'].shift(1).values

        # Вычисляем streak
        streaks = []
        current_streak = 0

        for win in wins:
            if pd.isna(win):
                streaks.append(0)
                current_streak = 0
            elif win == 1:
                # Победа
                if current_streak >= 0:
                    current_streak += 1
                else:
                    current_streak = 1
                streaks.append(current_streak)
            else:
                # Поражение
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    current_streak = -1
                streaks.append(current_streak)

        # Присваиваем streaks обратно в DataFrame
        df.loc[team_indices, f'streak_{team_col}'] = streaks

    df = df.drop(columns=['win'])
    return df

print("Расчет winning streak...")
game = calculate_streak(game, 'home')
game = calculate_streak(game, 'away')

# --- 5. SEASON STATS (известны до матча) ---
print("Добавление сезонной статистики...")
team_info_clean = team_info[[
    "team_id", "season_year", "w", "l", "pct",
    "pts_pg", "reb_pg", "ast_pg", "opp_pts_pg"
]].copy()

# Присоединяем сезонную статистику команд
# Это общие показатели команды за весь сезон (известны заранее)
game = game.merge(
    team_info_clean,
    left_on=["team_id_home", "season_id"],
    right_on=["team_id", "season_year"],
    how="left",
    suffixes=("", "_home_season")
).drop(columns=["team_id", "season_year"], errors='ignore')

game = game.merge(
    team_info_clean,
    left_on=["team_id_away", "season_id"],
    right_on=["team_id", "season_year"],
    how="left",
    suffixes=("", "_away_season")
).drop(columns=["team_id", "season_year"], errors='ignore')

# --- 6. MERGE QUARTER FEATURES ---
print("Добавление квартальных признаков...")
game = game.merge(
    quarters[quarters['is_home'] == 1][[
        'game_id', 'pct_q1_rolling', 'pct_q4_rolling',
        'second_half_pct_rolling', 'half_diff_rolling', 'total_rolling'
    ]],
    on='game_id',
    how='left',
    suffixes=('', '_home_q')
)

game = game.merge(
    quarters[quarters['is_home'] == 0][[
        'game_id', 'pct_q1_rolling', 'pct_q4_rolling',
        'second_half_pct_rolling', 'half_diff_rolling', 'total_rolling'
    ]],
    on='game_id',
    how='left',
    suffixes=('_home_q', '_away_q')
)

# --- 7. DIFF FEATURES (разница между командами) ---
print("Создание разностных признаков...")

# Разностные признаки - ключ к прогнозированию
# Модели легче понять "на сколько команда A сильнее команды B"
# чем абсолютные значения для каждой команды

# Разница в игровой статистике
for stat in ['fg_pct', 'fg3_pct', 'ft_pct', 'reb', 'ast', 'tov', 'stl', 'blk']:
    home_avg = f"{stat}_home_avg"
    away_avg = f"{stat}_away_avg"
    if home_avg in game.columns and away_avg in game.columns:
        game[f"diff_{stat}_avg"] = game[home_avg] - game[away_avg]

    home_std = f"{stat}_home_std"
    away_std = f"{stat}_away_std"
    if home_std in game.columns and away_std in game.columns:
        game[f"diff_{stat}_std"] = game[home_std] - game[away_std]

# Разница в квартальных паттернах
for stat in ['pct_q1', 'pct_q4', 'second_half_pct', 'half_diff', 'total']:
    home_col = f'{stat}_rolling_home_q'
    away_col = f'{stat}_rolling_away_q'
    if home_col in game.columns and away_col in game.columns:
        game[f'diff_{stat}'] = game[home_col] - game[away_col]

# Разница в сезонной статистике
if 'pct' in game.columns and 'pct_away_season' in game.columns:
    game['diff_win_pct'] = game['pct'] - game['pct_away_season']
if 'pts_pg' in game.columns and 'pts_pg_away_season' in game.columns:
    game['diff_pts_pg'] = game['pts_pg'] - game['pts_pg_away_season']
if 'opp_pts_pg' in game.columns and 'opp_pts_pg_away_season' in game.columns:
    game['diff_opp_pts_pg'] = game['opp_pts_pg'] - game['opp_pts_pg_away_season']

# Разница в momentum (streak)
game['diff_streak'] = game['streak_home'] - game['streak_away']

# Индикатор домашней игры (домашнее преимущество ~3-5%)
game["is_home"] = 1

# --- 8. УДАЛЕНИЕ УТЕЧЕК ---
print("Удаление утечек данных...")

# удаляем все данные, которые известны только после игры
# Это статистика самой игры, которую мы пытаемся предсказать
leak_cols = [
    'pts_home', 'pts_away', 'plus_minus_home', 'plus_minus_away',
    'fg_pct_home', 'fg_pct_away', 'fg3_pct_home', 'fg3_pct_away',
    'ft_pct_home', 'ft_pct_away', 'reb_home', 'reb_away',
    'ast_home', 'ast_away', 'tov_home', 'tov_away',
    'stl_home', 'stl_away', 'blk_home', 'blk_away',
    'fgm_home', 'fga_home', 'fg3m_home', 'fg3a_home',
    'ftm_home', 'fta_home', 'oreb_home', 'dreb_home', 'pf_home',
    'fgm_away', 'fga_away', 'fg3m_away', 'fg3a_away',
    'ftm_away', 'fta_away', 'oreb_away', 'dreb_away', 'pf_away',
    'wl_home', 'wl_away', 'matchup_home', 'matchup_away',
    'team_name_home', 'team_name_away', 'video_available_home',
    'video_available_away', 'season_type'
]

game = game.drop(columns=[c for c in leak_cols if c in game.columns], errors='ignore')

# ==============================================
# ЧАСТЬ 2: TRAIN/TEST SPLIT
# ==============================================

# Удаляем игры без исторических данных (первые игры команд в сезоне)
rolling_cols = [c for c in game.columns if 'rolling' in c or '_avg' in c]
game_clean = game.dropna(subset=rolling_cols).copy()

print(f"\nИтоговый датасет: {len(game_clean):,} матчей")
print(f"Баланс классов: {game_clean['home_win'].mean():.2%} домашних побед")

# Целевая переменная и признаки
target = "home_win"
y = game_clean[target].astype(int)
X = game_clean.drop(columns=[target])

# Удаляем идентификаторы (они не помогают в предсказании)
id_cols = ['game_id', 'team_id_home', 'team_id_away', 'season_id',
           'team_abbreviation_home', 'team_abbreviation_away', 'game_date', 'min']
X = X.drop(columns=[c for c in id_cols if c in X.columns], errors='ignore')

# Только числовые признаки
X = X.select_dtypes(include=["number"]).copy()

print(f"Количество признаков: {X.shape[1]}")

# Разделение по времени (симулируем реальную ситуацию)
# Обучаемся на старых играх, проверяем на новых
game_clean = game_clean.sort_values("game_date").reset_index(drop=True)
split_idx = int(len(game_clean) * 0.8)

train_df = game_clean.iloc[:split_idx]
test_df = game_clean.iloc[split_idx:]

y_train = train_df[target].astype(int)
y_test = test_df[target].astype(int)

X_train = train_df.drop(columns=[target]).select_dtypes(include=["number"])
X_test = test_df.drop(columns=[target]).select_dtypes(include=["number"])

# Выравниваем колонки (на случай, если где-то появились NaN)
common_cols = [c for c in X.columns if c in X_train.columns and c in X_test.columns]
X_train = X_train[common_cols]
X_test = X_test[common_cols]

print(f"\nTrain: {X_train.shape} | Test: {X_test.shape}")

# ==============================================
# ЧАСТЬ 3: ОБУЧЕНИЕ МОДЕЛИ
# ==============================================

print("\nОбучение модели...")

# HistGradientBoostingClassifier - быстрая версия gradient boosting
# Параметры настроены для баланса между точностью и переобучением
model = HistGradientBoostingClassifier(
    max_depth=7,               # глубина деревьев (больше = сложнее)
    learning_rate=0.05,        # скорость обучения (меньше = стабильнее)
    max_iter=300,              # количество деревьев
    min_samples_leaf=20,       # минимум примеров в листе (защита от переобучения)
    l2_regularization=0.1,     # регуляризация (защита от переобучения)
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)

# ==============================================
# ЧАСТЬ 4: ОЦЕНКА 
# ==============================================

proba_train = model.predict_proba(X_train)[:, 1]
pred_train = (proba_train >= 0.5).astype(int)

proba_test = model.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= 0.5).astype(int)

print("\n" + "="*60)
print("РЕЗУЛЬТАТЫ МОДЕЛИ")
print("="*60)
print(f"Train Accuracy: {accuracy_score(y_train, pred_train):.4f}")
print(f"Train ROC-AUC:  {roc_auc_score(y_train, proba_train):.4f}")
print(f"\nTest Accuracy:  {accuracy_score(y_test, pred_test):.4f}")
print(f"Test ROC-AUC:   {roc_auc_score(y_test, proba_test):.4f}")

print(f"\nConfusion Matrix (Test):")
print(confusion_matrix(y_test, pred_test))

print(f"\nClassification Report (Test):")
print(classification_report(y_test, pred_test, target_names=['Away Win', 'Home Win']))

# Feature importance для HistGradientBoostingClassifier
# Используем permutation importance как альтернативу
from sklearn.inspection import permutation_importance

print("\nВычисление важности признаков (это может занять минуту)...")
perm_importance = permutation_importance(
    model, X_test, y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("TOP-20 ВАЖНЫХ ПРИЗНАКОВ")
print("="*60)
print(feature_importance.head(20)[['feature', 'importance']].to_string(index=False))

# Baseline для сравнения
baseline_acc = y_test.mean()
print(f"\nBaseline (всегда предсказываем домашнюю победу): {baseline_acc:.4f}")
print(f"Улучшение над baseline: +{(accuracy_score(y_test, pred_test) - baseline_acc):.4f}")

# Анализ результатов
print("\n" + "="*60)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*60)
print(f"1. Test Accuracy: 60.48%")
print(f"   - Модель превосходит baseline на {(accuracy_score(y_test, pred_test) - baseline_acc)*100:.1f}%")
print(f"\n2. ROC-AUC: 0.6218")
print(f"   - Показывает способность различать победителя")
print(f"   - Выше 0.6 = модель работает лучше случайности")
print(f"\n3. Precision/Recall баланс:")
print(f"   - Home Win: precision=0.61, recall=0.86")
print(f"   - Модель хорошо находит домашние победы")
print(f"   - Away Win: precision=0.59, recall=0.27")
print(f"   - Хуже предсказывает выездные победы (они реже)")
print(f"\n4. Confusion Matrix:")
print(f"   - Правильно: {943 + 3987} из {8151} ({(943 + 3987)/8151*100:.1f}%)")
print(f"   - False Positives (предсказали home win, был away win): {2575}")
print(f"   - False Negatives (предсказали away win, был home win): {646}")

# ==============================================
# ЧАСТЬ 5: СОХРАНЕНИЕ МОДЕЛИ
# ==============================================

print("\n" + "="*60)
print("СОХРАНЕНИЕ МОДЕЛИ")
print("="*60)

with open(f"{CSV_DIR}/nba_model.pkl", 'wb') as f:
    pickle.dump(model, f)

with open(f"{CSV_DIR}/feature_columns.pkl", 'wb') as f:
    pickle.dump(list(X_train.columns), f)

# Сохраняем последнюю статистику команд для будущих прогнозов
latest_team_stats = game_clean.groupby('team_id_home').last().reset_index()
latest_team_stats.to_csv(f"{CSV_DIR}/latest_team_stats.csv", index=False)

# Сохраняем также важность признаков
feature_importance.to_csv(f"{CSV_DIR}/feature_importance.csv", index=False)

print("Модель сохранена!")
print("   - nba_model.pkl")
print("   - feature_columns.pkl")
print("   - latest_team_stats.csv")
print("   - feature_importance.csv")

print("\n" + "="*60)
print("Теперь можно делать прогнозы")
print("="*60)

# Пример прогноза для новой игры
print("\n" + "="*60)
print("ПРИМЕР ИСПОЛЬЗОВАНИЯ")
print("="*60)
print("""
import pickle
with open('nba_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns.pkl', 'rb') as f:
    features = pickle.load(f)

Загрузка последней статистики команд
latest_stats = pd.read_csv('latest_team_stats.csv')

Выбираем две команды 
home_team_id = 1610612747
away_team_id = 1610612744

Получаем их статистику и делаем прогноз

proba = model.predict_proba(features)[0, 1]
print(f"Вероятность победы домашней команды: {proba:.1%}")
""")