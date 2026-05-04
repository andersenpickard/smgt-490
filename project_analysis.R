library(dplyr)
library(tidyr)
library(lightgbm)

### READ DATA ### 

#entry_data <- read.csv("smgt_statcast.csv")
entry_data <- readRDS("/Users/andersenpickard/Downloads/statcast_data.rds")

data <- entry_data |>
  mutate(
    pfx_x = pfx_x * -12,
    pfx_z = pfx_z * 12,
    release_extension = release_extension * 12,
    release_pos_x = release_pos_x * 12,
    release_pos_y = release_pos_y * 12,
    release_pos_z = release_pos_z * 12,
    runs_scored = post_bat_score - bat_score
  )

#drop position players pitching (if they had 50+ pitches faced as batter)
batters_over_50 <- data |>
  filter(!is.na(batter)) |>
  count(batter, name = "pitches_faced") |>
  filter(pitches_faced > 50) |>
  pull(batter)
position_player_pitchers <- intersect(batters_over_50, unique(data$pitcher))
position_player_pitchers <- setdiff(position_player_pitchers, 660271)
#660271 is Shohei Ohtani
#drop rows where one of those IDs is pitching
data <- data |>
  filter(!(pitcher %in% position_player_pitchers))

### PITCH ARSENAL SUMMARY ### 

pitch_summary <- data |>
  group_by(pitcher, game_year, p_throws, pitch_name) |>
  summarise(
    n = n(),
    avg_pfx_x = mean(pfx_x, na.rm = TRUE),
    avg_pfx_z = mean(pfx_z, na.rm = TRUE),
    avg_release_spin = mean(release_spin_rate, na.rm = TRUE),
    avg_release_speed = mean(release_speed, na.rm = TRUE),
    avg_release_extension = mean(release_extension, na.rm = TRUE),
    avg_release_pos_x = mean(release_pos_x, na.rm = TRUE),
    avg_release_pos_y = mean(release_pos_y, na.rm = TRUE),
    avg_release_pos_z = mean(release_pos_z, na.rm = TRUE),
    avg_vx0 = mean(vx0, na.rm = TRUE),
    avg_vy0 = mean(vy0, na.rm = TRUE),
    avg_vz0 = mean(vz0, na.rm = TRUE),
    avg_ax = mean(ax, na.rm = TRUE),
    avg_ay = mean(ay, na.rm = TRUE),
    avg_az = mean(az, na.rm = TRUE),
    avg_spin_axis = mean(spin_axis, na.rm = TRUE),
    .groups = "drop_last"
  ) |>
  mutate(
    pct = n / sum(n),
  ) |>
  ungroup() |>
  filter(
    !(is.nan(avg_pfx_x) &
        is.nan(avg_pfx_z) &
        is.nan(avg_release_spin) &
        is.nan(avg_release_speed))
  )

### MODEL TRAINING ###

train <- data |>
  filter(game_year %in% 2021:2024)

features <- c(
  "balls","strikes","outs_when_up",
  "on_1b","on_2b","on_3b",
  "bat_score_diff",
  "stand","p_throws",
  "pitch_name",
  "release_speed","pfx_x","pfx_z",
  "release_spin_rate","spin_axis",
  "release_extension"
)

X <- train[,features]
y <- train$runs_scored

X$pitch_name <- as.factor(X$pitch_name)
X$stand <- as.factor(X$stand)
X$p_throws <- as.factor(X$p_throws)

dtrain <- lgb.Dataset(data.matrix(X), label = y)

params <- list(
  objective = "regression",
  metric = "l2",
  learning_rate = 0.05,
  num_leaves = 64
)

model <- lgb.train(params, dtrain, 400)

### TEST / IMPLEMENT MODEL SPLIT ### 

test <- data |>
  filter(game_year == 2025) |>
  mutate(row_id = row_number())

### GENERATE CANDIDATE COUNTERFACTUAL PITCHES ###

candidates <- test |>
  inner_join(
    pitch_summary,
    by = c("pitcher","game_year","p_throws"),
    relationship = "many-to-many"
  )

### SUB IN COUNTERFACTUAL PITCH CHARACTERISTICS / METRICS ###

candidates <- candidates |>
  mutate(
    actual_pitch = pitch_name.x,
    pitch_name = pitch_name.y,
    is_same_pitch = pitch_name.x == pitch_name.y,
    release_speed     = if_else(is_same_pitch, release_speed,     avg_release_speed),
    pfx_x             = if_else(is_same_pitch, pfx_x,             avg_pfx_x),
    pfx_z             = if_else(is_same_pitch, pfx_z,             avg_pfx_z),
    release_spin_rate = if_else(is_same_pitch, release_spin_rate, avg_release_spin),
    spin_axis         = if_else(is_same_pitch, spin_axis,         avg_spin_axis),
    release_extension = if_else(is_same_pitch, release_extension, avg_release_extension)
  ) |>
  select(-pitch_name.x, -pitch_name.y, -is_same_pitch)

### PREDICTION MATRIX ### 

pred_data <- candidates[,features]

pred_data$pitch_name <- as.factor(pred_data$pitch_name)
pred_data$stand <- as.factor(pred_data$stand)
pred_data$p_throws <- as.factor(pred_data$p_throws)

chunk_size <- 100000
n <- nrow(pred_data)

preds <- numeric(n)

pb <- txtProgressBar(min = 0, max = ceiling(n/chunk_size), style = 3)

for(i in seq(1,n,by=chunk_size)){
  end <- min(i+chunk_size-1,n)
  preds[i:end] <- predict(
    model,
    data.matrix(pred_data[i:end,])
  )
  setTxtProgressBar(pb, ceiling(end/chunk_size))
}

close(pb)

rm(pred_data)
gc()

candidates$predicted_run_value <- preds
rm(preds)

# drops unnecessary columns
candidates <- candidates |>
  select(row_id, actual_pitch, pitch_name, predicted_run_value)

gc()

recommended <- candidates |>
  group_by(row_id) |>
  slice_min(predicted_run_value, n = 1) |>
  ungroup() |>
  select(row_id, recommended_pitch = pitch_name, predicted_run_value)

actual_pred <- candidates |>
  filter(actual_pitch == pitch_name) |>
  select(row_id, actual_predicted_run_value = predicted_run_value)

rm(candidates)
gc()

### MERGE BACK W/ ORIGINAL ###

results <- test |>
  left_join(recommended, by = "row_id") |>
  left_join(actual_pred, by = "row_id")

### EVALUATE DECISIONS ### 

results <- results |>
  mutate(
    optimal_pitch = pitch_name == recommended_pitch,
    runs_lost = actual_predicted_run_value - predicted_run_value
  )

# get names
players <- chadwick_player_lu() |>
  select(key_mlbam, name_first, name_last)

### PITCHER EVALUATION ### 

pitcher_eval <- results |>
  group_by(pitcher) |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE)
  ) |>
  left_join(players, by = c("pitcher" = "key_mlbam")) |>
  mutate(
    player_name = paste(name_first, name_last)
  ) |>
  select(player_name, everything())

### CATCHER EVALUATION ### 

catcher_eval <- results |>
  group_by(fielder_2) |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE)
  ) |>
  left_join(players, by = c("fielder_2" = "key_mlbam")) |>
  mutate(
    player_name = paste(name_first, name_last)
  ) |>
  select(player_name, everything())


### TEAM EVALUATION ###

team_eval <- results |>
  mutate(
    team = if_else(inning_topbot == "Top", home_team, away_team)
  ) |>
  group_by(team) |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE)
  )

### MAKE TABLES OF LEADERBOARDS ###

library(knitr)
library(kableExtra)

pitcher_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  filter(pitches >= 10) |>
  arrange(per_pitch) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Player = player_name,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "pitcher_eval_best.png"))

pitcher_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  filter(pitches >= 10) |>
  arrange(desc(per_pitch)) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Player = player_name,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "pitcher_eval_worst.png"))

catcher_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  filter(pitches >= 10) |>
  arrange(per_pitch) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Player = player_name,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "catcher_eval_best.png"))

catcher_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  filter(pitches >= 10) |>
  arrange(desc(per_pitch)) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Player = player_name,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "catcher_eval_worst.png"))

team_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  arrange(per_pitch) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Team = team,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "team_eval_best.png"))

team_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  arrange(desc(per_pitch)) |>
  head(3) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Team = team,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "team_eval_worst.png"))

table_data <- team_eval |>
  mutate(
    per_pitch = runs_lost / pitches
  ) |>
  arrange(desc(per_pitch)) |>
  mutate(
    pct_optimal = sprintf("%.4f", pct_optimal),
    runs_lost = sprintf("%.4f", runs_lost),
    per_pitch = sprintf("%.4f", per_pitch)
  ) |>
  select(
    Team = team,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  )

top3 <- table_data |> head(3)
bottom3 <- table_data |> tail(3)

spacer <- tibble(
  Team = "...",
  `Optimal %` = "...",
  `Runs Lost` = "...",
  `Per Pitch` = "..."
)

team <- bind_rows(top3, spacer, bottom3)

team |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "team_eval.png"))

battery_eval_2 <- results |>
  group_by(pitcher, fielder_2) |>
  mutate(games_together = n_distinct(game_date)) |>
  filter(games_together >= 3) |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE),
    .groups = "drop"
  ) |>
  left_join(players, by = c("pitcher" = "key_mlbam")) |>
  mutate(pitcher_name = paste(name_first, name_last)) |>
  select(-name_first, -name_last) |>
  left_join(players, by = c("fielder_2" = "key_mlbam")) |>
  mutate(catcher_name = paste(name_first, name_last)) |>
  select(-name_first, -name_last) |>
  mutate(battery = paste(pitcher_name, "&", catcher_name)) |>
  select(Battery = battery, Pitches = pitches, `Optimal %` = pct_optimal, 
         `Runs Lost` = runs_lost)

battery_eval_2 |>
  mutate(
    `Per Pitch` = `Runs Lost` / Pitches,
    `Optimal %` = round(`Optimal %`, 4),
    `Runs Lost` = round(`Runs Lost`, 4),
    `Per Pitch` = round(`Per Pitch`, 4)
  ) |>  
  arrange(desc(`Per Pitch`)) |>
  head(3) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "battery_eval_worst.png"))

battery_eval_2 |>
  mutate(
    `Per Pitch` = `Runs Lost` / Pitches
  ) |>  
  arrange(`Per Pitch`) |>   # best = lowest runs lost
  head(3) |>
  mutate(
    `Optimal %` = sprintf("%.4f", `Optimal %`),
    `Runs Lost` = sprintf("%.4f", `Runs Lost`),
    `Per Pitch` = sprintf("%.4f", `Per Pitch`)
  ) |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "battery_eval_best.png"))

### INDIVIDUAL PITCH EVALUATION ###

library(stringr)
pitch_eval <- results |>
  left_join(players, by = c("pitcher" = "key_mlbam")) |>
  mutate(pitcher_name = paste(name_first, name_last)) |>
  select(-name_first, -name_last) |>
  left_join(players, by = c("fielder_2" = "key_mlbam")) |>
  mutate(catcher_name = paste(name_first, name_last)) |>
  select(-name_first, -name_last) |>
  left_join(players, by = c("batter" = "key_mlbam")) |>
  mutate(batter_name = paste(name_first, name_last)) |>
  select(-name_first, -name_last)

pitch_eval |>
  arrange(desc(runs_lost)) |>
  head(5) |>
  select(game_date, inning, pitcher_name, catcher_name, batter_name, runs_lost, pitch_type, description) |>
  kable(format = "latex", booktabs = TRUE) |>
  column_spec(1, width = "2cm") |>
  column_spec(2, width = "0.75cm") |>
  column_spec(3, width = "3.25cm") |>
  column_spec(4, width = "2.75cm") |>
  column_spec(5, width = "2.75cm") |>
  column_spec(6, width = "1.5cm") |>
  column_spec(7, width = "1.5cm") |>
  column_spec(8, width = "2cm") |>
  save_kable(file.path(getwd(), "pitch_worst.png"))

library(ggplot2)
library(latex2exp)

theme_set(theme_minimal(base_family = "serif"))

### ARSENAL EFFICIENCY ###

results_viz <- results |>
  filter(pitch_name != "")
results_viz |>
  group_by(pitch_name) |>
  summarise(
    pitches = n(),
    avg_runs_lost = mean(runs_lost, na.rm = TRUE)
  ) |>
  filter(pitches >= 500) |>
  filter(pitch_name != "Forkball") |>
  arrange(avg_runs_lost) |>
  mutate(pitch_name = factor(pitch_name, levels = pitch_name)) |>
  ggplot(aes(x = avg_runs_lost, y = pitch_name)) +
  geom_segment(aes(x = 0, xend = avg_runs_lost, y = pitch_name, 
                   yend = pitch_name), color = "grey60") +
  geom_point(size = 4, color = "steelblue") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey40") +
  labs(
    title = "Average Runs Lost by Pitch Type",
    subtitle = "Lower = better pitch selection relative to optimal",
    x = "Avg Runs Lost Per Pitch",
    y = NULL
  ) +
  theme_minimal() +
  scale_y_discrete(expand = expansion(add = 0.8)) +
  coord_cartesian(clip = "off")

ggsave(file.path(getwd(), "arsenal_efficiency.png"), width = 8, height = 5)

### OPTIMAL% BY COUNT ### 

results_viz |>
  mutate(count = paste0(balls, "-", strikes)) |>
  group_by(count, balls, strikes) |>
  summarise(
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    pitches = n(),
    .groups = "drop"
  ) |>
  ggplot(aes(x = strikes, y = balls, fill = pct_optimal)) +
  geom_tile(color = "white", linewidth = 1.5) +
  geom_text(aes(label = paste0(round(pct_optimal * 100, 1), "%\n(", scales::comma(pitches), ")")),
            size = 3.5, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "firebrick", high = "steelblue", labels = scales::percent) +
  scale_x_continuous(breaks = 0:2, labels = c("0 strikes", "1 strike", "2 strikes")) +
  scale_y_continuous(breaks = 0:3, labels = c("0 balls", "1 ball", "2 balls", "3 balls")) +
  labs(
    title = "Optimal Pitch Rate by Count",
    subtitle = "Blue = higher rate of optimal pitch selection",
    x = "Strikes",
    y = "Balls",
    fill = "% Optimal"
  ) +
  theme_minimal() +
  theme(panel.grid = element_blank())

ggsave(file.path(getwd(), "optimal_by_count.png"), width = 7, height = 6)

### RUNS LOST TREND OVER TIME ### 

results_viz |>
  mutate(
    team = if_else(inning_topbot == "Top", home_team, away_team),
    game_date = as.Date(game_date)
  ) |>
  group_by(team, game_date) |>
  summarise(daily_runs_lost = sum(runs_lost, na.rm = TRUE), .groups = "drop") |>
  arrange(team, game_date) |>
  group_by(team) |>
  mutate(cumulative_runs_lost = cumsum(daily_runs_lost)) |>
  ungroup() |>
  ggplot(aes(x = game_date, y = cumulative_runs_lost, group = team)) +
  geom_line(alpha = 0.4, color = "steelblue") +
  labs(
    title = "Cumulative Runs Lost by Team Over 2025 Season",
    subtitle = "Each line represents one team",
    x = "Date",
    y = "Cumulative Runs Lost"
  ) +
  theme_minimal()

ggsave(file.path(getwd(), "team_runs_lost_over_time.png"), width = 10, height = 6)

### RECOMMENDED VS ACTUAL PITCH ###

results_viz |>
  filter(!is.na(recommended_pitch)) |>
  group_by(pitch_name, recommended_pitch) |>
  summarise(n = n(), .groups = "drop") |>
  group_by(pitch_name) |>
  mutate(pct = n / sum(n)) |>
  ungroup() |>
  ggplot(aes(x = pitch_name, y = recommended_pitch, fill = pct)) +
  geom_tile(color = "white", linewidth = 0.5) +
  geom_text(aes(label = paste0(round(pct * 100, 1), "%")),
            size = 2.5, color = "white", fontface = "bold") +
  scale_fill_gradient(low = "grey90", high = "steelblue", labels = scales::percent) +
  labs(
    title = "Actual vs. Recommended Pitch Type",
    x = "Actual Pitch Thrown",
    y = "Recommended Pitch",
    fill = "% of Pitches"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(file.path(getwd(), "actual_vs_recommended.png"), width = 9, height = 7)

### OPTIMAL% PIE CHART ###

results |>
  summarise(
    Optimal = mean(optimal_pitch, na.rm = TRUE),
    `Not Optimal` = 1 - mean(optimal_pitch, na.rm = TRUE)
  ) |>
  tidyr::pivot_longer(everything(), names_to = "category", values_to = "pct") |>
  ggplot(aes(x = "", y = pct, fill = category)) +
  geom_col(width = 1, color = "white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = paste0(round(pct * 100, 1), "%")),
            position = position_stack(vjust = 0.5),
            size = 5, color = "white", fontface = "bold") +
  scale_fill_manual(values = c("Optimal" = "steelblue", "Not Optimal" = "firebrick")) +
  labs(
    fill = NULL
  ) +
  theme_void() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    legend.position = "right"
  )

ggsave(file.path(getwd(), "optimal_pie.png"), width = 6, height = 6)

### OPTIMAL% BY TEAM BAR CHART ###

results |>
  mutate(team = if_else(inning_topbot == "Top", home_team, away_team)) |>
  group_by(team) |>
  summarise(
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    pct_not_optimal = 1 - mean(optimal_pitch, na.rm = TRUE)
  ) |>
  arrange(desc(pct_optimal)) |>
  mutate(team = factor(team, levels = team)) |>
  tidyr::pivot_longer(cols = c(pct_optimal, pct_not_optimal),
                      names_to = "category", values_to = "pct") |>
  mutate(category = if_else(category == "pct_optimal", "Optimal", "Not Optimal"),
         category = factor(category, levels = c("Optimal", "Not Optimal"))) |>
  ggplot(aes(x = team, y = pct, fill = category)) +
  geom_col(width = 0.8) +
  scale_fill_manual(values = c("Optimal" = "steelblue", "Not Optimal" = "firebrick")) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Optimal Pitch Selection Rate by Team",
    subtitle = "Sorted by highest to lowest optimal rate",
    x = NULL,
    y = "% of Pitches",
    fill = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "bottom"
  )

ggsave(file.path(getwd(), "optimal_by_team.png"), width = 12, height = 6)


### HYPOTHESIS TESTING ###

# t-test
t.test(results$runs_lost, mu = 0, alternative = "greater")

pitcher_game <- results |>
  group_by(pitcher, game_date) |>
  summarise(avg_runs_lost = sum(runs_lost), .groups = "drop")

t.test(pitcher_game$avg_runs_lost, mu = 0, alternative = "greater")

t.test(runs_scored ~ optimal_pitch, data = results)

### DISTRIBUTION OF RUNS LOST PER PITCH ###
ggplot(results, aes(x = runs_lost)) +
  geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "solid", color = "red", linewidth = 1) +
  labs(
    title = "Distribution of Runs Lost per Pitch",
    subtitle = "Red line = zero (no difference)",
    x = "Runs Lost (Actual - Optimal)",
    y = "Count"
  ) + xlim(-.01, .05) +
  theme_minimal()

### AVG RUNS LOST PER APPEARANCE ###
ggplot(pitcher_game, aes(x = avg_runs_lost)) +
  geom_histogram(bins = 100, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Average Runs Lost per Appearance",
    x = "Average Runs Lost",
    y = "Count"
  ) + xlim(-.01, 2) +
  theme_minimal()

### AVG RUNS SCORED CONFIDENCE INTERVAL ON OPTIMAL VS NON OPTIMAL PITCH ###
summary_df <- results |>
  group_by(optimal_pitch) |>
  summarise(
    mean_runs = mean(runs_scored, na.rm = TRUE),
    se = sd(runs_scored, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) |>
  mutate(
    lower = mean_runs - 1.96 * se,
    upper = mean_runs + 1.96 * se
  )

ggplot(summary_df, aes(x = factor(optimal_pitch), y = mean_runs)) +
  geom_point(size = 4, color = "steelblue") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.1) +
  labs(
    title = "Average Runs Scored: Optimal vs Not Optimal",
    subtitle = "Error bars = 95% confidence intervals",
    x = "",
    y = "Average Runs per Pitch"
  ) +
  scale_x_discrete(labels = c("FALSE" = "Not Optimal", "TRUE" = "Optimal")) +
  theme_minimal()


### CHECK MARLINS' DUGOUT WINDOW AGAINST LEAGUE AVERAGE ###

# sets up batting team
results_dates <- results |>
  mutate(
    team = if_else(inning_topbot == "Top", home_team, away_team),
    game_date = as.Date(game_date)
  )

# filters only to window when Marlins called pitches from dugout
window_data <- results_dates |>
  filter(game_date >= as.Date("2025-09-19"),
         game_date <= as.Date("2025-09-28"))

# Marlins performance whne calling pitches from dugout
marlins_window <- window_data |>
  filter(team == "MIA") |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE),
    per_pitch = runs_lost / pitches
  )

# league average over same window
league_window <- window_data |>
  filter(team != "MIA") |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE),
    per_pitch = runs_lost / pitches
  )

comparison_1 <- bind_rows(
  marlins_window |> mutate(Group = "Marlins"),
  league_window |> mutate(Group = "League Avg")
) |>
  mutate(
    `Per Pitch` = runs_lost / pitches
  ) |>
  select(
    Group,
    Pitches = pitches,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch`
  ) |>
  mutate(
    `Optimal %` = sprintf("%.3f", `Optimal %`),
    `Runs Lost` = sprintf("%.2f", `Runs Lost`),
    `Per Pitch` = sprintf("%.4f", `Per Pitch`)
  )

comparison_1 |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "marlins_vs_league_avg.png"))

### CHECK MARLINS' WINDOW PRE-DUGOUT VS WHEN IMPLEMENTING DUGOUT ###

marlins_split <- results_dates |>
  filter(team == "MIA") |>
  mutate(period = if_else(game_date >= as.Date("2025-09-19") &
                            game_date <= as.Date("2025-09-28"),
                          "Sept 19-28",
                          "Before Sept 19"))

marlins_comparison <- marlins_split |>
  group_by(period) |>
  summarise(
    pitches = n(),
    pct_optimal = mean(optimal_pitch, na.rm = TRUE),
    runs_lost = sum(runs_lost, na.rm = TRUE),
    per_pitch = runs_lost / pitches,
    .groups = "drop"
  ) |>
  select(
    Period = period,
    Pitches = pitches,
    `Optimal %` = pct_optimal,
    `Runs Lost` = runs_lost,
    `Per Pitch` = per_pitch
  ) |>
  mutate(
    `Optimal %` = sprintf("%.3f", `Optimal %`),
    `Runs Lost` = sprintf("%.2f", `Runs Lost`),
    `Per Pitch` = sprintf("%.4f", `Per Pitch`)
  )

marlins_comparison |>
  kable(format = "latex", booktabs = TRUE) |>
  save_kable(file.path(getwd(), "marlins_pre_vs_during.png"))