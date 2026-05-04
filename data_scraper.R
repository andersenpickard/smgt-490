library(data.table)
library(glue)
library(baseballr)
library(dplyr)

statcast_savant_weekly <- function(year,
                                   start_date,
                                   end_date,
                                   player_type = "batter",
                                   sleep_time = 1) {
  
  # Create weekly date breaks
  date_seq <- seq.Date(as.Date(start_date),
                       as.Date(end_date),
                       by = "7 days")
  
  if (tail(date_seq, 1) < as.Date(end_date)) {
    date_seq <- c(date_seq, as.Date(end_date))
  }
  
  all_data <- list()  # store weekly pulls
  idx <- 1
  
  for (i in seq_len(length(date_seq) - 1)) {
    
    start <- date_seq[i]
    end   <- date_seq[i + 1] - 1
    
    end_exclusive <- as.character(as.Date(end) + 1)
    
    url <- glue(
      "https://baseballsavant.mlb.com/statcast_search/csv?",
      "all=true&hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&",
      "hfGT=R%7CPO%7CS%7C&hfC=&hfSea={year}%7C&hfSit=&hfOuts=&opponent=&",
      "pitcher_throws=&batter_stands=&hfSA=&player_type={player_type}&",
      "hfInfield=&team=&position=&hfOutfield=&hfRO=&home_road=&",
      "game_date_gt={start}&game_date_lt={end_exclusive}&",
      "hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&",
      "group_by=name&sort_col=pitches&",
      "player_event_sort=h_launch_speed&sort_order=desc&min_abs=0&type=details"
    )
    
    message("Pulling: ", start, " → ", end)
    
    dt <- tryCatch(
      fread(url),
      error = function(e) {
        message("Failed: ", start, " → ", end)
        return(NULL)
      }
    )
    
    if (!is.null(dt) && nrow(dt) > 0) {
      all_data[[idx]] <- dt
      idx <- idx + 1
    }
    
    Sys.sleep(sleep_time)  # avoid Savant rate limits
  }
  
  # Bind all weeks into one data frame
  if (length(all_data) == 0) {
    return(data.table())
  } else {
    return(rbindlist(all_data, fill = TRUE))
  }
}

### THE FIVE FUNCTION CALLS BELOW TAKE ABOUT 90 MINUTES TO RUN IN TOTAL ###
### OR ABOUT 18 MINUTES EACH IF YOU RUN THEM INDIVIDUALLY ###
### SAVE TIME: CHANGE START TO OPENING DAY & END TO LAST DAY OF REG SEASON ###

statcast_2021 <- statcast_savant_weekly(
  year = 2021,
  start_date = "2021-01-01",
  end_date   = "2021-12-31"
)

statcast_2022 <- statcast_savant_weekly(
  year = 2022,
  start_date = "2022-01-01",
  end_date   = "2022-12-31"
)

statcast_2023 <- statcast_savant_weekly(
  year = 2023,
  start_date = "2023-01-01",
  end_date   = "2023-12-31"
)

statcast_2024 <- statcast_savant_weekly(
  year = 2024,
  start_date = "2024-01-01",
  end_date   = "2024-12-31"
)

statcast_2025 <- statcast_savant_weekly(
  year = 2025,
  start_date = "2025-01-01",
  end_date   = "2025-12-31"
)

smgt_statcast <- rbind(statcast_2021, statcast_2022, statcast_2023, 
                       statcast_2024, statcast_2025) |>
  filter(game_type == "R")

write.csv(smgt_statcast, "smgt_statcast.csv")