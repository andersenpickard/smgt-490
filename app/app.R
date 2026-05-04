library(shiny)
library(dplyr)
library(tidyr)
library(DT)
library(baseballr)
library(lightgbm)

model <- lgb.load("pitch_model.txt")
pitch_summary <- readRDS("pitch_summary.rds")

features <- c(
  "balls", "strikes", "outs_when_up",
  "on_1b", "on_2b", "on_3b",
  "bat_score_diff",
  "stand", "p_throws",
  "pitch_name",
  "release_speed", "pfx_x", "pfx_z",
  "release_spin_rate", "spin_axis",
  "release_extension"
)

### SHINY APP ###

pitcher_choices <- pitch_summary |>
  filter(!is.na(pitcher_name)) |>
  distinct(pitcher_name) |>
  arrange(pitcher_name) |>
  pull(pitcher_name)

ui <- fluidPage(
  titlePanel("Pitch Recommendation Model"),
  
  sidebarLayout(
    sidebarPanel(
      selectInput(
        "pitcher_name",
        "Pitcher Name",
        choices = pitcher_choices,
        selected = pitcher_choices[1]
      ),
      
      selectInput("balls", "Balls", choices = 0:3, selected = 0),
      selectInput("strikes", "Strikes", choices = 0:2, selected = 0),
      selectInput("outs", "Outs", choices = 0:2, selected = 0),
      
      checkboxInput("on_1b", "Runner on 1B", value = FALSE),
      checkboxInput("on_2b", "Runner on 2B", value = FALSE),
      checkboxInput("on_3b", "Runner on 3B", value = FALSE),
      
      numericInput(
        "bat_score_diff",
        "Score Differential From Batting Team Perspective",
        value = 0,
        step = 1
      ),
      
      selectInput(
        "stand",
        "Batter Handedness",
        choices = c("R", "L"),
        selected = "R"
      ),
      
      selectInput(
        "game_year",
        "Season Arsenal to Use",
        choices = sort(unique(pitch_summary$game_year), decreasing = TRUE),
        selected = max(pitch_summary$game_year, na.rm = TRUE)
      ),
      
      actionButton("recommend", "Recommend Pitch")
    ),
    
    mainPanel(
      h3("Recommended Pitch"),
      verbatimTextOutput("recommendation"),
      
      h3("Run Value / Runs Lost by Pitch Type"),
      DTOutput("pitch_table"),
      
      h4("Interpretation"),
      p("Predicted Run Value is the model's expected runs scored on the pitch. 
        A lower run value is better for the pitcher, while a higher value is 
        better for the batter and worse for the pitcher. By extension, runs lost
        is calculated as each pitch's predicted run value minus the predicted
        run value of the pitcher's best possible pitch in that scenario. If a
        pitcher selects the ideal pitch, they lose zero runs. If a pitcher does
        not select the ideal pitch, they lose positive runs."))
  )
)

server <- function(input, output, session) {
  
  scored_pitches <- eventReactive(input$recommend, {
    
    pitcher_arsenal <- pitch_summary |>
      filter(
        pitcher_name == input$pitcher_name,
        game_year == as.numeric(input$game_year)
      )
    
    if (nrow(pitcher_arsenal) == 0) {
      return(NULL)
    }
    
    p_throw_value <- pitcher_arsenal |>
      count(p_throws, sort = TRUE) |>
      slice(1) |>
      pull(p_throws)
    
    candidate_data <- pitcher_arsenal |>
      transmute(
        pitch_name,
        arsenal_pct = pct,
        balls = as.numeric(input$balls),
        strikes = as.numeric(input$strikes),
        outs_when_up = as.numeric(input$outs),
        on_1b = as.integer(input$on_1b),
        on_2b = as.integer(input$on_2b),
        on_3b = as.integer(input$on_3b),
        bat_score_diff = input$bat_score_diff,
        stand = input$stand,
        p_throws = p_throw_value,
        release_speed = avg_release_speed,
        pfx_x = avg_pfx_x,
        pfx_z = avg_pfx_z,
        release_spin_rate = avg_release_spin,
        spin_axis = avg_spin_axis,
        release_extension = avg_release_extension
        )
    
    pred_data <- candidate_data[, features]
    
    pred_data$pitch_name <- as.factor(pred_data$pitch_name)
    pred_data$stand <- as.factor(pred_data$stand)
    pred_data$p_throws <- as.factor(pred_data$p_throws)
    
    candidate_data$predicted_run_value <- predict(
      model,
      data.matrix(pred_data)
    )
    
    best_rv <- min(candidate_data$predicted_run_value, na.rm = TRUE)
    
    candidate_data |>
      mutate(
        runs_lost = predicted_run_value - best_rv,
        recommendation = if_else(
          predicted_run_value == best_rv,
          "Recommended",
          ""
        )
      ) |>
      arrange(predicted_run_value) |>
      select(
        recommendation,
        pitch_name,
        arsenal_pct,
        predicted_run_value,
        runs_lost,
        release_speed,
        pfx_x,
        pfx_z,
        release_spin_rate,
        spin_axis,
        release_extension
      )
  })
  
  output$recommendation <- renderText({
    results <- scored_pitches()
    
    if (is.null(results)) {
      return("No arsenal found for this pitcher and season.")
    }
    
    best_pitch <- results |>
      slice(1)
    
    paste0(
      "Throw: ", best_pitch$pitch_name,
      "\nPredicted Run Value: ", sprintf("%.4f", best_pitch$predicted_run_value)
    )
  })
  
  output$pitch_table <- renderDT({
    results <- scored_pitches()
    
    if (is.null(results)) {
      return(datatable(data.frame(Message = "No arsenal found.")))
    }
    
    results |>
      mutate(
        arsenal_pct = sprintf("%.1f%%", arsenal_pct * 100),
        predicted_run_value = sprintf("%.4f", predicted_run_value),
        runs_lost = sprintf("%.4f", runs_lost),
        release_speed = round(release_speed, 1),
        pfx_x = round(pfx_x, 1),
        pfx_z = round(pfx_z, 1),
        release_spin_rate = round(release_spin_rate, 0),
        spin_axis = round(spin_axis, 0),
        release_extension = round(release_extension, 1)
      ) |>
      rename(
        Recommendation = recommendation,
        `Pitch Type` = pitch_name,
        `Arsenal %` = arsenal_pct,
        `Predicted Run Value` = predicted_run_value,
        `Runs Lost` = runs_lost,
        Velocity = release_speed,
        HB = pfx_x,
        IVB = pfx_z,
        Spin = release_spin_rate,
        `Spin Axis` = spin_axis,
        Extension = release_extension
      ) |>
      datatable(
        rownames = FALSE,
        options = list(
          pageLength = 10,
          dom = "tip"
        )
      )
  })
}

shinyApp(ui = ui, server = server)
