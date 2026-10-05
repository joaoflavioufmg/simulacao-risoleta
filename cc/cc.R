library(bupaR)
library(processanimateR)
library(ggplot2)
library(htmlwidgets)

# -------------------------------------------------
# Colour mapping
# -------------------------------------------------
priority_colors <- c(
  "Pacientes" = "#d73027",   # red for patients
  "Unknown"   = "#999999"    # grey for NA / other
)

# -------------------------------------------------
# 1. Load data
# -------------------------------------------------
event_log <- read.csv(
  "C:/Users/EEUFMG/Desktop/desk-risoleta/ccdet/r_animation/cc_event_log.csv",
  quote = "\"",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# -------------------------------------------------
# 2. Filter BG_ cases / activities
# -------------------------------------------------
event_log <- event_log[
  !grepl("^BG_", event_log$case_id) & 
    !grepl("^BG_", event_log$activity), 
]

# -------------------------------------------------
# 3. Clean resource
# -------------------------------------------------
event_log$resource <- trimws(as.character(event_log$resource))
event_log$resource[event_log$resource == ""] <- NA

# -------------------------------------------------
# 4. Priority label (patients = red)
# -------------------------------------------------
event_log$priority_char  <- as.character(event_log$priority)
event_log$priority_label <- ifelse(
  event_log$priority_char == "0",
  "Pacientes",
  "Unknown"
)
event_log$priority_label <- factor(
  event_log$priority_label,
  levels = names(priority_colors)
)

# -------------------------------------------------
# 5. Create activity_instance_id + convert timestamp
# -------------------------------------------------
event_log$activity_instance_id <- paste(
  event_log$case_id,
  event_log$activity,
  event_log$timestamp,
  sep = "_"
)

reference_date <- as.POSIXct("2026-01-01 00:00:00", tz = "UTC")
event_log$timestamp <- as.numeric(event_log$timestamp)
event_log$timestamp <- reference_date + as.difftime(event_log$timestamp, units = "mins")

# -------------------------------------------------
# 6. Create eventlog object
# -------------------------------------------------
hospital_log <- eventlog(
  event_log,
  case_id              = "case_id",
  activity_id          = "activity",
  activity_instance_id = "activity_instance_id",
  lifecycle_id         = "lifecycle",
  timestamp            = "timestamp",
  resource_id          = "resource"
)

# -------------------------------------------------
# 7. Time range
# -------------------------------------------------
time_range <- attr(hospital_log, "time_range")
start_time <- time_range[1]
end_time   <- time_range[2]

# -------------------------------------------------
# 8. Animation – patients in red + legend
# -------------------------------------------------
final_animation <- animate_process(
  hospital_log,
  mode       = "absolute",
  duration   = 120,
  start_time = start_time,
  end_time   = end_time,
  mapping = token_aes(
    color = token_scale(
      "priority_label",
      scale = "ordinal",
      range = unname(priority_colors)
    )
  ),
  legend   = "color",   # ← shows the legend
  timeline = TRUE
)

print(final_animation)
htmlwidgets::saveWidget(final_animation, "cc.html", selfcontained = TRUE)

cat("Animação gerada com sucesso (pacientes em vermelho + legenda)!\n")