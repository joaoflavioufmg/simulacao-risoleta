# -------------------------------------------------
# Libraries
# -------------------------------------------------
library(bupaR)
library(processanimateR)
library(ggplot2)
library(dplyr)

# -------------------------------------------------
# 1. Manchester Triage Priority Colors
# -------------------------------------------------
priority_colors <- c(
  "Emergency"   = "#d73027",   # Priority 1 - Red
  "Very Urgent" = "#fc8d59",   # Priority 2 - Orange
  "Urgent"      = "#fee08b",   # Priority 3 - Yellow
  "Standard"    = "#1a9850",   # Priority 4 - Green
  "Non-Urgent"  = "#4575b4",   # Priority 5 - Blue
  "Unclassified"= "#000000"    # Genérico / antes da classificação no PS → PRETO
)

# Numeric priority -> Manchester label
priority_mapping <- c(
  "1" = "Emergency",
  "2" = "Very Urgent",
  "3" = "Urgent",
  "4" = "Standard",
  "5" = "Non-Urgent",
  "6" = "Unclassified"   # ou "White" se quiser manter o nome antigo
)

# -------------------------------------------------
# 2. Color Verification
# -------------------------------------------------
color_verification <- data.frame(
  priority = factor(
    names(priority_colors),
    levels = names(priority_colors),
    ordered = TRUE
  ),
  color = unname(priority_colors),
  order = seq_along(priority_colors)
)

verification_plot <- ggplot(
  color_verification,
  aes(x = reorder(priority, order), y = 1, fill = priority)
) +
  geom_col(color = "black") +
  scale_fill_manual(
    values = priority_colors,
    drop = FALSE
  ) +
  labs(
    title = "Manchester Triage Priority Colors",
    subtitle = "Priority 1 (highest) to Unclassified (black – before PS)",
    x = "Priority Level",
    y = ""
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  )

print(verification_plot)

# -------------------------------------------------
# 3. Load Data
# -------------------------------------------------
event_log <- read.csv(
  "hrtn_event_log.csv",
  quote = "\"",
  stringsAsFactors = FALSE,
  check.names = FALSE
)

event_log$resource <- trimws(as.character(event_log$resource))

# -------------------------------------------------
# 4. Create Activity Instance ID
# -------------------------------------------------
event_log$activity_instance_id <- paste(
  event_log$case_id,
  event_log$activity,
  event_log$timestamp,
  sep = "_"
)

# -------------------------------------------------
# 5. Convert Timestamp
# -------------------------------------------------
reference_date <- as.POSIXct("2026-01-01 00:00:00", tz = "UTC")

event_log$timestamp <- as.numeric(event_log$timestamp)
event_log$timestamp <- reference_date + as.difftime(event_log$timestamp, units = "mins")

# -------------------------------------------------
# 6. Manchester Priority Mapping  ← AQUI ESTÁ O AJUSTE PRINCIPAL
# -------------------------------------------------
if ("priority" %in% names(event_log)) {
  event_log$priority_char <- trimws(as.character(event_log$priority))
  event_log$priority_label <- unname(priority_mapping[event_log$priority_char])
  
  # Fallback para valores ausentes / inesperados → Unclassified (preto)
  event_log$priority_label[is.na(event_log$priority_label)] <- "Unclassified"
} else {
  event_log$priority_label <- "Unclassified"
}

# Ordenação clínica
event_log$priority_label <- factor(
  event_log$priority_label,
  levels = names(priority_colors),
  ordered = TRUE
)

# -------------------------------------------------
# 7. Create Event Log
# -------------------------------------------------
ex1_log <- eventlog(
  event_log,
  case_id               = "case_id",
  activity_id           = "activity",
  activity_instance_id  = "activity_instance_id",
  lifecycle_id          = "lifecycle",
  timestamp             = "timestamp",
  resource_id           = "resource"
)

# -------------------------------------------------
# Reduce size
# -------------------------------------------------
n_cases_to_keep <- 2000

last_cases <- ex1_log %>%
  cases() %>%
  arrange(desc(start_timestamp)) %>%
  slice_head(n = n_cases_to_keep) %>%
  pull(case_id)

ex1_log_small <- ex1_log %>%
  filter_case(cases = last_cases)

# -------------------------------------------------
# 8. Animation
# -------------------------------------------------
time_range <- attr(ex1_log, "time_range")   # ou use o time_range do log pequeno se preferir
start_time <- time_range[1]
end_time   <- time_range[2]

final_animation <- animate_process(
  ex1_log_small,
  mode = "absolute",
  duration = 120,
  start_time = start_time,
  end_time = end_time,
  mapping = token_aes(
    color = token_scale(
      "priority_label",
      scale  = "ordinal",
      domain = names(priority_colors),
      range  = unname(priority_colors)
    )
  ),
  legend = "color",
  timeline = TRUE
)

print(final_animation)

# -------------------------------------------------
# 9. Save Animation
# -------------------------------------------------
htmlwidgets::saveWidget(
  final_animation,
  "hrtn.html",
  selfcontained = TRUE
)

cat("Manchester priority colors:\n")
print(priority_colors)