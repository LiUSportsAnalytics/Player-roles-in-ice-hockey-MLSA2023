library(ggplot2)

windowsFonts(serif = windowsFont("Times New Roman"))

library(extrafont)
loadfonts(device = "win")

theme_custom <- function(){
  # Specify font
  font <- "sans"
  
  # Start with a base theme
  theme_classic() %+replace%    
    
    # Customization
    theme(
      
      # Text font
      text = element_text(family=font, color="black"),
      
      # Blank panel background
      panel.grid = element_blank(), 
      
      # Remove axis ticks
      axis.ticks = element_line(color="#A9A9A9"), 
      
      # Gray axis line
      axis.line = element_line(color="#A9A9A9"),
      
      # Plot titles
      plot.title = element_text(size=14, face="bold", hjust=0.5, vjust=2),               
      plot.subtitle = element_text(size=12, hjust=0.5), 
      plot.caption = element_text(size=9, hjust=1),
      
      # Facet text/label
      strip.background = element_rect(fill="white", color="white"),
      strip.text = element_text(size=12, face="bold"),
      
      # Axis text
      axis.title = element_text(size=10),
      axis.title.y = element_text(margin=margin(t=0, r=5, b=0, l=0), angle=90),
      axis.text = element_text(size=8),
      
      # Legend defaults
      legend.title = element_text(size=10),
      legend.text = element_text(size=10),
    )
}
  