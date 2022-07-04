# Create Circular packing visualisation of topics

library(packcircles)
library(ggplot2)

packing <- circleProgressiveLayout(COP20Counts_copy)
dat.gg <- circleLayoutVertices(packing)

ggplot(data=dat.gg) +
  geom_polygon(aes(x, y, group = id, fill = factor(id)),
               colour = "black",
               show.legend=FALSE) +
  scale_fill_manual(values = COP20Counts_copy$colour) +
  geom_text(data = packing, aes(x, y), label = COP20Counts_copy$label, size=2) +
  theme_void() + 
  theme(legend.position="none") +
  coord_equal()