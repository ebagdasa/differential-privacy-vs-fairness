library(ggplot2)
library(ggthemes)

results <- data.frame(
  attr_val=c(0,1,0,1,0,1),
  crossentropy_loss=c(0.1605, 0.3887, 0.8683, 1.236, 0.996, 1.555),
  uid=c("Non-DP", "Non-DP", 
        "S=1 z=0.8\neps=13.0", "S=1 z=0.8\neps=13.0", 
        "S=1 z=1.0\neps=8.06", "S=1 z=1.0\neps=8.06"))

results$attr_val = forcats::fct_recode(factor(results$attr_val), "Majority"="0", "Minority"="1")

ggplot(results, 
       aes(group=attr_val, 
           y=crossentropy_loss, 
           x=uid, 
           fill=factor(attr_val))) +
  geom_bar(stat="identity", position="dodge", colour="black") +
  xlab("Model") +
  ylab("Loss") +
  labs(fill="Test Subset") +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        axis.text.x = element_text(size=rel(1.5)),
        # panel.grid.major = element_blank(), 
        axis.title = element_text(size=rel(1.5)),
        # panel.grid.minor = element_blank(),
        plot.caption = element_text(hjust = 0.5)) +
  labs(caption = TeX("LFW Dataset, minority attribute label: Black $(\\alpha = 0.978)$")) + 
  ggtitle("Subgroup Loss vs. Model Type") +
  scale_fill_manual(values=c("#E69F00", "#56B4E9"))
ggsave("lfw_loss_vs_model.pdf", device="pdf", width=7, height=4)

# celeba:
# celeba-S1-z1.0-sigmaNone-alpha-None-adaFalse-dpTrue-nNone-Smiling-Blond_Hair-trattrsub1-freezept
# eps = 4.36 and delta = 1e-06

# celeba-S1-z1.0-sigmaNone-alpha-None-adaFalse-dpTrue-nNone-Smiling-Blond_Hair-trattrsub0-freezept
# eps = 1.8 and delta = 1e-06

# celeba-S1-z1.0-sigmaNone-alpha-None-adaFalse-dpTrue-nNone-Smiling-Blond_Hair-freezept
# eps = 1.69 and delta = 1e-06