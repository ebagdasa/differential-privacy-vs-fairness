setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")

phi_alpha_results <- read.csv("/Users/jpgard/Documents/github/differential-privacy-vs-fairness/rscripts/data/lr-phi-alpha/lr_phi_alpha.csv")
phi_alpha_results$eps_fct = factor(as.character(phi_alpha_results$eps))
levels(phi_alpha_results$eps_fct) <- c("epsilon == 12.5", "epsilon == 25", 
                                        "epsilon == 50")

phi_alpha_results %>%
  ggplot(aes(x=alpha, y=phi_alpha, col=h_maj, group=factor(h_maj))) + 
  geom_point(size=2.5) + 
  geom_line() +
  # ylim(0,1) +
  facet_grid(cols=vars(eps_fct), labeller=label_parsed) + 
  theme_bw() +
  ylab(TeX("$\\phi(\\alpha)")) +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  labs(col=TeX("Value of Diagonal Entries of $H_1$"),
       caption=TeX("Fixed $\\delta = 10^{-5}$ for all experiments")) +
  ggtitle(TeX("Disparity Measure $\\phi(\\alpha)$ vs. $\\alpha$")) +
  theme(plot.title = element_text(hjust=0.5),
        legend.position = "bottom")


ggsave("./phi_alpha_lr.pdf", width=16, height=7, device="pdf")
