setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")

cifar10_results <- read_data("./data/cifar10-grouped/0-1-vs-9-2", "attr")



cifar10_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MC10 Dataset")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(4)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        strip.text = element_text(size=rel(3)), # size of the facet label text
        # panel.grid.major = element_blank(), 
        # panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  # xlim(0, lim) +
  # ylim(0, lim) + 
  labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")
ggsave("./L0_vs_L1_cifar10_grouped.pdf", device="pdf", height=8, width=16)


# Notes: what happens here is that, as alpha becomes very extreme, the loss becomes so large for
# both DP and non_DP on the underrepresented group, that it effectively cannot become larger.
# This is an artifact of the classification loss used; it does not apply to the squared
# loss used in our regression experiments.
cifar10_results %>%
  dplyr::filter(dp_str == "DP-SGD, S = 1, z = 0.8" | dp_str == "No DP") %>%
  select(c("alpha", "dp", "L_0", "L_1")) %>%
  tidyr::pivot_wider(id_cols=c(alpha), values_from = c(L_0, L_1), names_from=dp) %>%
  dplyr::mutate(rho_alpha = (L_0_TRUE - L_1_TRUE)/(L_0_FALSE - L_1_FALSE)) %>%
  dplyr::mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  ggplot(aes(x=as.numeric(alpha), y = rho_alpha)) +
  geom_line(group=1, col="dodgerblue") +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  geom_point() +
  geom_vline(xintercept = 0.5, col="orange", lty="dashed") +
  ylim(0, 5) +
  ylab(TeX("$\\rho(\\alpha)$")) + 
  xlab(TeX("Majority Fraction  $\\alpha$")) + 
  ggtitle(TeX("$\\rho(\\alpha)$ vs. Majority Fraction $\\alpha$")) + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5))) +
  labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") 
ggsave("./rho_alpha_vs_alpha_cifar10_grouped.pdf", device="pdf", height=8, width=10)
