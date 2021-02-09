setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")

results <- read_data("./data/mnist", "class")


L_0_lim = max(results$L_0) * 1.1
results %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha)) + 
  geom_point() + 
  geom_line(col="dodgerblue") +
  geom_text(aes(label=paste0("\u03B1 = ", alpha)),hjust=-0.1, vjust=0, size=4, col="grey40") +
  ggtitle("L_0 vs. L_1 with Fixed Sample Size") +
  xlab("Loss on Minority (L_0)") +
  ylab("Loss on Manjority (L_1)") +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()) +
  xlim(0, L_0_lim) +
  labs(caption="Task: MNIST 3 vs. 8 classification")


# Read the grouped MNIST results

grouped_results <- read_data("./data/mnist-grouped", "attr")
lim = max(c(grouped_results$L_0, grouped_results$L_1))

grouped_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  dplyr::filter(z < 1) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
    ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MMNIST Dataset")) +
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
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")
ggsave("./L0_vs_L1_mnist_grouped.pdf", device="pdf", height=8, width=16)


# Notes: what happens here is that, as alpha becomes very extreme, the loss becomes so large for
# both DP and non_DP on the underrepresented group, that it effectively cannot become larger.
# This is an artifact of the classification loss used; it does not apply to the squared
# loss used in our regression experiments.
grouped_results %>%
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
  ylim(0, 30) +
  ylab(TeX("$\\rho(\\alpha)$")) + 
  xlab(TeX("Majority Fraction  $\\alpha$")) + 
  ggtitle(TeX("$\\rho(\\alpha)$ vs. Majority Fraction $\\alpha$ on MMNIST Dataset")) + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5))) +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)") 
ggsave("./rho_alpha_vs_alpha_mnist_grouped.pdf", device="pdf", height=8, width=10)

phi_results = read_data_2("./data/mnist-grouped", "attr") %>%
  dplyr::select(c("alpha", "phi8", "phi1")) %>%
  tidyr::pivot_longer(c(phi8, phi1), names_to="metric") 
phi_results$metric <- forcats::fct_recode(phi_results$metric, "DP-SGD with Noise Multiplier z=0.8"="phi8",
                                          "DP-SGD with Noise Multiplier z=1.0"="phi1")



phi_results %>%
  ggplot(aes(x=alpha, y=value)) + 
  geom_point() + 
  geom_line(aes(group=1), col="dodgerblue") + 
  geom_hline(yintercept = 1, col = "orange", linetype="dashed") +
  theme_bw() +
  ylab(TeX("$\\phi(\\alpha)$")) +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  facet_grid(. ~ metric) +
  annotation_logticks() +
  scale_y_log10(breaks = log_breaks()) +
  ggtitle(TeX("Disparity Measure $\\phi(\\alpha)$ Under Different Privacy Levels")) +
  theme(plot.title=element_text(hjust=0.5))
ggsave("./phi_alpha.pdf", height=5, width=10, device="pdf")
