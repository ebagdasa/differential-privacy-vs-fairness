setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")
grouped_results <- read_data("./data/mnist-grouped", "attr")
lim = max(c(grouped_results$L_0, grouped_results$L_1))

p1 <- grouped_results %>%
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
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MMNIST")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(1.5)),
        axis.text = element_text(size=rel(1)),
        strip.text = element_text(size=rel(1.5)), # size of the facet label text
        # panel.grid.major = element_blank(), 
        # panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  # xlim(0, lim) +
  # ylim(0, lim) + 
  # labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")
# ggsave("./L0_vs_L1_mnist_grouped.pdf", device="pdf", height=8, width=16)


cifar10_results <- read_data("./data/cifar10-grouped/0-1-vs-9-2", "attr")



p2 <- cifar10_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MC10")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(1.5)),
        axis.text = element_text(size=rel(1)),
        strip.text = element_text(size=rel(1.5)), # size of the facet label text
        # panel.grid.major = element_blank(), 
        # panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  # xlim(0, lim) +
  # ylim(0, lim) + 
  # labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")

g = arrangeGrob(p1, p2, nrow=1)
ggsave("L0_vs_L1_mmnist_and_cifar10.pdf", g, width=13, height=4)
