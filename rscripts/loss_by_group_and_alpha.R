setwd("~/Documents/github/characterizing-rd-of-dp/rscripts")
source("utils.R")


# Read the grouped MNIST results

mmnist_results <- read_data("./data/mnist-grouped", "attr")

mmnist_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  dplyr::filter(z == 0.8, S == 1) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MMNIST Dataset"),
          subtitle=TeX("with clipping bound $\\beta = 1$, $\\sigma_{DP}=0.8$")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(4)),
        plot.subtitle=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        strip.text = element_text(size=rel(3)), # size of the facet label text
        legend.position = "bottom") +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)")
  # scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  # facet_wrap(. ~ dp_str, scales="free")
ggsave("./L0_vs_L1_mnist_grouped.pdf", device="pdf", height=8, width=16)

mmnist_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  dplyr::filter(dp==FALSE) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MMNIST Dataset"),
          subtitle=TeX("with no DP")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(4)),
        plot.subtitle=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        strip.text = element_text(size=rel(3)), # size of the facet label text
        legend.position = "bottom") +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)")


mmnist_results %>%
  mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  dplyr::arrange(desc(alpha)) %>%
  tidyr::replace_na(list(z=0, S=0)) %>%
  dplyr::filter(S == Inf) %>%
  ggplot(aes(x=L_0, y=L_1, label=alpha, group=factor(dp))) + 
  geom_path(col="dodgerblue", size=2) + 
  geom_point(size=2) +
  geom_text(
    aes(label=TeX(alpha_str, output = "character")), parse=TRUE, hjust=-0.2, vjust=-0.15, size=2.5, angle=30, col="grey40",
  ) +
  ggtitle(TeX("$L_0(\\hat{w})$ vs. $L_1\\hat{w}$ on MMNIST Dataset"),
          subtitle=TeX("with no clipping, $\\sigma_{DP}=0.8$ or $\\sigma_{DP}=50$")) +
  xlab(TeX("Loss on Minority ($L_0$)")) +
  ylab(TeX("Loss on Majority ($L_1$)")) +
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(4)),
        plot.subtitle=element_text(hjust=0.5, size=rel(2)),
        axis.title = element_text(size=rel(2)),
        axis.text = element_text(size=rel(1.5)),
        strip.text = element_text(size=rel(3)), # size of the facet label text
        legend.position = "bottom") +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")

rho_alpha_mmnist_plot <- mmnist_results %>%
  dplyr::filter(dp_str == "DPTRUES1z0.8sigmaNA" | dp_str == "DPFALSESNAzNAsigmaNA") %>%
  select(c("alpha", "dp", "L_0", "L_1")) %>%
  tidyr::pivot_wider(id_cols=c(alpha), values_from = c(L_0, L_1), names_from=dp) %>%
  dplyr::mutate(rho_alpha = (L_0_TRUE - L_1_TRUE)/(L_0_FALSE - L_1_FALSE)) %>%
  dplyr::mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  ggplot(aes(x=as.numeric(alpha), y = rho_alpha)) +
  geom_line(group=1, col="dodgerblue") +
  geom_point() +
  geom_vline(xintercept = 0.5, col="orange", lty="dashed") +
  # ylim(0, 30) +
  ylab(TeX("$\\rho(\\alpha)$")) + 
  xlab(TeX("Majority Fraction  $\\alpha$")) + 
  ggtitle(TeX("Disparity $\\rho(\\alpha)$ on MMNIST, with clipping")) + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        axis.title = element_text(size=rel(1.5)),
        )
rho_alpha_mmnist_plot
ggsave("./rho_alpha_vs_alpha_mmnist.pdf", device="pdf", height=8, width=8)

mmnist_results %>%
  dplyr::filter(dp_str == "DPTRUESInfzNAsigma50" | dp_str == "DPFALSESNAzNAsigmaNA") %>%
  # Exclude 0.7 due to numerical issues right around crossing of L_0, L_1 lines;
  # and exclude 0.3 for symmetry.
  dplyr::filter(alpha !=0.3, alpha !=0.7) %>%
  select(c("alpha", "dp", "L_0", "L_1")) %>%
  tidyr::pivot_wider(id_cols=c(alpha), values_from = c(L_0, L_1), names_from=dp) %>%
  dplyr::mutate(rho_alpha = (L_0_TRUE - L_1_TRUE)/(L_0_FALSE - L_1_FALSE)) %>%
  dplyr::mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  ggplot(aes(x=as.numeric(alpha), y = rho_alpha)) +
  geom_line(group=1, col="dodgerblue") +
  geom_point() +
  geom_vline(xintercept = 0.5, col="orange", lty="dashed") +
  # ylim(0, 30) +
  ylab(TeX("$\\rho(\\alpha)$")) + 
  xlab(TeX("Majority Fraction  $\\alpha$")) + 
  ggtitle(TeX("Disparity $\\rho(\\alpha)$ on MMNIST, no clipping, $\\sigma_{DP} = 50$")) + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        axis.title = element_text(size=rel(1.5)),
  ) + 
  expand_limits(y=0)

mmnist_phi_results = read_data_2("./data/mnist-grouped", "attr") 

mmnist_phi_alpha_plot <- mmnist_phi_results %>%
  dplyr::select(c("alpha", "phi8_01", "phi8_10", "phimax8")) %>%
  ggplot(aes(x=alpha, y=phimax8)) + 
  geom_point() + 
  geom_line(aes(group=1), col="dodgerblue") + 
  geom_hline(yintercept = 1, col = "orange", linetype="dashed") +
  theme_bw() +
  ylab(TeX("$\\phi(\\alpha)$")) +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  # facet_grid(. ~ metric) +
  ggtitle(TeX("Disparity $\\phi(\\alpha)$ on MMNIST")) +
  theme(plot.title=element_text(hjust=0.5))
mmnist_phi_alpha_plot
ggsave("./phi_alpha_mmnist.pdf", height=5, width=10, device="pdf")
mmnist_phi_alpha_plot + annotation_logticks() + scale_y_log10(breaks = log_breaks())
ggsave("./phi_alpha_mmnist_logscale.pdf", height=5, width=10, device="pdf")



mc10_results <- read_data("./data/cifar10-grouped/0-1-vs-9-2", "attr")

mc10_results %>%
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
        legend.position = "bottom") +
  labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") +
  scale_color_discrete(name = "Differential Privacy", labels = c("No DP", "DP-SGD S = 1, z = 0.8")) +
  facet_wrap(. ~ dp_str, scales="free")
ggsave("./L0_vs_L1_cifar10_grouped.pdf", device="pdf", height=8, width=16)


rho_alpha_mc10_plot <- mc10_results %>%
  dplyr::filter(dp_str == "DP-SGD, S = 1, z = 0.8" | dp_str == "No DP") %>%
  select(c("alpha", "dp", "L_0", "L_1")) %>%
  tidyr::pivot_wider(id_cols=c(alpha), values_from = c(L_0, L_1), names_from=dp) %>%
  dplyr::mutate(rho_alpha = (L_0_TRUE - L_1_TRUE)/(L_0_FALSE - L_1_FALSE)) %>%
  dplyr::mutate(alpha_str = paste0("$\\alpha = ", alpha, "$")) %>%
  ggplot(aes(x=as.numeric(alpha), y = rho_alpha)) +
  geom_line(group=1, col="dodgerblue") +
  geom_point() +
  geom_vline(xintercept = 0.5, col="orange", lty="dashed") +
  ylim(0, 5) +
  ylab(TeX("$\\rho(\\alpha)$")) + 
  xlab(TeX("Majority Fraction  $\\alpha$")) + 
  ggtitle(TeX("Disparity $\\rho(\\alpha)$ on MC10")) + 
  theme_bw() +
  # labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        axis.title = element_text(size=rel(1.5)),
        # axis.text = element_text(size=rel(1.5))
        )
  
rho_alpha_mc10_plot
ggsave("./rho_alpha_vs_alpha_mc10.pdf", device="pdf", height=8, width=8)

ggarrange(rho_alpha_mmnist_plot, rho_alpha_mc10_plot, nrow=2, ncol=1)
ggsave("rho_alpha_mmnist_mc10.pdf", width=4, height=8)

mc10_phi_results = read_data_2("data/cifar10-grouped/0-1-vs-9-2", "attr")

mc10_phi_alpha_plot <- mc10_phi_results %>%
  dplyr::select(c("alpha", "phi8_01", "phi8_10", "phimax8")) %>%
  ggplot(aes(x=alpha, y=phimax8)) + 
  geom_point() + 
  geom_line(aes(group=1), col="dodgerblue") + 
  geom_hline(yintercept = 1, col = "orange", linetype="dashed") +
  theme_bw() +
  ylab(TeX("$\\phi(\\alpha)$")) +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  # facet_grid(. ~ metric) +
  ggtitle(TeX("Disparity $\\phi(\\alpha)$ on MC10")) +
  theme(plot.title=element_text(hjust=0.5))
mc10_phi_alpha_plot
ggsave("./phi_alpha_mc10.pdf", height=5, width=10, device="pdf")
mmnist_phi_alpha_plot + annotation_logticks() + scale_y_log10(breaks = log_breaks())
ggsave("./phi_alpha_mc10_logscale.pdf", height=5, width=10, device="pdf")


