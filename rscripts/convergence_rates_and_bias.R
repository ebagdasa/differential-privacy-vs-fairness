setwd("~/Documents/github/characterizing-rd-of-dp/rscripts")
source("utils.R")

test_class_levels = c("0" = TeX("Loss on Minority (L_0)"), 
                      "1" = TeX("Loss on Majority (L_1)"))
# dp_str_levels = c("No DP" = TeX("No DP"),
#                   "DP-SGD, S = 1, z = 0.8"= TeX("DP-SGD, $\\beta = 1,  \\sigma_{DP}^2 = 0.8$"),
#                   "DP-SGD, S = 1, z = 1" = TeX("DP-SGD, $\\beta = 1,  \\sigma_{DP}^2 = 1.0$"))

mnist_convergence <- read_convergence_data("./data/mnist-grouped", "attr")
mnist_convergence$test_class = factor(mnist_convergence$test_class)
levels(mnist_convergence$test_class) <- test_class_levels
# levels(mnist_convergence$dp_str) <- dp_str_levels

convergence_plot_theme <- theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
                                strip.text = element_text(size=rel(0.9)), # size of the facet label text
                                panel.grid.major = element_blank(), 
                                panel.grid.minor = element_blank(),
                                axis.text.y = element_text(size=rel(0.6)),
                                axis.text.x = element_text(size=rel(0.55)),
                                axis.title.y = element_blank(), # Suppress y-axis; use shared title with p2
                                legend.position = "bottom")

loss_plot_theme <- theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
                         strip.text = element_text(size=rel(0.9)), # size of the facet label text
                         panel.grid.major = element_blank(), 
                         panel.grid.minor = element_blank(),
                         axis.text.y = element_text(size=rel(0.6)),
                         axis.text.x = element_text(size=rel(0.55)),
                         legend.position = "bottom")
loss_plot_title <- ggtitle(TeX("Loss $L_j(\\hat{w})$ vs. Majority Fraction $\\alpha$"))
alpha_label_name = unname(TeX("Majority Fraction $\\alpha$"))

mnist_convergence %>%
  # dplyr::filter(dp_str=="DPFALSESNAzNAsigmaNA" | dp_str=="DPTRUESInfzNAsigma50") %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=as.numeric(alpha))) +
  facet_grid(rows = vars(factor(test_class)),
             cols = vars(dp_str),
             # labeller=label_parsed
             ) +
  theme_bw() +
  xlab("Epoch") +
  ylab("Loss") +
  ggtitle(TeX("Convergence Behavior vs. Majority Fraction $\\alpha$")) +
  scale_color_continuous(name = unname(TeX("Majority Fraction $\\alpha$"))) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)")
ggsave("./convergence_behavior_mnist_grouped.pdf", device="pdf", height=7, width=12)  

p1 <- mnist_convergence %>%
  dplyr::filter(dp_str=="DPFALSESNAzNAsigmaNA" | dp_str=="DPTRUES1z0.8sigmaNA") %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=as.numeric(alpha))) +
  facet_grid(rows = vars(factor(test_class)),
             cols = vars(dp_str),
             # labeller=label_parsed
             ) +
  theme_bw() +
  xlab("Epoch") +
  ylab("Subgroup Loss") +
  ggtitle(TeX("Convergence of $L_j(\\hat{w}^t)$ on MMNIST"),
          subtitle = "With Clipping") +
  scale_color_continuous(name = alpha_label_name) +
  convergence_plot_theme
p1

mnist_convergence %>%
  dplyr::filter(dp_str=="DPFALSESNAzNAsigmaNA" | 
                  dp_str=="DPTRUESInfzNAsigma0.8" |
                  dp_str=="DPTRUESInfzNAsigma50") %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=as.numeric(alpha))) +
  facet_grid(rows = vars(factor(test_class)),
             cols = vars(dp_str),
             # labeller=label_parsed
  ) +
  theme_bw() +
  xlab("Epoch") +
  ylab("Subgroup Loss") +
  ggtitle(TeX("Convergence of $L_j(\\hat{w}^t)$ on MMNIST"),
          subtitle = "With Clipping") +
  scale_color_continuous(name = alpha_label_name) +
  convergence_plot_theme

########################
mmnist_maxstep = max(mnist_convergence$Step)
mmnist_subgroup_loss_plot <- mnist_convergence %>%
  dplyr::filter(Step==mmnist_maxstep, 
                dp_str=="DPFALSESNAzNAsigmaNA" | dp_str=="DPTRUES1z0.8sigmaNA") %>%
  ggplot(aes(x=alpha, y=Value, col=test_class)) + 
  geom_line(aes(group=test_class)) +
  geom_point() +
  facet_grid(
    cols = vars(dp_str),
    # labeller=label_parsed
    ) +
  theme_bw() +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab(TeX("Subgroup Loss $L_j$")) + 
  scale_color_grey(name="Subgroup Loss",
                     labels=c("Loss on Minority", "Loss on Majority")) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  ggtitle("Subgroup Losses on MMNIST")
mmnist_subgroup_loss_plot

mnist_convergence %>%
  dplyr::filter(Step==mmnist_maxstep, 
                dp_str=="DPFALSESNAzNAsigmaNA" | 
                  dp_str=="DPTRUESInfzNAsigma0.8" |
                  dp_str=="DPTRUESInfzNAsigma50") %>%
  ggplot(aes(x=alpha, y=Value, col=test_class)) + 
  geom_line(aes(group=test_class)) +
  geom_point() +
  facet_wrap(. ~ dp_str,
    # labeller=label_parsed
    scales = "free"
  ) +
  theme_bw() +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab(TeX("Subgroup Loss $L_j$")) + 
  scale_color_grey(name="Subgroup Loss",
                   labels=c("Loss on Minority", "Loss on Majority")) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  ggtitle("Subgroup Losses on MMNIST", subtitle="No clipping")
#######################

mc10_convergence <- read_convergence_data("./data/cifar10-grouped/0-1-vs-9-2", "attr")
mc10_convergence$test_class = factor(mc10_convergence$test_class)
levels(mc10_convergence$test_class) <- test_class_levels
levels(mc10_convergence$dp_str) <- dp_str_levels

mc10_labs <- labs(
  caption=
    "Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)"
  )

mc10_convergence_plot <- mc10_convergence %>%
  dplyr::filter(!is.na(test_class)) %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=alpha)) +
  facet_grid(rows = vars(factor(test_class)),
             cols = vars(dp_str),
             labeller=label_parsed) +
  theme_bw() +
  xlab("Epoch") +
  ylab("Loss") +
  scale_color_continuous(name = alpha_label_name) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom")
mc10_convergence_plot + ggtitle(TeX("Convergence Behavior vs. Majority Fraction $\\alpha$")) + mc10_labs
ggsave("./convergence_behavior_cifar10_grouped.pdf", device="pdf", height=7, width=8)  


###############
mc10_subgroup_loss_plot <- mc10_convergence %>%
  dplyr::filter(!is.na(test_class)) %>%
  filter(Step==119) %>%
  ggplot(aes(x=alpha, y=Value, col=test_class)) + 
  geom_line() +
  geom_point() +
  facet_grid(
             cols = vars(dp_str),
             labeller=label_parsed) +
  theme_bw() +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab(TeX("Subgroup Loss $L_0$")) + 
  scale_color_grey(name="Subgroup Loss",
                     labels=c("Loss on Minority", "Loss on Majority")) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  ggtitle("Subgroup Losses on MC10")
mc10_subgroup_loss_plot
###############
  
p3 <- mc10_convergence_plot + ggtitle(TeX("Convergence of $L_j(\\hat{w}^t)$ on MC10")) + convergence_plot_theme


ggarrange(p1, p3, nrow=1, ncol=2, common.legend = TRUE, legend="bottom")
ggsave("./convergence_mmnist_mc10.pdf", device="pdf", width=10, height=5)

ggarrange(mmnist_subgroup_loss_plot, mc10_subgroup_loss_plot, nrow=2, ncol=1, common.legend = TRUE, legend="bottom")
ggsave("./subgroup_losses_mmnist_mc10.pdf", device="pdf", width=8, height=8)
