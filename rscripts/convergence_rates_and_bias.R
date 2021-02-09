setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")

mnist_convergence <- read_convergence_data("./data/mnist-grouped", "attr")
mnist_convergence$test_class = factor(mnist_convergence$test_class)
levels(mnist_convergence$test_class) <- c("0"="Loss on Minority (L_0)", "1"="Loss on Majority (L_1)")
mnist_convergence %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=alpha)) +
  facet_grid(factor(test_class) ~ dp_str) +
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
  dplyr::filter(dp_str != "DP-SGD, S = 1, z = 1") %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=alpha)) +
  facet_grid(factor(test_class) ~ dp_str) +
  theme_bw() +
  xlab("Epoch") +
  ylab("Subgroup Loss") +
  ggtitle(TeX("Convergence $L_j(\\hat{w}^t)$")) +
  scale_color_continuous(name = unname(TeX("Majority Fraction $\\alpha$"))) +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        strip.text = element_text(size=rel(0.9)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(size=rel(0.6)),
        axis.text.x = element_text(size=rel(0.55)),
        axis.title.y = element_blank(), # Suppress y-axis; use shared title with p2
        legend.position = "bottom") +
p1

mnist_convergence %>%
  filter(Step==1) %>%
  ggplot(aes(x=factor(alpha), y=Value)) + 
  geom_col(aes(fill=alpha)) + 
  facet_grid(factor(test_class) ~ dp_str) +
  theme_bw() +
  scale_color_continuous(name = unname(TeX("Majority Fraction $\\alpha$"))) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  labs(caption="Task: MNIST (1, 3) vs. (7, 8) digit classification\nMinority Group (1, 8)") +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab("Subgroup Loss") + 
  ggtitle(TeX("Bias $L_j(\\hat{w}_0)$ vs. Majority Fraction $\\alpha$"))
ggsave("./bias_mnist_grouped.pdf", device="pdf", height=7, width=12) 

p2 <- mnist_convergence %>%
  dplyr::filter(Step==1, dp_str != "DP-SGD, S = 1, z = 1") %>%
  ggplot(aes(x=factor(alpha), y=Value)) + 
  geom_col(aes(fill=alpha)) + 
  facet_grid(factor(test_class) ~ dp_str) +
  theme_bw() +
  scale_color_continuous(name = unname(TeX("Majority Fraction $\\alpha$"))) +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        strip.text = element_text(size=rel(0.9)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        axis.text.y = element_text(size=rel(0.6)),
        axis.text.x = element_text(size=rel(0.55)),
        legend.position = "bottom") +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab("Subgroup Loss") + 
  ggtitle(TeX("Bias $L_j(\\hat{w}^0)$")) +
  scale_fill_continuous(name = unname(TeX("Majority Fraction $\\alpha$")))

ggarrange(p2, p1, nrow=1, ncol=2, common.legend = TRUE, legend="bottom")
ggsave("./convergence_bias_mnist_subset.pdf", device="pdf", width=8, height=5)

mc10_convergence <- read_convergence_data("./data/cifar10-grouped/0-1-vs-9-2", "attr")
mc10_convergence$test_class = factor(mc10_convergence$test_class)
levels(mc10_convergence$test_class) <- c("0"="Loss on Minority (L_0)", "1"="Loss on Majority (L_1)")
mc10_convergence %>%
  dplyr::filter(!is.na(test_class)) %>%
  ggplot(aes(x=Step, y=Value, group=factor(alpha))) +
  geom_line(aes(col=alpha)) +
  facet_grid(factor(test_class) ~ dp_str) +
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
  labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)")
ggsave("./convergence_behavior_cifar10_grouped.pdf", device="pdf", height=7, width=8)  

mc10_convergence %>%
  dplyr::filter(!is.na(test_class)) %>%
  filter(Step==1) %>%
  ggplot(aes(x=factor(alpha), y=Value)) + 
  geom_col(aes(fill=alpha)) + 
  facet_grid(factor(test_class) ~ dp_str) +
  theme_bw() +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ylab("Subgroup Loss") + 
  scale_color_continuous(name = unname(TeX("Majority Fraction $\\alpha$"))) +
  theme(plot.title=element_text(hjust=0.5, size=rel(2)),
        strip.text = element_text(size=rel(1.25)), # size of the facet label text
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = "bottom") +
  labs(caption="Task: CIFAR10 (airplane, automobile) vs. (bird, truck) classification\nMinority Group (automobile, bird)") +
  ggtitle(TeX("Bias $L_j(\\hat{w}_0)$ vs. Majority Fraction $\\alpha$"))
ggsave("./bias_cifar10_grouped.pdf", device="pdf", height=7, width=8) 
  
