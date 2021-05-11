setwd("~/Documents/github/characterizing-rd-of-dp/rscripts")
source("utils.R")
library(ggallin)  # used for pseudolog10_trans

rho_alpha_results <- read.csv("/Users/jpgard/Documents/github/characterizing-rd-of-dp/rscripts/data/lr-rho-alpha/lr_rho_alpha_fullgrid.csv")
rho_lr_alpha_results <- read.csv("/Users/jpgard/Documents/github/characterizing-rd-of-dp/rscripts/data/lr-rho-alpha/rho_lr_values.csv")

plot_theme <-  theme(plot.title = element_text(hjust=0.5, size=rel(1.1)),
                     axis.text = element_text(size=rel(0.5)),
                     )
rho_alpha_results %>%
  dplyr::mutate(neg_rho_alpha = ifelse(rho_alpha < 0, 1, 0)) %>%
  data_summary(varname="neg_rho_alpha", 
               groupnames=c("alpha")) %>%
  ggplot(aes(x=factor(alpha), y = neg_rho_alpha)) + 
  geom_line(aes(group=1)) + 
  ggtitle("Fraction of Trials with Reverse Disparity")

p1 <- rho_alpha_results %>%
  data_summary(varname="rho_alpha", 
               groupnames=c("alpha")) %>%
  ggplot(aes(x=factor(alpha), y=rho_alpha)) +
  geom_point(col="dodgerblue", size=rel(1.5)) +
  geom_errorbar(aes(ymin=rho_alpha-sd, ymax=rho_alpha+sd), width=.25, size=0.5) +
  theme_bw() +
  plot_theme +
  ylab(TeX("$\\rho(\\alpha)")) +
  xlab(TeX("Majority Fraction $\\alpha$")) +
  ggtitle(TeX("Disparity Measure $\\rho(\\alpha)$ in Linear Regression")) +
  scale_y_continuous(trans=pseudolog10_trans, breaks = c(-4000, -1000, -100, -10, 0, 10), minor_breaks = seq(-4000, 100, 100))
p1
ggsave("rho_alpha_fullgrid.pdf", width=5, height=5)

neg_rho_alpha_values = dplyr::filter(rho_lr_alpha_results, rho_LR < 0) %>% dplyr::pull(alpha)
alphamin = min(neg_rho_alpha_values)
alphamax = max(neg_rho_alpha_values)
rect = data.frame(amin=alphamin, amax=alphamax)

lb = -10^7
ub = 10^7

p2 <- rho_lr_alpha_results %>%
  # dplyr::mutate(rev_disp = as.numeric(rho_LR < 0)) %>%
  dplyr::filter(alpha != 0) %>%
  ggplot(aes(x=alpha, y=rho_LR)) + 
  geom_line(col="dodgerblue", size=1.25) +
  scale_y_log10() +
  theme_bw() +
  ylim(lb, ub) +
  geom_hline(yintercept = 0, col="black", lty = "dashed") +
  geom_rect(data=rect, aes(NULL, NULL, xmin=alphamin,xmax=alphamax), ymin=lb, ymax=ub, fill="red", alpha=0.2) +
  ggtitle(TeX("Disparity Measure $\\rho_{LR}$ vs. $\\alpha$")) +
  plot_theme +
  ylab(TeX("Exact Value of $\\rho_{LR}$")) +
  xlab(TeX("Majority Fraction $\\alpha$"))
p2
ggsave("rho_lr_alpha.pdf", width=5, height=5)

g = arrangeGrob(p1, p2, nrow=1)
g
ggsave("rho_alpha_plots_fullgrid.pdf", g, width=8, height=3)

