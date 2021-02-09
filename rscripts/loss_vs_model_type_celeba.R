setwd("~/Documents/github/differential-privacy-vs-fairness/rscripts")
source("utils.R")

tmp = list()
for (f in list.files("./data/celeba-finetune", full.names = TRUE)){
  df = read.csv(f)
  df = dplyr::filter(df, Step == 59)
  # df$fname = f
  df$uid = stringr::str_extract(f, "celeba-S.*sigma[a-zA-Z0-9]+")
  df$tag = sub("tag-", "", stringr::str_extract(f, "tag-[a-zA-Z0-9_]+"))
  df$train_subset = sub("tag-", "", stringr::str_extract(f, "trattrsub[0-9]"))
  tmp[[f]] = df
}
results = dplyr::bind_rows(tmp) %>%
  replace_na(list("train_subset" = "None")) %>%
  dplyr::select(c("uid", "train_subset", "Value", "tag")) %>% 
  tidyr::pivot_wider(id_cols=c(uid, train_subset), 
                     names_from=tag, values_from=Value) 

results$test_crossentropy_loss = results$test_crossentropy_loss * 100
results = tidyr::pivot_longer(results, cols=starts_with("test"), names_to="metric")

results$S = sub("S", "", stringr::str_extract(results$uid, "S[[a-zA-Z]|0-9]+"))
results$z = sub("z", "", stringr::str_extract(results$uid, "z[[a-zA-Z]|0-9\\.]+"))
results$train_subset = forcats::fct_recode(
  results$train_subset, "Union" = "None",
                    "Majority" = "trattrsub0",
                    "Minority"= "trattrsub1")
results$metric = forcats::fct_recode(results$metric, 
                    "Union" = "test_crossentropy_loss",
                    "Majority" = "test_loss_per_attr_0",
                    "Minority" = "test_loss_per_attr_1")

results$uid = factor(results$uid, 
       levels=c(
         "celeba-SNone-zNone-sigmaNone", # Makes non-DP first in ordered factor levels
         "celeba-S1-z0.8-sigmaNone",
         "celeba-S1-z1.0-sigmaNone"))
results$uid = forcats::fct_recode(results$uid,
                                  "No DP" = "celeba-SNone-zNone-sigmaNone",
                                  "S = 1, z = 0.8" = "celeba-S1-z0.8-sigmaNone",
                                  "S = 1, z = 1.0" = "celeba-S1-z1.0-sigmaNone"
)

results %>%
ggplot(aes(x=train_subset, y = value, fill = metric)) + 
  geom_bar(stat="identity", position="dodge", colour="black") +
  facet_wrap(uid ~ .) +
  labs(fill = "Test Subset",
       caption = TeX("Celeba Dataset; Minority/Majority Attribute Label = Blond Hair $(\\alpha = 0.852)$")) +
  ylab("Loss") + 
  xlab("Training Subset") + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5, size=rel(1.5)),
        axis.text = element_text(size=rel(1.1)),
        axis.title = element_text(size=rel(2)),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.text = element_text(size=rel(1.5)),
        legend.title = element_text(size=rel(1.5)),
        strip.text = element_text(size=rel(1.5)), # size of the facet label text
        legend.position = 'bottom'
        ) +
  ggtitle("Subgroup Loss vs. Model Type and Training Subset") +
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"))
ggsave("celeba_subset_loss_finetune.pdf", width=8, height=6, device="pdf")


ggplot(results, aes(x=uid, y = value, fill = metric)) + 
  geom_bar(stat="identity", position="dodge", colour="black") +
  facet_wrap(train_subset ~ .) +
  labs(fill = "Test Subset",
       caption = TeX("Celeba Dataset; Minority/Majority Attribute Label = Blond Hair $(\\alpha = 0.852)$")) +
  ylab("Loss") + 
  xlab("Training Subset") + 
  theme_bw() +
  theme(plot.title=element_text(hjust=0.5),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank(),
        legend.position = 'bottom'
  ) +
  ggtitle("Subgroup Loss vs. Model Type and Training Subset") +
  scale_fill_manual(values=c("#0072B2", "#E69F00", "#56B4E9"))
