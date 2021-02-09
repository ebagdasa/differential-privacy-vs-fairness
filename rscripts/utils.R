library(ggplot2)
library(dplyr)
library(tidyr)
library(latex2exp)
library(magrittr)
library(gridExtra)
library(glue)
library(scales)
library(ggpubr)

#+++++++++++++++++++++++++
# Function to calculate the mean and the standard deviation
# for each group
#+++++++++++++++++++++++++
# data : a data frame
# varname : the name of a column containing the variable
#to be summariezed
# groupnames : vector of column names to be used as
# grouping variables
data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  data_sum <- rename(data_sum, c("mean" = varname))
  return(data_sum)
}


base_breaks <- function(n = 10){
  function(x) {
    axisTicks(log10(range(x, na.rm = TRUE)), log = TRUE, n = n)
  }
}


# Read the TensorBoard results CSV files, keeping only the final (59th iteration)
# dirname: the directory to read from.
# loss_group: the group defined by a valid tag, e.g.: tag-test_loss_per_{loss_group}_
read_data <- function(dirname, loss_group){
  tmp = list()
  for (f in list.files(dirname, full.names = TRUE)){
    df = read.csv(f)
    df$uid = f
    
    # Casting of string "None" to numeric will produce "Warning message: NAs introduced by coercion"
    # which can be safely ignored
    df$S = as.numeric(sub("-S", "", stringr::str_extract(f, "-S[a-zA-Z0-9\\.]+")))
    df$z = as.numeric(sub("-z", "", stringr::str_extract(f, "-z[a-zA-Z0-9\\.]+")))
    
    df$dp = as.logical(sub("-dp", "", stringr::str_extract(f, "-dp[a-zA-Z]+")))
    df$alpha = sub("alpha", "", stringr::str_extract(f, "alpha\\d\\.\\d+"))
    df$test_class = sub(glue("tag-test_loss_per_{loss_group}_"), "", 
                        stringr::str_extract(f, glue("tag-test_loss_per_{loss_group}_[01]")))
    tmp[[f]] = df
  }
  results = dplyr::bind_rows(tmp) %>% 
    dplyr::filter(Step == 59) %>%
    select(c("Value", "alpha", "dp", "S", "z", "test_class")) %>%
    tidyr::pivot_wider(id_cols=c(alpha, dp, S, z), 
                       names_from=test_class, values_from=Value) %>%
    dplyr::rename("L_0"="0", "L_1"="1")
  results$dp_str = factor(tidyr::replace_na(results$z, 0))
  levels(results$dp_str) <- c("No DP", "DP-SGD, S = 1, z = 0.8", "DP-SGD, S = 1, z = 1")
  return(results)
}

# Read the TensorBoard results CSV files, and create a column for phi.
# dirname: the directory to read from.
# loss_group: the group defined by a valid tag, e.g.: tag-test_loss_per_{loss_group}_
read_data_2 <- function(dirname, loss_group){
  tmp = list()
  for (f in list.files(dirname, full.names = TRUE)){
    df = read.csv(f)
    df$uid = f
    
    # Casting of string "None" to numeric will produce "Warning message: NAs introduced by coercion"
    # which can be safely ignored
    df$S = as.numeric(sub("-S", "", stringr::str_extract(f, "-S[a-zA-Z0-9\\.]+")))
    df$z = as.numeric(sub("-z", "", stringr::str_extract(f, "-z[a-zA-Z0-9\\.]+")))
    
    df$dp = as.logical(sub("-dp", "", stringr::str_extract(f, "-dp[a-zA-Z]+")))
    df$alpha = sub("alpha", "", stringr::str_extract(f, "alpha\\d\\.\\d+"))
    df$test_class = sub(glue("tag-test_loss_per_{loss_group}_"), "", 
                        stringr::str_extract(f, glue("tag-test_loss_per_{loss_group}_[01]")))
    tmp[[f]] = df
  }
  results = dplyr::bind_rows(tmp) 
  results$dp_str = factor(tidyr::replace_na(results$z, 0))
  levels(results$dp_str) <- c("NoDP", "DPSGDS1z0.8", "DPSGDS1z1")
  results <- results %>%
    dplyr::filter(Step == 59) %>%
    select(c("Value", "alpha", "test_class", "dp_str")) %>%
    tidyr::pivot_wider(id_cols=c(alpha, test_class),  names_from=dp_str, values_from=Value) %>%
    dplyr::mutate(dp_nodp_diff8=DPSGDS1z0.8-NoDP, dp_nodp_diff1=DPSGDS1z1-NoDP) %>%
    # Create a column for Loss(w_hat_DP) - Loss(w_hat_noDP)
    tidyr::pivot_wider(id_cols=c(alpha), names_from=test_class, values_from=c(dp_nodp_diff8, dp_nodp_diff1)) %>%
    # Create the column for phi
    dplyr::mutate(phi8 = dp_nodp_diff8_0/dp_nodp_diff8_1, phi1 = dp_nodp_diff1_0/dp_nodp_diff1_1)
  
  return(results)
}


# Read the TensorBoard results CSV files keeping all iterations.
# dirname: the directory to read from.
# loss_group: the group defined by a valid tag, e.g.: tag-test_loss_per_{loss_group}_
read_convergence_data <- function(dirname, loss_group){
  tmp = list()
  for (f in list.files(dirname, full.names = TRUE)){
    df = read.csv(f)
    df$uid = f
    
    # Casting of string "None" to numeric will produce "Warning message: NAs introduced by coercion"
    # which can be safely ignored
    df$S = as.numeric(sub("-S", "", stringr::str_extract(f, "-S[a-zA-Z0-9\\.]+")))
    df$z = as.numeric(sub("-z", "", stringr::str_extract(f, "-z[a-zA-Z0-9\\.]+")))
    
    df$dp = as.logical(sub("-dp", "", stringr::str_extract(f, "-dp[a-zA-Z]+")))
    df$alpha = as.numeric(sub("alpha", "", stringr::str_extract(f, "alpha\\d\\.\\d+")))
    df$test_class = sub(glue("tag-test_loss_per_{loss_group}_"), "", 
                        stringr::str_extract(f, glue("tag-test_loss_per_{loss_group}_[01]")))
    tmp[[f]] = df
  }
  results = dplyr::bind_rows(tmp) %>%
    select(c("Value", "alpha", "dp", "S", "z", "test_class", "Step"))
  results$dp_str = factor(tidyr::replace_na(results$z, 0))
  levels(results$dp_str) <- c("No DP", "DP-SGD, S = 1, z = 0.8", "DP-SGD, S = 1, z = 1")
  return(results)
}
