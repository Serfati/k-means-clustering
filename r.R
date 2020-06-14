# Assigment 2 - R
# --------------------------------------------------------
library(caTools)
library(classInt)
library(DescTools)
library(rattle)
library(RColorBrewer)
library(rpart)

# --------------------------------------------------------
# Auxiliary functions
# ------------------
discretize_category <-
  function(df_, category_, style_, numOfBins_) {
    i <- 0
    for (x in df_[[category_]]) {
      i <- i + 1
      if (is.na(x) == TRUE) {
        next
      }
      else if (x < create_bin(df__ = df_, category__ = category_, numOfBins__ = numOfBins_, style__ = style_, bin = 2)) {
        df_[[i, category_]] <- 1
      }
      else if (x >= create_bin(df__ = df_, category__ = category_, numOfBins__ = numOfBins_, style__ = style_, bin = numOfBins_)) {
        df_[[i, category_]] <- (numOfBins_)
      }
      else {
        for (j in 2:(numOfBins_ - 1)) {
          if (x >= create_bin(df__ = df_, category__ = category_, numOfBins__ = numOfBins_, style__ = style_, bin = j)
            & x < create_bin(df__ = df_, category__ = category_, numOfBins__ = numOfBins_, style__ = style_, bin = (j + 1))) {
            df_[[i, category_]] <- j
          }
        }
      }
    }
    return(df_)
  }

create_bin <- function(df__, category__, numOfBins__, style__, bin) {
  bins <- classIntervals(df__[[category__]], n = numOfBins__, style = style__)
  return(bins$brks[bin])
}
# --------------------------------------------------------
# 1. DATA PREPERATION
# ---------------------

# 1.1 Read CSV into R, print each column type, replace empty strings with NA
wd <- getwd()
csv_path <- paste0(wd, "/Ex-2_dataset.csv")
df <- read.csv(file = csv_path, header = TRUE, sep = ",", na.strings = c("", "NA"))

#print("Original data-frame types:")
#print("--------------------------")
#str(df)

# 1.3 Completing missing values (numeric: average, categorial: most common)
complete_missing_values <- function(df) {
  for (j in seq_len(ncol(df))) {
    for (i in seq_len(nrow(df))) {
      if (is.numeric(df[i, j]) &
        is.na(df[i, j]))
        df[i, j] <- mean(df[, j], na.rm = TRUE)
      if (is.factor(df[i, j]) &
        is.na(df[i, j]))
        df[i, j] <- Mode(df[, j], na.rm = TRUE)
    }
  }
  #print("data-frame types **after** completing missing values:")
  #print("-----------------------------------------------------")
  #str(df)
  return(df)
}
df <- complete_missing_values(df)

# Checking before discretization:
#hist(df$Monthly_Profit)
#print(mean(df$Monthly_Profit))
#print(sd(df$Monthly_Profit))
#hist(df$Spouse_Income)
#print(mean(df$Spouse_Income))
#print(sd(df$Spouse_Income))
#hist(df$Loan_Amount)
#print(mean(df$Loan_Amount))
#print(sd(df$Loan_Amount))

# 1.4 Discretization of Monthly_Profit, Spouse_Income, Loan_Amount
categories <- c("Monthly_Profit", "Spouse_Income", "Loan_Amount")
styles <- c("quantile", "quantile", "quantile") # Equal-frequency discretization
bins <- c(4, 3, 3) # Monthly_Profit: 4 bins, Spouse_Income: 3 bins, Loan_Amount: 3 bins
for (i in seq_along(categories)) {
  df <- discretize_category(df_ = df, category_ = categories[i], style_ = styles[i], numOfBins_ = bins[i])
  df[[categories[i]]] <- as.factor(df[[categories[i]]])
}

#print("data-frame types after discretization:")
#print("--------------------------------------")
#str(df)

# 1.2 Randomly split data-frame into: test-set (30%), training-set (70%)
trn_index <- sample.split(seq_len(nrow(df)), 0.70)
trn_set <- df[trn_index,]
tst_set <- df[!trn_index,]

# target class:
tst_set$Request_Approved <- NULL

# --------------------------------------------------------
# 2. Decision-Tree Model
# -----------------------
class(fo <- Request_Approved ~ Employees +
  Monthly_Profit +
  Credit_History +
  Customers +
  Export_Abroad +
  Loan_Amount +
  Payment_Terms +
  Gender +
  Education +
  Married +
  Spouse_Income)

method_ <- "class"
splits <- c("gini", "information")
minsplits <- c(12, 31)

for (i in seq_along(splits))
  for (j in seq_along(minsplits)) {
    fname_ <- paste0("dt_", splits[i], "_", minsplits[j])
    dt <- fname_
    assign(dt, rpart(formula = fo, data = trn_set, method = method_,
                     parms = list(split = splits[i]), minsplit = minsplits[j]))
    save_plot <- function(dir, fname, model, split, min_split) {
      dir.create(dir, showWarnings = FALSE)
      full_path <- paste0(dir, fname)
      jpeg(filename = paste0(full_path, ".jpg"))
      fancyRpartPlot(model, caption = paste0(split, " with minsplit=", min_split))
      dev.off()
    }
    save_plot(dir = paste0(wd, "/Plots/"), fname = fname_,
              model = get(dt), split = splits[i], min_split = minsplits[j])

    # 3. Model Performance Evaluation
    # -------------------------------
    check_accuracy <-
      function(chosen_model = get(dt), tst_set_ = tst_set, df_ = df, split = splits[i], min_split = minsplits[j]) {
        tst_set_$Request_Approved_Class <-
          predict(chosen_model, newdata = tst_set_, type = "class")
        tst_set_$Request_Approved_Prob <-
          predict(chosen_model, newdata = tst_set_, type = "prob")
        sum <- 0
        for (i in seq_len(nrow(tst_set_))) {
          tst_set_loan_status <- tst_set_[i, "Request_Approved_Class"]
          tst_set_id <- tst_set_[i, "Request_Number"]
          df_loan_status <- df_[df_$Request_Number == tst_set_id, "Request_Approved"]
          if (tst_set_loan_status == df_loan_status)
            sum <- sum + 1
        }
        accuracy <- ((sum / (nrow(tst_set))) * 100)
        cat("The accuracy of the", split, "model with minsplit=", min_split, "is:\n")
        cat(accuracy, "%\n")
        cat("------------------------------------------------------------\n")
        Sys.sleep(3)
      }
    check_accuracy()
  }