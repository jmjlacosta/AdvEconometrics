library(corrplot)
library(rJava)
library(xlsx)


cor.mtest <- function(mat, ...) {
  mat <- as.matrix(mat)
  n <- ncol(mat)
  p.mat<- matrix(NA, n, n)
  diag(p.mat) <- 0
  for (i in 1:(n - 1)) {
    for (j in (i + 1):n) {
      tmp <- cor.test(mat[, i], mat[, j], ...)
      p.mat[i, j] <- p.mat[j, i] <- tmp$p.value
    }
  }
  colnames(p.mat) <- rownames(p.mat) <- colnames(mat)
  p.mat
}

data <- read.csv('credit.csv')
bal <- data$Balance
df <- subset(data, select = -c(X, Gender, Student, Married, Ethnicity, Balance))
dfexp <- df**2
df$Gender <- (data$Gender=="Female")*1
df$Student <- (data$Student=="Yes")*1
df$Married <- (data$Married=="Yes")*1
df$Asian <- (data$Ethnicity=="Asian")*1
df$African <- (data$Ethnicity=="African American")*1

colnames(dfexp) <- c("Income*Income", "Limit*Limit", "Rating*Rating", "Cards*Cards",
                   "Age*Age", "Education*Education")

dfinter <- do.call(cbind, combn(colnames(df), 2, FUN=function(x)
  list(setNames(data.frame(df[,x[1]]*df[,x[2]]), paste(x, collapse='*')))))

df1 <- merge(dfinter, dfexp)

write.xlsx(df1, "interactions.xlsx")

M <- cor(df)
p.mat <- cor.mtest(df)
col <- colorRampPalette(c("#BB4444", "#EE9988", "#FFFFFF", "#77AADD", "#4477AA"))
corrplot(M, method="color", col=col(200),
         type="upper", order="hclust",
         addCoef.col = "black", # Add coefficient of correlation
         tl.col="black", tl.srt=45, #Text label color and rotation
         # Combine with significance
         p.mat = p.mat, sig.level = 0.05, insig = "blank",
         # hide correlation coefficient on the principal diagonal
         diag=FALSE
)


# Running a quick correlation matrix we can see that there is clearly some high correlation between Age, Income, Limit, and Rating.
# Therefore, LASSO could potentially be a beter model for these parameters as it would adjust for this correlation.

data <- read.csv('credit.csv')
summary(data)
summary(data.frame(data))
head(data)
