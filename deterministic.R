
mu <- function(n){
  (n^4 - n^2 - 2) / 4
}

sigma2 <- function(n){
  (2*n^6 - 9*n^4 + 7*n^2 + 6) / 72
}

mean = mu(10)
variance = sigma2(10)
sd = sqrt(variance)

print(mean)
print(variance)
print(pnorm(1200, mean=mean, sd=sd))

x <- seq(0, 10, 0.01)
y1 <- 2 * cos(x) + 8
y2 <- 3 * sin(x) + 4

plot(c(), c(), xlim=c(0,10), ylim=c(0,12))
polygon(c(x, rev(x)), c(y2, rev(y1)),
        col = "#ffc0c0", border = NA)

