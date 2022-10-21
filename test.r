
x = 1:100

layout(matrix(1:2, ncol = 1), widths = 1, heights = c(1.5,1.5), respect = FALSE)
par(mar = c(0, 4.1, 4.1, 2.1))
plot(x, x, type = 'l', xaxt = 'n', main = 'My Great Graph')
par(mar = c(4.1, 4.1, 0, 2.1))
plot(x, x^2, type = 'l')