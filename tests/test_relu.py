from isdhic.logistic import log_prob as logistic
from isdhic.relu import log_prob as relu

x = np.linspace(-10.,10.,10000)
a = 10.

l_logistic = [logistic(y.reshape(1,)*0., y.reshape(1,), a) for y in x]
l_relu = [relu(y.reshape(1,)*0., y.reshape(1,), a) for y in x]

clf()
plot(x, l_logistic)
plot(x, l_relu)
raise
plot(x, 1/(1+np.exp(a*x)))
plot(x, np.exp(np.array(lp)))
plot(x, np.exp(-np.clip(x,0,1e308)))
