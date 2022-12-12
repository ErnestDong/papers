set_dynare_seed(11201+11203+11204);
var y,c,c1,c2,i,k,l,l1,l2,r,w,a;
varexo e;
parameters alpha, beta, b, lambda, delta, rho, sigma;

//parameter values
alpha = 0.35;
beta = 0.97;
b = 1.5;
lambda = 0.5;
delta = 0.06;
rho = 0.95;
sigma = 0.01;

//model specifications
model;
    y = lambda*i + c;
    c = lambda*c1 + (1-lambda)*c2;
    y = a*(lambda*k)^alpha * l^(1-alpha);
    l = l1+l2;
    k = i + (1-delta)*k(-1);
    r = alpha*y/k + 1-delta;
    w = (1-alpha)*y/l;
    l1 = 1 - b/w*c1;
    l2 = 1 - b/w*c2;
    c1 = 1/beta*c1(+1)/r(+1);
    l1 = 1/(b*c1/y/(1-alpha) + 1);
    a = a(-1)^rho*exp(e);
end;

shocks;
    var e;
    stderr sigma;
end;

// initial vars
initval;
y = 0.5;
c = 0.5;
c1 = 0.5;
c2 = 0.5;
i = 0.1;
k = 2;
l = 0.4;
l1 = 0.4;
l2 = 0;
r = 1;
w = 1;
a = 1;
end;

steady;

stoch_simul(periods=200, order=1, irf=50) y,c,i,k,l,r,w,a,c1,c2;
rplot y;
rplot i;
rplot c;
rplot c1;
rplot c2;
