%data_file=argv(1)
LAMBDA=1.55
data_file='../data/synthetic.csv';
num_restarts=100000;
data=csvread(data_file);
[c1,c2,k1,k2]=sgd(data,num_restarts,LAMBDA);
c1
c2
k1
k2
sse_val=sse(c1,c2,k1,k2,data,LAMBDA)


function pred=predict(c1,c2,k1,k2,t,LAMBDA)
    pred=c1*(LAMBDA*exp(-k1*t)-k1*exp(-LAMBDA*t))/(LAMBDA-k1)+c2*(LAMBDA*exp(-k2*t)-k2*exp(-LAMBDA*t))/(LAMBDA-k2);
end

function sse_val=sse(c1,c2,k1,k2,data,LAMBDA)
  sse_val=0;
  for i=1:size(data,1)
    sse_val=sse_val+(predict(c1,c2,k1,k2,data(i,1),LAMBDA)-data(i,2))^2;
  end
end

function [c1,c2,k1,k2]=gradient(c1,c2,k1,k2,t,y,alpha,LAMBDA)
  p=predict(c1,c2,k1,k2,t,LAMBDA);
  c1_grad=(p-y)*(LAMBDA*exp(-k1*t)-k1*exp(-LAMBDA*t))/(LAMBDA-k1);
  c2_grad=(p-y)*(LAMBDA*exp(-k2*t)-k2*exp(-LAMBDA*t))/(LAMBDA-k2);
  k1_grad=(y-p)*c1*((LAMBDA*t*exp(-k1*t)+exp(-LAMBDA*t))/(LAMBDA-k1)+(k1*exp(-LAMBDA*t)-LAMBDA*exp(-k1*t))/(LAMBDA-k1)^2.0);
  k2_grad=(y-p)*c2*((LAMBDA*t*exp(-k2*t)+exp(-LAMBDA*t))/(LAMBDA-k2)+(k2*exp(-LAMBDA*t)-LAMBDA*exp(-k2*t))/(LAMBDA-k2)^2.0);
  c1=c1-c1_grad*alpha;
  c2=c2-c2_grad*alpha;
  c1=max(c1,0);
  c2=max(c2,0);
  s=c1+c2;
  c1=c1/s;
  c2=c2/s;
  k1=k1-k1_grad*alpha;
  k2=k2-k2_grad*alpha;
  k1=max(k1,0.0);
  k2=max(k2,0.0);
end

function [c1,c2,k1,k2]=sgd(data,num_restarts,LAMBDA)
  best_sse=Inf;
  best_params=[0,0,0,0];
  prev_sse=Inf;
  for i=0:num_restarts
    c1=rand;
    c2=1.0-c1;
    k1=rand;
    k2=rand*10.0;
    alpha=.01;
    for j=0:1000000
      rand_index=ceil(rand*size(data,1));
      t=data(rand_index,2);
      y=data(rand_index,1);
      [c1,c2,k1,k2]=gradient(c1,c2,k1,k2,t,y,alpha,LAMBDA);
      alpha=alpha*.9999;
      cur_sse=sse(c1,c2,k1,k2,data,LAMBDA);
      if cur_sse>prev_sse
	break
      end
      prev_sse=cur_sse;
    end
    if cur_sse<best_sse
      best_sse=cur_sse;
      best_params=[c1,c2,k1,k2];
    end
  end
  c1=best_params(1);
  c2=best_params(2);
  k1=best_params(3);
  k2=best_params(4);
end

