%data_file=argv(1)
LAMBDA=1.55
data_file='../data/synthetic.csv';
num_restarts=100000;
data=csvread(data_file);
[c1,c2,k1,k2,k3]=sgd(data,num_restarts,LAMBDA);
c1
c2
k1
k2
k3
sse_val=sse(c1,c2,k1,k2,k3,data,LAMBDA)


function pred=predict(c1,c2,k1,k2,k3,t,LAMBDA)
  pred=(c1*(LAMBDA-k3)*exp(-t*(k1+k3))-k1*c1*exp(-LAMBDA*t))/(LAMBDA-(k1+k3))+(c2*(LAMBDA-k3)*exp(-t*(k2+k3))-k2*c2*exp(-LAMBDA*t))/(LAMBDA-(k2+k3));
end

function sse_val=sse(c1,c2,k1,k2,k3,data,LAMBDA)
  sse_val=0;
  for i=1:size(data,1)
    sse_val=sse_val+(predict(c1,c2,k1,k2,k3,data(i,1),LAMBDA)-data(i,2))^2;
  end
end

function [c1,c2,k1,k2,k3]=gradient(c1,c2,k1,k2,k3,t,y,alpha,LAMBDA)
  p=predict(c1,c2,k1,k2,k3,t,LAMBDA);
  c1_grad=(p-y)*((LAMBDA-k3)*exp(-t*(k1+k3))-k1*exp(-LAMBDA*t))/(LAMBDA-(k1+k3));
  c2_grad=(p-y)*((LAMBDA-k3)*exp(-t*(k2+k3))-k2*exp(-LAMBDA*t))/(LAMBDA-(k2+k3));
  k1_grad=(y-p)*c1*((exp(-LAMBDA*t)+t*(LAMBDA-k3)*exp(-t*(k1+k3)))/(LAMBDA-(k1+k3))+(k1*exp(-LAMBDA*t)-(LAMBDA-k3)*exp(-t*(k1+k3)))/(LAMBDA-(k1+k3))^2);
  k2_grad=(y-p)*c2*((exp(-LAMBDA*t)+t*(LAMBDA-k3)*exp(-t*(k2+k3)))/(LAMBDA-(k2+k3))+(k2*exp(-LAMBDA*t)-(LAMBDA-k3)*exp(-t*(k2+k3)))/(LAMBDA-(k2+k3))^2);
  k3_grad=(p-y)*(c1*((k3*t-LAMBDA*t-1)*exp(-t*(k1+k3))/(LAMBDA-(k1+k3))+((LAMBDA-k3)*exp(-t*(k1+k3))-k1*c1*exp(-LAMBDA*t))/(LAMBDA-k1-k3)^2)+c2*((k3*t-LAMBDA*t-1)*exp(-t*(k2+k3))/(LAMBDA-(k2+k3))+((LAMBDA-k3)*exp(-t*(k2+k3))-k2*c2*exp(-LAMBDA*t))/(LAMBDA-k2-k3)^2));
  c1=c1-c1_grad*alpha;
  c2=c2-c2_grad*alpha;
  c1=max(c1,0);
  c2=max(c2,0);
  s=c1+c2;
  c1=c1/s;
  c2=c2/s;
  k1=k1-k1_grad*alpha;
  k2=k2-k2_grad*alpha;
  k3=k3-k3_grad*alpha;
  k1=max(k1,0.0);
  k2=max(k2,0.0);
  k3=max(k3,0.0);
end

function [c1,c2,k1,k2,k3]=sgd(data,num_restarts,LAMBDA)
  best_sse=Inf;
  best_params=[0,0,0,0];
  prev_sse=Inf;
  for i=0:num_restarts
    c1=rand;
    c2=1.0-c1;
    k1=rand;
    k2=rand*10.0;
    k3=rand;
    alpha=.01;
    for j=0:1000000
      rand_index=ceil(rand*size(data,1));
      t=data(rand_index,2);
      y=data(rand_index,1);
      [c1,c2,k1,k2,k3]=gradient(c1,c2,k1,k2,k3,t,y,alpha,LAMBDA);
      alpha=alpha*.9999;
      cur_sse=sse(c1,c2,k1,k2,k3,data,LAMBDA);
      if cur_sse>prev_sse
	break
      end
      prev_sse=cur_sse;
    end
    if cur_sse<best_sse
      best_sse=cur_sse;
      best_params=[c1,c2,k1,k2,k3];
    end
  end
  c1=best_params(1);
  c2=best_params(2);
  k1=best_params(3);
  k2=best_params(4);
  k3=best_params(5);
end

