%data_file=argv(1)
data_file='../data/synthetic.csv';
num_restarts=100000;
data=csvread(data_file);
[c1,c2,k1,k2]=sgd(data,num_restarts);
c1
c2
k1
k2
sse_val=sse(c1,c2,k1,k2,data)

function pred=predict(c1,c2,k1,k2,t)
  pred=c1*exp(-t*k1)+c2*exp(-t*k2);
end

function sse_val=sse(c1,c2,k1,k2,data)
  sse_val=0;
  for i=1:size(data,1)
    sse_val=sse_val+(predict(c1,c2,k1,k2,data(i,1))-data(i,2))^2;
  end
end

function [c1,c2,k1,k2]=gradient(c1,c2,k1,k2,t,y,alpha)
  c1_grad=-2*y*exp(-t*k1)-2*c1*exp(-2*t*k1)-2*c2*exp(-t*(k1+k2));
  c2_grad=-2*y*exp(-t*k2)-2*c2*exp(-2*t*k2)-2*c1*exp(-t*(k1+k2));
  k1_grad=2*y*c1*t*exp(-t*k1)+2*t*c1*exp(-2*t*k1)+2*t*c1*c2*exp(-t*(k1+k2));
  k2_grad=2*y*c2*t*exp(-t*k2)+2*t*c2*exp(-2*t*k2)+2*t*c1*c2*exp(-t*(k1+k2));
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

function [c1,c2,k1,k2]=sgd(data,num_restarts)
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
      [c1,c2,k1,k2]=gradient(c1,c2,k1,k2,t,y,alpha);
      alpha=alpha*.9999;
      cur_sse=sse(c1,c2,k1,k2,data);
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

