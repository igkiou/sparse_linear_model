function result=merge(x,y);

result=[x(:); y(:)];
result=sort(result);
i=find(result-[0; result(1:end-1)]);
result=result(i);