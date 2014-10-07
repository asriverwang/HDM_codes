function [trX, trY] = prepareData(X, Label, scale, res)

N=length(Label);

trX=[];
trY=[];  
for i=1:N
    trY=[trY; genHDEVector(Label{i}, scale, res)];
    trX=[trX; X{i}];
end
