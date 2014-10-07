function [trX, trY] = prepareDataHDEd(X, Label, scale, res, wlen)

N=length(Label);

for i=1:N
    Y{i} = genConserHDEVector(Label{i}, scale, res, wlen);
end

trX = stackCell(X);
trY = stackCellCell(Y);