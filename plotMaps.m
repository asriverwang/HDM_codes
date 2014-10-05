function plotMaps(W, res, M, N)

% M=4;
% N=8;

for i=1:size(W,1)
    Map{i}=reshape(W(i,:), res ,res);
end

count=1;
for i=1: M
for j=1: N
    if count>size(W,1)
        break;
    else
        subplot(M, N, count);
        imagesc(Map{count});
        set(gca, 'XTickLabel', {'','',''},'XTick',[-.5 0 .5], 'YTickLabel',{'','',''},'YTick',[-.5 0 5]);
        count=count+1;
    end
end
end
suplabel('Histogram Density Topics with K=128', 't');