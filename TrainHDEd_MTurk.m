function Model = TrainHDEd_MTurk(X, Label, RP, scale, res, dlen)
% X: acoustic posterior N-by-K
% Label: annotations in each cell
% RP: a set of random permutation for train-test spliting
% res: resolution to make a res-by-res grid over the VA space

Model = cell(length(RP),1);
N=length(Label);
splitN=round(N*.7);

for i=1:N
    Y{i} = genConserHDEVector(Label{i}, scale, res, dlen);
    for j=1:length(Label{i})
        gtVA{i}(j,:) = mean(Label{i}{j});
    end
end
    
for f=1:length(RP)
    
    Model{f}.trIdx = RP{f}(1:splitN);
    Model{f}.teIdx = RP{f}(splitN+1:end);
        
    Model{f}.scale = scale;
    Model{f}.res = res;

% -----  record the test information  ----- %    
    Model{f}.teX = X(Model{f}.teIdx);
    Model{f}.gtVA = stackCell(gtVA(Model{f}.teIdx));
    Model{f}.teY = stackCell(Y(Model{f}.teIdx));
% ----------------------------------------- %     
    
    trX = stackCell(X(Model{f}.trIdx));
    trY = stackCellCell(Y(Model{f}.trIdx));
    
    tic
    Model{f}.W = HDEd_train(trX, trY);  % train the model   
    toc

end