function Yp = stackCellCell(Y)

Yp = cell(1, length(Y{1}));

for i =1:length(Y)
    for d =1:length(Y{1})
        Yp{d} = [Yp{d}; Y{i}{d}];
    end
end
    