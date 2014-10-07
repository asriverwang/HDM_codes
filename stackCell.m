function Xp = stackCell(X)

Xp = [];

for i =1:length(X)
    Xp = [Xp; X{i}];
end
    