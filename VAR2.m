function R2 = VAR2(m, gm)

% for i=1:length(X)
%     m(i,:) = X{i}.mu;
%     gm(i,:) = Y{i}.mu;
% end

d = (m - gm).^2;
d = sum(d,1);

nd = (gm - repmat(mean(gm), size(m,1), 1)).^2;
nd = sum(nd,1);

R2(1) = 1 - (d(1)/nd(1));
R2(2) = 1 - (d(2)/nd(2));


