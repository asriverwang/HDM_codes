function VA = predict_HDE(teX, W, scale, res)

step = abs(scale(2)-scale(1))/res;
cen = scale(1):step:scale(2);
cen=cen(1:res);
cen = cen+ (step/2);

M = repmat(cen, res, 1);
C(1,:) = reshape(M, 1, res*res);
M = flipud(M');
C(2,:) = reshape(M, 1, res*res);

for t=1: length(W)
    S = teX*W{t};
    S= S./ repmat(sum(S,2), 1, size(S,2));

    VA(:,:,t) = S*C';
end

VA = mean(VA,3);

