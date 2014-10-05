function pVA = predict_HDEd(teX, W, scale, res)

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

    VA{t} = S*C';
end

w=[];
wc=1;
dlen = floor(length(W)/2);
for i=1:dlen+1
    w = [w wc];
    wc = wc/2;
end
w = [fliplr(w(2:end)), w];

pVA = 0;
for t=1:length(W)
    pVA = pVA + w(t) * [ zeros(t-1, 2); VA{t}; zeros(2*dlen-t+1, 2)];    
end

pVA = pVA(dlen+1:end-dlen,:)/sum(w);

