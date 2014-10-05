
for k=1:length(RPC)
rp=RPC{k};
for t=1:7
for i=1: length(id)
    te = rp(1:72);
    tr = rp(73:end);
    [chk, trf] = ismember(id{i}, tr);
    if sum(chk)==1
        trf=sum(trf);
        tmp = tr(trf);
        rp(trf+72)=[];
        rp = [tmp, rp];
    end
end
end
RP{k}=rp;
end