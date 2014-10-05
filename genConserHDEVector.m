function Y = genConserHDEVector(Label, scale, resolution, dlen)

sec = length(Label);

if size(Label,1)>1
    Label=Label';
end

Label = [repmat(Label(1), 1, dlen), Label, repmat(Label(end), 1, dlen)];

Y = cell(1, 2*dlen+1);
for d=1:2*dlen+1
    conLabel = Label(d:d+sec-1); 
    Y{d} = genHDEVector(conLabel, scale, resolution);
end



    
