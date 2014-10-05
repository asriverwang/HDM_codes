function Y = genHDEVector(Label, scale, resolution)

step = abs(scale(2)-scale(1))/resolution;
edges = scale(1):step:scale(2);
edges(end) = edges(end)+0.1;


for s=1:length(Label)
    
    Map = zeros(resolution);
    
    sl=Label{s};
    
    idx = find(sl<scale(1));
    if ~isempty(idx)
        sl(idx)=scale(1);      
    end
    idx = find(sl>scale(2));
    if ~isempty(idx)    
        sl(idx)=scale(2);
    end
  
    [~, bin]=histc(sl, edges);

    for i=1: size(Label{s},1);
        x = bin(i,1);
        y = bin(i,2);
        Map(x,y) = Map(x,y)+1;
    end

    Map=rot90(Map);
    Y(s,:) = reshape(Map, 1, resolution*resolution);
end

Y = Y ./ repmat(sum(Y,2), 1, resolution*resolution);

