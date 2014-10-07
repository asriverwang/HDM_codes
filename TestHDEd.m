function [Result, ED, R2] = TestHDEd(Model)

    for f = 1:length(Model)
        
        tic
        for i=1:length(Model{f}.teX)
            tmpVA{i} = predict_HDEd(Model{f}.teX{i}, Model{f}.W, Model{f}.scale, Model{f}.res);
        end
        toc
        
        teVA = stackCell(tmpVA);
        
        diff = teVA - Model{f}.gtVA;
        diff = diff.^2;
        sdiss = sqrt(sum(diff, 2));        
        
        ED(f) = mean(sdiss);
        R2(f,:) = VAR2(teVA, Model{f}.gtVA);
        RMSVA(f,:) = sqrt(mean(diff));      
        tmpVA = stackCell(tmpVA);
        tmp = corrcoef(tmpVA(:,1)+.5, Model{f}.gtVA(:,1)+.5);
        r2(f,1) = tmp(1,2)^2;
        tmp = corrcoef(tmpVA(:,2)+.5, Model{f}.gtVA(:,2)+.5);
        r2(f,2) = tmp(1,2)^2;  
        clear tmpVA;

    end

    Result{1,1} = 'Euc Dist';
    Result{1,2} = 'Euc std';
    Result{1,3} = 'RMS V';
    Result{1,4} = 'RMS A';  
    Result{1,5} = 'R^2 V';
    Result{1,6} = 'R^2 A';  
    
    Result{2,1} = mean(ED);
    Result{2,2} = std(ED);
    RMS = mean(RMSVA);
    Result{2,3} = RMS(1);
    Result{2,4} = RMS(2);    
%     R2 = mean(R2);
%     Result{2,5} = R2(1);
%     Result{2,6} = R2(2);    

    r2 = mean(r2);
    Result{2,5} = r2(1);
    Result{2,6} = r2(2);   
end